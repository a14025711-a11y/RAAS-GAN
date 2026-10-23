from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from torch.amp import GradScaler, autocast
from darts.train_utils import (
    append_jsonl,
    atomic_torch_save,
    build_image_encoder,
    build_text_encoder,
    build_train_val_loaders,
    discriminator_hinge_loss,
    endless,
    generator_proxy_loss,
    load_cfg,
    make_device,
    make_lstm,
    original_ma_gp_penalty,
    prepare_batch,
    save_json,
    set_params_requires_grad,
    set_requires_grad,
    set_seed,
)
from miscc.config import cfg
from model import NetD, NetGDARTS
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("RAAS supernet search")
    p.add_argument("--cfg", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", "--bs", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=223)
    p.add_argument("--out_dir", default="")
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument(
        "--split_seed",
        type=int,
        default=2026,
        help="Seed used only to construct the fixed train/validation split.",
    )
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()
def main() -> None:
    args = parse_args()
    load_cfg(args.cfg, batch_size=args.batch_size, workers=args.workers, gpu=args.gpu)
    set_seed(args.seed)
    device = make_device(args.gpu)
    batch_size = int(cfg.TRAIN.BATCH_SIZE)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("darts_runs") / f"search_{stamp}"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", {"args": vars(args), "cfg": json.loads(json.dumps(cfg))})

    dataset, train_loader, val_loader = build_train_val_loaders(
        val_fraction=args.val_fraction,
        seed=args.split_seed,
        batch_size=batch_size,
        workers=cfg.WORKERS,
    )
    val_iter = endless(val_loader)

    lstm = make_lstm(device)
    netG = NetGDARTS(
        cfg.TRAIN.NF,
        100,
        lstm,
        unit_alpha_init=1.0e-3,
        arch_tau=1.0,
    ).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)

    text_encoder = build_text_encoder(dataset.n_words, device)
    image_encoder = build_image_encoder(device) if 0.1 > 0 else None

    g_weight_params = list(netG.weight_parameters())
    arch_params = list(netG.arch_parameters())
    optG = torch.optim.Adam(g_weight_params, lr=1.0e-4, betas=tuple((0.0, 0.9)))
    optD = torch.optim.Adam(netD.parameters(), lr=4.0e-4, betas=tuple((0.0, 0.9)))
    optA = torch.optim.Adam(arch_params, lr=3.0e-4, betas=(0.5, 0.999))

    use_amp = bool(True and device.type == "cuda")
    scalerG = GradScaler("cuda", enabled=use_amp)
    scalerD = GradScaler("cuda", enabled=use_amp)
    scalerA = GradScaler("cuda", enabled=use_amp)
    match_labels = torch.arange(batch_size, device=device, dtype=torch.long)
    total_epochs = int(5 + 25)
    global_step = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(
        f"[DARTS] out={out_dir} bs={batch_size} warmup={5} "
        f"search={25} device={device}",
        flush=True,
    )
    print(
        f"[PROTOCOL] lrG={1.0e-4:.1e} lrD={4.0e-4:.1e} Adam={tuple((0.0, 0.9))} "
        f"D_every=1 gp=MA-GP DAMSM={0.1} every_step "
        f"mismatch={0.5} lr_alpha={3.0e-4:.1e}",
        flush=True,
    )

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()
        netG.train()
        netD.train()
        is_search_epoch = epoch > int(5)
        epoch_logs = []

        for step, train_data in enumerate(train_loader):
            global_step += 1
            train_batch = prepare_batch(train_data, text_encoder, batch_size, device)
            z = torch.randn(batch_size, 100, device=device)

            # 1) D update: fake generation does not keep the G graph.
            set_requires_grad(netD, True)
            set_params_requires_grad(g_weight_params, False)
            set_params_requires_grad(arch_params, False)
            optD.zero_grad(set_to_none=True)
            with torch.no_grad():
                with autocast(device_type=device.type, enabled=use_amp):
                    fake_d, _ = netG(z, train_batch["sent_emb"], train_batch["words_embs"], train_batch["mask"])
            with autocast(device_type=device.type, enabled=use_amp):
                d_loss = discriminator_hinge_loss(
                    netD,
                    train_batch["imgs"],
                    fake_d,
                    train_batch["sent_emb"],
                    mismatch_weight=0.5,
                )
            scalerD.scale(d_loss).backward()
            scalerD.unscale_(optD)
            torch.nn.utils.clip_grad_norm_(netD.parameters(), 1.0)
            scalerD.step(optD)
            scalerD.update()
            optD.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=False):
                ma_gp = original_ma_gp_penalty(netD, train_batch["imgs"], train_batch["sent_emb"])
            scalerD.scale(ma_gp).backward()
            scalerD.step(optD)
            scalerD.update()

            # 2) G weight update: D is fixed, alpha participates but is not updated.
            set_requires_grad(netD, False)
            set_params_requires_grad(g_weight_params, True)
            set_params_requires_grad(arch_params, False)
            optG.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, 100, device=device)
            with autocast(device_type=device.type, enabled=use_amp):
                fake_g, _ = netG(z, train_batch["sent_emb"], train_batch["words_embs"], train_batch["mask"])
                g_loss, g_logs = generator_proxy_loss(
                    netD,
                    image_encoder,
                    fake_g,
                    train_batch["sent_emb"],
                    train_batch["words_embs"],
                    train_batch["cap_lens"],
                    train_batch["class_ids"],
                    batch_size,
                    match_labels,
                    0.1,
                )
            scalerG.scale(g_loss).backward()
            scalerG.unscale_(optG)
            torch.nn.utils.clip_grad_norm_(g_weight_params, 1.0)
            scalerG.step(optG)
            scalerG.update()

            alpha_logs = {"total": 0.0, "adv": 0.0, "damsm": 0.0}
            if is_search_epoch:
                # 3) Alpha update: G/D weights are fixed, only architecture params move.
                val_batch = prepare_batch(next(val_iter), text_encoder, batch_size, device)
                set_requires_grad(netD, False)
                set_params_requires_grad(g_weight_params, False)
                set_params_requires_grad(arch_params, True)
                optA.zero_grad(set_to_none=True)
                z = torch.randn(batch_size, 100, device=device)
                with autocast(device_type=device.type, enabled=use_amp):
                    fake_a, _ = netG(z, val_batch["sent_emb"], val_batch["words_embs"], val_batch["mask"])
                    a_loss, alpha_logs = generator_proxy_loss(
                        netD,
                        image_encoder,
                        fake_a,
                        val_batch["sent_emb"],
                        val_batch["words_embs"],
                        val_batch["cap_lens"],
                        val_batch["class_ids"],
                        batch_size,
                        match_labels,
                        0.1,
                    )
                scalerA.scale(a_loss).backward()
                scalerA.unscale_(optA)
                torch.nn.utils.clip_grad_norm_(arch_params, 1.0)
                scalerA.step(optA)
                scalerA.update()

            row = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "phase": "search" if is_search_epoch else "warmup",
                "loss_d": float(d_loss.detach()),
                "loss_g": g_logs,
                "loss_alpha": alpha_logs,
            }
            epoch_logs.append(row)
            if step % int(args.log_every) == 0:
                print(
                    f"[E{epoch:03d} S{step:04d}] "
                    f"D={row['loss_d']:.3f} G={g_logs['total']:.3f} A={alpha_logs['total']:.3f}",
                    flush=True,
                )

        summary = {
            "epoch": epoch,
            "phase": "search" if is_search_epoch else "warmup",
            "wall_time_min": (time.time() - t0) / 60.0,
            "alpha": netG.alpha_summary(),
            "last": epoch_logs[-1] if epoch_logs else {},
        }
        append_jsonl(out_dir / "alpha_history.jsonl", summary)
        print(
            f"[EPOCH_DONE] epoch={epoch} phase={summary['phase']} "
            f"time_min={summary['wall_time_min']:.2f} genotype={netG.genotype()}",
            flush=True,
        )

    # RAAS needs only six architecture-logit vectors and its alpha history.
    alpha_state = {key: value.detach().cpu() for key, value in netG.state_dict().items()
                   if key.endswith("alpha_visual") or key.endswith("alpha_text")}
    atomic_torch_save({"netG": alpha_state}, out_dir / "latest.pth")
    print(f"[DONE] latest={out_dir / 'latest.pth'}", flush=True)
if __name__ == "__main__":
    main()
