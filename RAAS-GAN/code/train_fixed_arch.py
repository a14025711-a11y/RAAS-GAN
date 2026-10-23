from __future__ import annotations
import argparse
import copy
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
    generator_proxy_loss,
    load_cfg,
    load_json,
    make_device,
    make_lstm,
    original_ma_gp_penalty,
    prepare_batch,
    save_json,
    set_requires_grad,
    set_seed,
)
from miscc.config import cfg
from model import NetD, NetGFixedDARTS
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Short-train a fixed DARTS architecture")
    p.add_argument("--cfg", required=True)
    p.add_argument("--arch", required=True)
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
        help="Must match the supernet train/validation split seed.",
    )
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        msd = model.state_dict()
        esd = ema_model.state_dict()
        for k, v in esd.items():
            if k in msd:
                v.mul_(decay).add_(msd[k], alpha=1.0 - decay)
        ema_model.load_state_dict(esd, strict=False)
def main() -> None:
    args = parse_args()
    load_cfg(args.cfg, batch_size=args.batch_size, workers=args.workers, gpu=args.gpu)
    set_seed(args.seed)
    device = make_device(args.gpu)
    batch_size = int(cfg.TRAIN.BATCH_SIZE)
    arch_payload = load_json(args.arch)
    arch = arch_payload.get("arch", arch_payload)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stem = Path(args.arch).stem
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("darts_runs") / f"fixed_{stem}_{stamp}"
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "arch.json", {"arch": arch, "source": str(args.arch)})
    save_json(out_dir / "config.json", {"args": vars(args)})

    split_seed = int(args.seed if args.split_seed is None else args.split_seed)
    dataset, train_loader, _ = build_train_val_loaders(
        val_fraction=args.val_fraction,
        seed=split_seed,
        batch_size=batch_size,
        workers=cfg.WORKERS,
    )

    lstm = make_lstm(device)
    netG = NetGFixedDARTS(
        arch=arch,
        ngf=cfg.TRAIN.NF,
        nz=100,
        lstm=lstm,
        unit_alpha_init=1.0e-3,
    ).to(device)
    netD = NetD(
        cfg.TRAIN.NF,
    ).to(device)
    netG_ema = copy.deepcopy(netG).to(device)
    netG_ema.eval()

    text_encoder = build_text_encoder(dataset.n_words, device)
    image_encoder = build_image_encoder(device) if 0.1 > 0 else None

    optG = torch.optim.Adam(netG.parameters(), lr=1.0e-4, betas=tuple((0.0, 0.9)))
    optD = torch.optim.Adam(netD.parameters(), lr=4.0e-4, betas=tuple((0.0, 0.9)))
    use_amp = bool(True and device.type == "cuda")
    scalerG = GradScaler("cuda", enabled=use_amp)
    scalerD = GradScaler("cuda", enabled=use_amp)
    match_labels = torch.arange(batch_size, device=device, dtype=torch.long)
    global_step = 0

    print(f"[FIXED] out={out_dir} arch={netG.genotype()} bs={batch_size} epochs={30}", flush=True)
    print(
        f"[PROTOCOL] lrG={1.0e-4:.1e} lrD={4.0e-4:.1e} Adam={tuple((0.0, 0.9))} "
        f"D_every=1 gp=MA-GP DAMSM={0.1} every_step "
        f"mismatch={0.5} ema={0.99995}",
        flush=True,
    )
    for epoch in range(1, int(30) + 1):
        t0 = time.time()
        netG.train()
        netD.train()
        last_row = None
        for step, data in enumerate(train_loader):
            global_step += 1
            batch = prepare_batch(data, text_encoder, batch_size, device)

            set_requires_grad(netD, True)
            set_requires_grad(netG, False)
            optD.zero_grad(set_to_none=True)
            with torch.no_grad():
                z = torch.randn(batch_size, 100, device=device)
                with autocast(device_type=device.type, enabled=use_amp):
                    fake_d, _ = netG(
                        z, batch["sent_emb"], batch["words_embs"], batch["mask"],
                    )
            with autocast(device_type=device.type, enabled=use_amp):
                d_loss = discriminator_hinge_loss(
                    netD,
                    batch["imgs"],
                    fake_d,
                    batch["sent_emb"],
                    mismatch_weight=0.5,
                )
            scalerD.scale(d_loss).backward()
            scalerD.unscale_(optD)
            torch.nn.utils.clip_grad_norm_(netD.parameters(), 1.0)
            scalerD.step(optD)
            scalerD.update()
            optD.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=False):
                ma_gp = original_ma_gp_penalty(netD, batch["imgs"], batch["sent_emb"])
            scalerD.scale(ma_gp).backward()
            scalerD.step(optD)
            scalerD.update()

            set_requires_grad(netD, False)
            set_requires_grad(netG, True)
            optG.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, 100, device=device)
            with autocast(device_type=device.type, enabled=use_amp):
                fake_g, _ = netG(z, batch["sent_emb"], batch["words_embs"], batch["mask"])
                g_loss, g_logs = generator_proxy_loss(
                    netD,
                    image_encoder,
                    fake_g,
                    batch["sent_emb"],
                    batch["words_embs"],
                    batch["cap_lens"],
                    batch["class_ids"],
                    batch_size,
                    match_labels,
                    0.1,
                )
            scalerG.scale(g_loss).backward()
            scalerG.unscale_(optG)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 1.0)
            scalerG.step(optG)
            scalerG.update()
            update_ema(netG_ema, netG, float(0.99995))

            last_row = {
                "epoch": epoch,
                "step": step,
                "global_step": global_step,
                "loss_d": float(d_loss.detach()),
                "loss_g": g_logs,
            }
            if step % int(args.log_every) == 0:
                print(
                    f"[E{epoch:03d} S{step:04d}] D={last_row['loss_d']:.3f} G={g_logs['total']:.3f}",
                    flush=True,
                )

        row = {
            "epoch": epoch,
            "wall_time_min": (time.time() - t0) / 60.0,
            "last": last_row,
            "arch": netG.genotype(),
        }
        append_jsonl(out_dir / "train_history.jsonl", row)
        print(f"[EPOCH_DONE] epoch={epoch} time_min={row['wall_time_min']:.2f}", flush=True)

    atomic_torch_save(netG_ema.state_dict(), out_dir / "netGEMA_final.pth")
    print(f"[DONE] netGEMA={out_dir / 'netGEMA_final.pth'}", flush=True)
if __name__ == "__main__":
    main()
