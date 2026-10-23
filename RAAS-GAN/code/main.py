from __future__ import print_function
import argparse
import copy
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from DAMSM import RNN_ENCODER, MogLSTM, CNN_ENCODER
from datasets import TextDataset, prepare_data, prepare_labels
from miscc.config import cfg, cfg_from_file
from miscc.losses import sent_loss, words_loss
from miscc.utils import mkdir_p
from model import NetD, NetGFixedDARTS

def load_darts_arch_file(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is not JSON and PyYAML is not installed; "
                "export_darts_topk.py writes JSON-compatible YAML by default."
            ) from exc
        payload = yaml.safe_load(text)
    return payload.get("arch", payload)

def parse_args():
    parser = argparse.ArgumentParser(description="RAAS-GAN final full retraining")
    parser.add_argument('--cfg', dest='cfg_file', required=True)
    parser.add_argument('--darts_arch', required=True)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--seed', type=int, default=223)
    parser.add_argument('--train_workers', type=int, default=None)
    parser.add_argument('--ma_gp_chunk', type=int, default=0,
                        help='MA-GP microbatch size; 0 uses the full batch, without changing the objective.')
    parser.add_argument('--output', required=True, help='Final EMA generator .pth path')
    return parser.parse_args()

def _set_module_requires_grad(module, flag: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(flag)

def train(dataloader, netG, netD, text_encoder,
          optimizerG, optimizerD, batch_size, device, args):
    from torch.amp import autocast, GradScaler
    amp_dtype = torch.bfloat16
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Final full retraining requires a BF16-capable GPU")
    scalerG = GradScaler("cuda", enabled=False)
    scalerD = GradScaler("cuda", enabled=False)
    EMPTY_CACHE_EVERY = 300
    LOG_EVERY = 100
    CLIP_NORM_G = 1.0
    CLIP_NORM_D = 1.0
    EMA_DECAY = 0.99995
    PTCDFM_EVERY = 1

    def _ptcdfm_d_weight(epoch_idx):
        return 0.10 * min(1.0, float(epoch_idx) / 10.0)

    def _ptcdfm_g_weight(epoch_idx):
        return 0.05 * min(1.0, float(epoch_idx) / 10.0)

    def _ptcdfm_fm_weight(epoch_idx):
        return 0.03 * min(1.0, float(epoch_idx) / 10.0)

    def _tcdfm_feature_matching_loss(real_feats, fake_feats):
        real_sel = list(real_feats)[-2:]
        fake_sel = list(fake_feats)[-2:]
        if not real_sel or not fake_sel:
            return torch.tensor(0.0, device=fake_feats[-1].device if fake_feats else device)
        loss = torch.tensor(0.0, device=fake_sel[-1].device)
        used = 0
        for real_f, fake_f in zip(real_sel, fake_sel):
            real_stat = real_f.detach().float().mean(dim=(0, 2, 3))
            fake_stat = fake_f.float().mean(dim=(0, 2, 3))
            loss = loss + F.l1_loss(fake_stat, real_stat)
            used += 1
        return loss / max(1, used)

    def _ptcdfm_d_loss(ptc_head, cnn_real, cnn_fake, sent_cond):
        if ptc_head is None:
            return torch.tensor(0.0, device=sent_cond.device)
        sent = sent_cond.detach()
        score_real = ptc_head(cnn_real.detach(), sent)
        score_fake = ptc_head(cnn_fake.detach(), sent)
        loss_real = torch.relu(1.0 - score_real)
        loss_fake = torch.relu(1.0 + score_fake)
        loss = loss_real.mean() + 0.5 * loss_fake.mean()
        if cnn_real.size(0) > 1:
            score_mis = ptc_head(cnn_real[:-1].detach(), sent[1:cnn_real.size(0)])
            loss_mis = torch.relu(1.0 + score_mis)
            loss = loss + 0.5 * loss_mis.mean()
        return loss

    def _get_model(m):
        return m.module if isinstance(m, nn.DataParallel) else m

    netG_base = _get_model(netG)
    netG_ema = copy.deepcopy(netG_base).to(device)
    for p in netG_ema.parameters():
        p.requires_grad_(False)
    netG_ema.eval()
    netG_ema.load_state_dict(netG_base.state_dict(), strict=True)

    @torch.no_grad()
    def ema_update(ema_model, model, decay: float):
        msd = model.state_dict()
        esd = ema_model.state_dict()
        for k, v in esd.items():
            if k in msd:
                esd[k].mul_(decay).add_(msd[k], alpha=(1.0 - decay))
        ema_model.load_state_dict(esd, strict=False)

    _, _, match_labels = prepare_labels(batch_size)
    mG = _get_model(netG)
    mD = _get_model(netD)
    mG.train()
    mD.train()

    for epoch in range(1, cfg.TRAIN.MAX_EPOCH + 1):
        epoch_t0 = time.time()
        lrG, lrD = 1.0e-4, 4.0e-4
        damsm_w = 0.1
        for step, data in enumerate(dataloader, 0):
            if (step % EMPTY_CACHE_EVERY) == 0 and step > 0:
                torch.cuda.empty_cache()

            imgs_list, captions, cap_lens, class_ids, keys = prepare_data(data)

            imgs = imgs_list[-1].to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            cap_lens = cap_lens.to(device, non_blocking=True)

            # text enc (frozen)
            with torch.no_grad():
                hidden = text_encoder.init_hidden(batch_size)
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs = words_embs.detach()
                sent_emb = sent_emb.detach()

            sent_cond = sent_emb

            # ===== G forward =====
            z = torch.randn(batch_size, 100, device=device)

            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                fake, attention = netG(z, sent_cond, words_embs, mask)

            if not torch.isfinite(fake).all():
                optimizerG.zero_grad(set_to_none=True)
                optimizerD.zero_grad(set_to_none=True)
                continue


            real_in = imgs
            fake_in = fake.detach()

            # ===== train D =====
            errD = torch.tensor(0.0, device=device)
            optimizerD.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                real_features = netD(real_in)
                out_real = mD.COND_DNET(real_features, sent_cond.detach())
                errD_real = torch.relu(1.0 - out_real).mean()

                out_mismatch = mD.COND_DNET(real_features[:batch_size - 1], sent_cond.detach()[1:batch_size])
                errD_mismatch = torch.relu(1.0 + out_mismatch).mean()

                fake_features = netD(fake_in)
                out_fake = mD.COND_DNET(fake_features, sent_cond.detach())
                errD_fake = torch.relu(1.0 + out_fake).mean()

                errD = errD_real + 0.5 * errD_fake + 0.5 * errD_mismatch


                ptcdfm_dw = _ptcdfm_d_weight(epoch)
                if (
                    ptcdfm_dw > 0.0
                    and ((step % PTCDFM_EVERY) == 0)
                    and getattr(mD, "PTC_DNET", None) is not None
                ):
                    with torch.no_grad():
                        _, cnn_real_ptc = image_encoder(real_in)
                        _, cnn_fake_ptc = image_encoder(fake_in)
                    ptcdfm_d_loss = _ptcdfm_d_loss(
                        mD.PTC_DNET,
                        cnn_real_ptc,
                        cnn_fake_ptc,
                        sent_cond,
                    )
                    errD = errD + ptcdfm_dw * ptcdfm_d_loss


                # This is train-time only and does not alter generator inference.

                errD_total = errD
            if torch.isfinite(errD_total):
                scalerD.scale(errD_total).backward()
                scalerD.unscale_(optimizerD)
                torch.nn.utils.clip_grad_norm_(mD.parameters(), CLIP_NORM_D)
                scalerD.step(optimizerD)
                scalerD.update()
            optimizerD.zero_grad(set_to_none=True)
            gp_chunk = args.ma_gp_chunk if args.ma_gp_chunk > 0 else batch_size
            gp_chunk = max(1, min(int(gp_chunk), batch_size))
            original_ma_gp_total = torch.tensor(0.0, device=device)
            original_ma_gp_ok = True
            for gp_start in range(0, batch_size, gp_chunk):
                gp_end = min(batch_size, gp_start + gp_chunk)
                scale = float(gp_end - gp_start) / float(batch_size)
                imgs_gp = imgs[gp_start:gp_end].detach().float().requires_grad_(True)
                sent_gp = sent_cond[gp_start:gp_end].detach().float().requires_grad_(True)
                with autocast(device_type=device.type, enabled=False):
                    real_features_gp = netD(imgs_gp)
                    out_real_gp = mD.COND_DNET(real_features_gp, sent_gp)
                    grad_img, grad_sent = torch.autograd.grad(
                        out_real_gp,
                        (imgs_gp, sent_gp),
                        grad_outputs=torch.ones_like(out_real_gp),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )
                    grad = torch.cat(
                        (
                            grad_img.reshape(grad_img.size(0), -1),
                            grad_sent.reshape(grad_sent.size(0), -1),
                        ),
                        dim=1,
                    )
                    grad_l2norm = torch.sqrt(torch.sum(grad.pow(2), dim=1) + 1.0e-12)
                    original_ma_gp = 2.0 * torch.mean(grad_l2norm.pow(6)) * scale

                if torch.isfinite(original_ma_gp):
                    original_ma_gp_total = original_ma_gp_total + original_ma_gp.detach()
                    scalerD.scale(original_ma_gp).backward()
                else:
                    original_ma_gp_ok = False
                    break
            if original_ma_gp_ok:
                scalerD.step(optimizerD)
                scalerD.update()

            # ===== train G =====
            optimizerG.zero_grad(set_to_none=True)

            region_fake_G = None
            cnn_fake_G = None
            ptcdfm_fm_w = _ptcdfm_fm_weight(epoch) if ((step % PTCDFM_EVERY) == 0) else 0.0

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                features_fake_for_G, fake_d_feats_for_G = netD(fake, return_features=True)
                out_fake_for_G = mD.COND_DNET(features_fake_for_G, sent_cond)
                errG = (-out_fake_for_G.mean())


                if fake_d_feats_for_G is not None:
                    with torch.no_grad():
                        _, real_d_feats_for_G = netD(imgs, return_features=True)
                    if ptcdfm_fm_w > 0.0:
                        ptcdfm_fm_loss = _tcdfm_feature_matching_loss(
                            real_d_feats_for_G,
                            fake_d_feats_for_G,
                        )
                        errG = errG + ptcdfm_fm_w * ptcdfm_fm_loss

                # DAMSM
                damsm_every = 1
                if damsm_w > 0 and ((step % damsm_every) == 0):
                    region_fake_G, cnn_fake_G = image_encoder(fake)
                    w_loss0, w_loss1, _ = words_loss(region_fake_G, words_embs, match_labels, cap_lens, class_ids, batch_size)
                    w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
                    s_loss0, s_loss1 = sent_loss(cnn_fake_G, sent_cond, match_labels, class_ids, batch_size)
                    s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
                    errG = errG + damsm_w * (w_loss + s_loss)


                # Freeze the PTC head; gradients still reach G through image features.


                ptcdfm_gw = _ptcdfm_g_weight(epoch) if ((step % PTCDFM_EVERY) == 0) else 0.0
                if ptcdfm_gw > 0.0 and getattr(mD, "PTC_DNET", None) is not None:
                    if cnn_fake_G is None:
                        _, cnn_fake_G = image_encoder(fake)
                    _set_module_requires_grad(mD.PTC_DNET, False)
                    ptcdfm_fake_score = mD.PTC_DNET(cnn_fake_G, sent_cond)
                    errG = errG + ptcdfm_gw * (-ptcdfm_fake_score.mean())
                    _set_module_requires_grad(mD.PTC_DNET, True)


            if torch.isfinite(errG):
                scalerG.scale(errG).backward()
                scalerG.unscale_(optimizerG)
                torch.nn.utils.clip_grad_norm_(mG.parameters(), CLIP_NORM_G)
                scalerG.step(optimizerG)
                scalerG.update()

                ema_update(netG_ema, mG, EMA_DECAY)

            if LOG_EVERY and (step % LOG_EVERY == 0):
                d_val = float(errD.detach().item()) if torch.isfinite(errD).all() else float("nan")
                g_val = float(errG.detach().item()) if torch.isfinite(errG).all() else float("nan")
                print(f"[e{epoch}][{step}/{len(dataloader)}] "
                      f"D={d_val:.3f} G={g_val:.3f} lrG={lrG:.1e} lrD={lrD:.1e} "
                      f"PTFM=enabled", flush=True)

        epoch_sec = time.time() - epoch_t0
        steps_per_epoch = max(1, len(dataloader))
        print(
            f"[EPOCH_TIME] epoch={epoch} sec={epoch_sec:.1f} min={epoch_sec / 60.0:.2f} "
            f"steps={len(dataloader)} sec_per_step={epoch_sec / steps_per_epoch:.3f} "
            f"batch_size={batch_size}",
            flush=True,
        )

    # Only the final EMA generator is exported; no resumable training state.
    output = os.path.abspath(args.output)
    mkdir_p(os.path.dirname(output))
    torch.save(netG_ema.state_dict(), output)
    print(f"[FINAL_EMA] {output}", flush=True)
    return 0

if __name__ == "__main__":
    args = parse_args()
    cfg_from_file(args.cfg_file)
    cfg.GPU_ID = args.gpu_id
    cfg.CUDA = args.gpu_id >= 0 and torch.cuda.is_available()
    if args.data_dir:
        cfg.DATA_DIR = args.data_dir
    if args.train_workers is not None:
        cfg.WORKERS = args.train_workers
    if args.ma_gp_chunk < 0 or cfg.TRAIN.BATCH_SIZE < 2:
        raise ValueError('MA-GP chunk must be nonnegative; training batch size must be >= 2')
    if os.path.exists(args.output):
        raise FileExistsError(f'Refusing to overwrite final EMA output: {args.output}')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cfg.CUDA:
        torch.cuda.set_device(cfg.GPU_ID)
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    train_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.CenterCrop(imsize),
    ])
    dataset = TextDataset(cfg.DATA_DIR, 'train',
                        base_size=cfg.TREE.BASE_SIZE,
                        transform=train_transform)
    assert dataset
    nw = int(cfg.WORKERS)
    if nw > 0:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=nw,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
    print(
        f"[DATALOADER] workers={nw} pin_memory=True "
        f"persistent={nw > 0} prefetch={4 if nw > 0 else 'NA'}",
        flush=True
    )
    lstm = MogLSTM(256, 256, 5)
    arch = load_darts_arch_file(args.darts_arch)
    netG = NetGFixedDARTS(arch=arch, ngf=cfg.TRAIN.NF, nz=100, lstm=lstm).to(device)
    netD = NetD(cfg.TRAIN.NF, use_ptcdfm=True).to(device)
    if cfg.CUDA and torch.cuda.device_count() > 1:
        netG = torch.nn.DataParallel(netG)
        netD = torch.nn.DataParallel(netD)
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.to(device)


    #####20220601 idea-3 use DAMSM Loss #####
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    img_encoder_path = cfg.TEXT.DAMSM_IMGNAME
    state_dict = \
        torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict)
    image_encoder.to(device)

    for p in image_encoder.parameters():
        p.requires_grad = False
    print('Load image encoder from:', img_encoder_path)
    image_encoder.eval()
    #####20220601 idea-3 use DAMSM Loss #####

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=1.0e-4, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=4.0e-4, betas=(0.0, 0.9))
    train(dataloader, netG, netD, text_encoder,
          optimizerG, optimizerD, batch_size, device, args)
