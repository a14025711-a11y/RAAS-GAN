from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from DAMSM import CNN_ENCODER, MogLSTM, RNN_ENCODER
from datasets import TextDataset, prepare_data
from miscc.config import cfg, cfg_from_file
from miscc.losses import sent_loss, words_loss
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def load_cfg(cfg_file: str, batch_size: int | None = None, workers: int | None = None, gpu: int | None = None) -> None:
    cfg_from_file(cfg_file)
    if batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = int(batch_size)
    if workers is not None:
        cfg.WORKERS = int(workers)
    if gpu is not None:
        cfg.GPU_ID = int(gpu)
    cfg.CUDA = torch.cuda.is_available() and int(cfg.GPU_ID) >= 0
def make_device(gpu: int | None = None) -> torch.device:
    if torch.cuda.is_available() and (gpu is None or int(gpu) >= 0):
        if gpu is not None:
            torch.cuda.set_device(int(gpu))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    cfg.CUDA = False
    return torch.device("cpu")
def build_train_val_loaders(
    val_fraction: float,
    seed: int,
    batch_size: int | None = None,
    workers: int | None = None,
):
    imsize = cfg.TREE.BASE_SIZE
    batch_size = int(batch_size or cfg.TRAIN.BATCH_SIZE)
    workers = int(cfg.WORKERS if workers is None else workers)

    train_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.CenterCrop(imsize),
    ])
    train_full = TextDataset(cfg.DATA_DIR, "train", base_size=cfg.TREE.BASE_SIZE, transform=train_transform)
    val_full = TextDataset(cfg.DATA_DIR, "train", base_size=cfg.TREE.BASE_SIZE, transform=val_transform)

    n = len(train_full)
    n_val = max(batch_size, int(round(n * float(val_fraction))))
    n_val = min(n_val, max(batch_size, n - batch_size))
    rng = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=rng).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_set = torch.utils.data.Subset(train_full, train_idx)
    val_set = torch.utils.data.Subset(val_full, val_idx)

    loader_kwargs = dict(
        batch_size=batch_size,
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
    )
    if workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)

    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=True, **loader_kwargs)
    return train_full, train_loader, val_loader
def endless(loader: Iterable[Any]) -> Iterator[Any]:
    while True:
        for item in loader:
            yield item
def build_text_encoder(n_words: int, device: torch.device) -> RNN_ENCODER:
    text_encoder = RNN_ENCODER(n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location="cpu", weights_only=False)
    text_encoder.load_state_dict(state_dict)
    text_encoder.to(device)
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    text_encoder.eval()
    return text_encoder
def build_image_encoder(device: torch.device) -> CNN_ENCODER:
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_IMGNAME, map_location="cpu", weights_only=False)
    image_encoder.load_state_dict(state_dict)
    image_encoder.to(device)
    for p in image_encoder.parameters():
        p.requires_grad_(False)
    image_encoder.eval()
    return image_encoder
def make_lstm(device: torch.device) -> MogLSTM:
    return MogLSTM(256, 256, 5).to(device)
def prepare_batch(data: Any, text_encoder: RNN_ENCODER, batch_size: int, device: torch.device) -> Dict[str, Any]:
    imgs_list, captions, cap_lens, class_ids, keys = prepare_data(data)
    imgs = imgs_list[-1].to(device, non_blocking=True)
    captions = captions.to(device, non_blocking=True)
    cap_lens = cap_lens.to(device, non_blocking=True)
    with torch.no_grad():
        hidden = text_encoder.init_hidden(batch_size)
        words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
        words_embs = words_embs.detach()
        sent_emb = sent_emb.detach()
    mask = captions == 0
    num_words = words_embs.size(2)
    if mask.size(1) > num_words:
        mask = mask[:, :num_words]
    return {
        "imgs": imgs,
        "captions": captions,
        "cap_lens": cap_lens,
        "class_ids": class_ids,
        "keys": keys,
        "words_embs": words_embs,
        "sent_emb": sent_emb,
        "mask": mask,
    }
def cond_logits(netD, features: torch.Tensor, sent_emb: torch.Tensor) -> torch.Tensor:
    return netD.COND_DNET(features, sent_emb).view(-1)
def discriminator_hinge_loss(netD, real_imgs: torch.Tensor,
                             fake_imgs: torch.Tensor, sent_emb: torch.Tensor,
                             mismatch_weight: float = 0.5) -> torch.Tensor:
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    out_real = cond_logits(netD, real_features, sent_emb.detach())
    out_fake = cond_logits(netD, fake_features, sent_emb.detach())
    loss = torch.relu(1.0 - out_real).mean() + 0.5 * torch.relu(1.0 + out_fake).mean()
    if real_imgs.size(0) > 1 and mismatch_weight > 0:
        out_mis = cond_logits(netD, real_features[:-1], sent_emb.detach()[1:])
        loss = loss + float(mismatch_weight) * torch.relu(1.0 + out_mis).mean()
    return loss
def original_ma_gp_penalty(netD, real_imgs: torch.Tensor, sent_emb: torch.Tensor) -> torch.Tensor:
    real_imgs = real_imgs.detach().float().requires_grad_(True)
    sent_emb = sent_emb.detach().float().requires_grad_(True)
    real_features = netD(real_imgs)
    out_real = cond_logits(netD, real_features, sent_emb)
    grad_img, grad_sent = torch.autograd.grad(
        out_real,
        (real_imgs, sent_emb),
        grad_outputs=torch.ones_like(out_real),
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
    return 2.0 * torch.mean(grad_l2norm.pow(6))
def generator_proxy_loss(netD, image_encoder: CNN_ENCODER | None,
                         fake_imgs: torch.Tensor, sent_emb: torch.Tensor,
                         words_embs: torch.Tensor, cap_lens: torch.Tensor,
                         class_ids, batch_size: int, match_labels: torch.Tensor,
                         damsm_weight: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    features_fake = netD(fake_imgs)
    out_fake = cond_logits(netD, features_fake, sent_emb)
    adv = -out_fake.mean()
    loss = adv
    logs: Dict[str, float] = {"adv": float(adv.detach())}
    if image_encoder is not None and float(damsm_weight) > 0:
        region_fake, cnn_fake = image_encoder(fake_imgs)
        w0, w1, _ = words_loss(region_fake, words_embs, match_labels, cap_lens, class_ids, batch_size)
        s0, s1 = sent_loss(cnn_fake, sent_emb, match_labels, class_ids, batch_size)
        w_loss = (w0 + w1) * cfg.TRAIN.SMOOTH.LAMBDA
        s_loss = (s0 + s1) * cfg.TRAIN.SMOOTH.LAMBDA
        damsm = w_loss + s_loss
        loss = loss + float(damsm_weight) * damsm
        logs.update({
            "damsm": float(damsm.detach()),
            "word": float(w_loss.detach()),
            "sent": float(s_loss.detach()),
        })
    else:
        logs.update({"damsm": 0.0, "word": 0.0, "sent": 0.0})
    logs["total"] = float(loss.detach())
    return loss, logs
def set_requires_grad(module: torch.nn.Module | None, flag: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(flag)
def set_params_requires_grad(params: Iterable[torch.nn.Parameter], flag: bool) -> None:
    for p in params:
        p.requires_grad_(flag)
def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
