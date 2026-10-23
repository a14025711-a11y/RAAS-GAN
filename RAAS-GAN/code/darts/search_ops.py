from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


VISUAL_OPS: List[str] = [
    "identity",
    "plain_conv",
    "resblock",
    "glu_block",
    "light_resblock",
]

TEXT_OPS: List[str] = [
    "no_fusion",
    "additive_fusion",
    "film_fusion",
    "gated_film",
    "light_attention_fusion",
]


VISUAL_DISPLAY: Dict[str, str] = {
    "identity": "Identity",
    "plain_conv": "Plain-Conv Block",
    "resblock": "ResBlock",
    "glu_block": "GLU-Block",
    "light_resblock": "Light-ResBlock",
}

TEXT_DISPLAY: Dict[str, str] = {
    "no_fusion": "No-Fusion",
    "additive_fusion": "Additive-Fusion",
    "film_fusion": "FiLM-Fusion",
    "gated_film": "Gated-FiLM",
    "light_attention_fusion": "Light-Attention-Fusion",
}


def canonical_visual_name(name: str) -> str:
    return _canonical_name(name, VISUAL_OPS, VISUAL_DISPLAY)


def canonical_text_name(name: str) -> str:
    return _canonical_name(name, TEXT_OPS, TEXT_DISPLAY)


def _canonical_name(name: str, keys: Iterable[str], display: Dict[str, str]) -> str:
    raw = str(name).strip()
    norm = raw.lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    alias = {
        "baseline_preserve": "identity",
        "identity_baseline_preserve": "identity",
        "plain_conv_block": "plain_conv",
        "glu": "glu_block",
        "light_res_block": "light_resblock",
        "none": "no_fusion",
        "baseline": "no_fusion",
        "no_fusion_baseline_preserve": "no_fusion",
        "film": "film_fusion",
        "film_block": "film_fusion",
        "light_attention": "light_attention_fusion",
    }
    norm = alias.get(norm, norm)
    if norm in keys:
        return norm
    for key, label in display.items():
        if raw == label or norm == label.lower().replace("-", "_").replace(" ", "_"):
            return key
    raise ValueError(f"Unsupported op name: {name}")


def zero_module(module: nn.Module) -> nn.Module:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class IdentityVisual(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PlainConvBlock(nn.Module):
    def __init__(self, channels: int, alpha_init: float = 1.0e-3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            zero_module(nn.Conv2d(channels, channels, 3, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.net(x)


class ResBlockVisual(nn.Module):
    def __init__(self, channels: int, alpha_init: float = 1.0e-3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        hidden = max(16, channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            zero_module(nn.Conv2d(hidden, channels, 3, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.net(x)


class GLUBlockVisual(nn.Module):
    def __init__(self, channels: int, alpha_init: float = 1.0e-3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.in_proj = nn.Conv2d(channels, channels * 2, 3, 1, 1)
        self.out_proj = zero_module(nn.Conv2d(channels, channels, 1, 1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=1)
        h = value * torch.sigmoid(gate)
        return x + self.alpha * self.out_proj(h)


class LightResBlockVisual(nn.Module):
    def __init__(self, channels: int, alpha_init: float = 1.0e-3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.LeakyReLU(0.2, inplace=True),
            zero_module(nn.Conv2d(channels, channels, 1, 1, 0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.net(x)


class NoFusion(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        sent_emb: torch.Tensor | None = None,
        words_embs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return x


class AdditiveFusion(nn.Module):
    def __init__(self, channels: int, cond_dim: int = 256, alpha_init: float = 1.0e-3):
        super().__init__()
        self.channels = int(channels)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.to_bias = zero_module(nn.Linear(cond_dim, channels))

    def forward(
        self,
        x: torch.Tensor,
        sent_emb: torch.Tensor | None = None,
        words_embs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sent_emb is None:
            return x
        bias = torch.tanh(self.to_bias(sent_emb.float())).view(x.size(0), self.channels, 1, 1)
        return x + self.alpha * bias.to(dtype=x.dtype)


class FiLMFusion(nn.Module):
    def __init__(self, channels: int, cond_dim: int = 256, hidden: int = 256, alpha_init: float = 1.0e-3):
        super().__init__()
        self.channels = int(channels)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.to_film = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            zero_module(nn.Linear(hidden, channels * 2)),
        )
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            zero_module(nn.Conv2d(channels, channels, 1, 1, 0)),
        )

    def forward(
        self,
        x: torch.Tensor,
        sent_emb: torch.Tensor | None = None,
        words_embs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sent_emb is None:
            return x
        gamma, beta = self.to_film(sent_emb.float()).chunk(2, dim=1)
        gamma = torch.tanh(gamma).view(x.size(0), self.channels, 1, 1).to(dtype=x.dtype)
        beta = torch.tanh(beta).view(x.size(0), self.channels, 1, 1).to(dtype=x.dtype)
        h = x * (1.0 + gamma) + beta
        return x + self.alpha * self.adapter(h)


class GatedFiLMFusion(nn.Module):
    def __init__(self, channels: int, cond_dim: int = 256, hidden: int = 256, alpha_init: float = 1.0e-3):
        super().__init__()
        self.channels = int(channels)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.to_film = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            zero_module(nn.Linear(hidden, channels * 2)),
        )
        self.to_gate = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            zero_module(nn.Linear(hidden, channels)),
            nn.Sigmoid(),
        )
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            zero_module(nn.Conv2d(channels, channels, 3, 1, 1)),
        )

    def forward(
        self,
        x: torch.Tensor,
        sent_emb: torch.Tensor | None = None,
        words_embs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sent_emb is None:
            return x
        gamma, beta = self.to_film(sent_emb.float()).chunk(2, dim=1)
        gamma = torch.tanh(gamma).view(x.size(0), self.channels, 1, 1).to(dtype=x.dtype)
        beta = torch.tanh(beta).view(x.size(0), self.channels, 1, 1).to(dtype=x.dtype)
        gate = self.to_gate(sent_emb.float()).view(x.size(0), self.channels, 1, 1).to(dtype=x.dtype)
        h = x * (1.0 + gamma) + beta
        return x + self.alpha * gate * self.adapter(h)


class LightAttentionFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        word_dim: int = 256,
        att_dim: int = 96,
        max_side: int = 32,
        alpha_init: float = 1.0e-3,
    ):
        super().__init__()
        self.channels = int(channels)
        self.word_dim = int(word_dim)
        self.att_dim = int(att_dim)
        self.max_side = int(max_side)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.q = nn.Conv2d(channels, att_dim, 1, 1, 0)
        self.k = nn.Conv1d(word_dim, att_dim, 1, 1, 0)
        self.v = nn.Conv1d(word_dim, att_dim, 1, 1, 0)
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 1, 1, 0),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + att_dim, channels, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            zero_module(nn.Conv2d(channels, channels, 3, 1, 1)),
        )
        self.sent_gate = nn.Sequential(
            zero_module(nn.Linear(word_dim, channels)),
            nn.Sigmoid(),
        )
        self.scale = float(att_dim) ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        sent_emb: torch.Tensor | None = None,
        words_embs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if words_embs is None:
            return x
        b, c, h, w = x.size()
        rh, rw = min(h, self.max_side), min(w, self.max_side)
        pooled = x if (rh == h and rw == w) else F.adaptive_avg_pool2d(x, (rh, rw))
        q = self.q(pooled).flatten(2).transpose(1, 2)
        k = self.k(words_embs.float()).transpose(1, 2)
        v = self.v(words_embs.float()).transpose(1, 2)
        attn = torch.bmm(q.float(), k.transpose(1, 2)) * self.scale
        if mask is not None:
            cur_mask = mask
            if cur_mask.size(1) > attn.size(-1):
                cur_mask = cur_mask[:, :attn.size(-1)]
            elif cur_mask.size(1) < attn.size(-1):
                pad = torch.zeros(
                    cur_mask.size(0),
                    attn.size(-1) - cur_mask.size(1),
                    dtype=cur_mask.dtype,
                    device=cur_mask.device,
                )
                cur_mask = torch.cat([cur_mask, pad], dim=1)
            attn = attn.masked_fill(cur_mask.unsqueeze(1).bool(), -1.0e4)
        ctx = torch.bmm(torch.softmax(attn, dim=-1), v).transpose(1, 2)
        ctx = ctx.contiguous().view(b, self.att_dim, rh, rw).to(dtype=x.dtype)
        if rh != h or rw != w:
            ctx = F.interpolate(ctx, size=(h, w), mode="bilinear", align_corners=False)
        residual = self.fuse(torch.cat([self.local(x), ctx], dim=1))
        if sent_emb is not None:
            gate = self.sent_gate(sent_emb.float()).view(b, c, 1, 1).to(dtype=x.dtype)
            residual = residual * gate
        return x + self.alpha * residual


def make_visual_op(name: str, channels: int, alpha_init: float = 1.0e-3) -> nn.Module:
    name = canonical_visual_name(name)
    if name == "identity":
        return IdentityVisual()
    if name == "plain_conv":
        return PlainConvBlock(channels, alpha_init=alpha_init)
    if name == "resblock":
        return ResBlockVisual(channels, alpha_init=alpha_init)
    if name == "glu_block":
        return GLUBlockVisual(channels, alpha_init=alpha_init)
    if name == "light_resblock":
        return LightResBlockVisual(channels, alpha_init=alpha_init)
    raise ValueError(f"Unsupported visual op: {name}")


def make_text_op(name: str, channels: int, alpha_init: float = 1.0e-3) -> nn.Module:
    name = canonical_text_name(name)
    if name == "no_fusion":
        return NoFusion()
    if name == "additive_fusion":
        return AdditiveFusion(channels, alpha_init=alpha_init)
    if name == "film_fusion":
        return FiLMFusion(channels, alpha_init=alpha_init)
    if name == "gated_film":
        return GatedFiLMFusion(channels, alpha_init=alpha_init)
    if name == "light_attention_fusion":
        return LightAttentionFusion(channels, alpha_init=alpha_init)
    raise ValueError(f"Unsupported text op: {name}")

