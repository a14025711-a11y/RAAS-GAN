from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from darts.search_ops import (
    TEXT_DISPLAY,
    TEXT_OPS,
    VISUAL_DISPLAY,
    VISUAL_OPS,
    canonical_text_name,
    canonical_visual_name,
    make_text_op,
    make_visual_op,
)


SEARCH_STAGES = ["s1", "s2", "s3"]
STAGE_LABELS = {
    "s1": "block5_after_before_attention",
    "s2": "attention_after_before_block6",
    "s3": "block7_after_before_conv_img",
}


class MixedVisualBlock(nn.Module):
    def __init__(self, channels: int, alpha_init: float = 1.0e-3):
        super().__init__()
        self.ops = nn.ModuleList([make_visual_op(name, channels, alpha_init) for name in VISUAL_OPS])

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        out = None
        for w, op in zip(weights, self.ops):
            cur = op(x) * w
            out = cur if out is None else out + cur
        return out


class MixedTextFusionBlock(nn.Module):
    def __init__(self, channels: int, alpha_init: float = 1.0e-3):
        super().__init__()
        self.ops = nn.ModuleList([make_text_op(name, channels, alpha_init) for name in TEXT_OPS])

    def forward(
        self,
        x: torch.Tensor,
        sent_emb: torch.Tensor | None,
        words_embs: torch.Tensor | None,
        mask: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        out = None
        for w, op in zip(weights, self.ops):
            cur = op(x, sent_emb, words_embs, mask) * w
            out = cur if out is None else out + cur
        return out


class SearchableCondGenUnit(nn.Module):
    def __init__(self, channels: int, tau: float = 1.0, alpha_init: float = 1.0e-3):
        super().__init__()
        self.channels = int(channels)
        self.tau = float(tau)
        self.visual = MixedVisualBlock(channels, alpha_init=alpha_init)
        self.text = MixedTextFusionBlock(channels, alpha_init=alpha_init)
        self.alpha_visual = nn.Parameter(torch.zeros(len(VISUAL_OPS)))
        self.alpha_text = nn.Parameter(torch.zeros(len(TEXT_OPS)))

    def visual_weights(self) -> torch.Tensor:
        return F.softmax(self.alpha_visual / max(self.tau, 1.0e-6), dim=0)

    def text_weights(self) -> torch.Tensor:
        return F.softmax(self.alpha_text / max(self.tau, 1.0e-6), dim=0)

    def forward(
        self,
        x: torch.Tensor,
        sent_emb: torch.Tensor | None = None,
        words_embs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.visual(x, self.visual_weights())
        x = self.text(x, sent_emb, words_embs, mask, self.text_weights())
        return x

    def genotype(self) -> Dict[str, str]:
        vi = int(torch.argmax(self.alpha_visual.detach()).item())
        ti = int(torch.argmax(self.alpha_text.detach()).item())
        return {"visual": VISUAL_OPS[vi], "text": TEXT_OPS[ti]}

    def alpha_summary(self) -> Dict[str, object]:
        vw = self.visual_weights().detach().cpu()
        tw = self.text_weights().detach().cpu()
        return {
            "visual": {name: float(vw[i]) for i, name in enumerate(VISUAL_OPS)},
            "text": {name: float(tw[i]) for i, name in enumerate(TEXT_OPS)},
            "genotype": self.genotype(),
            "entropy": {
                "visual": float(-(vw * (vw + 1.0e-12).log()).sum()),
                "text": float(-(tw * (tw + 1.0e-12).log()).sum()),
            },
        }


class FixedCondGenUnit(nn.Module):
    def __init__(self, channels: int, visual: str, text: str, alpha_init: float = 1.0e-3):
        super().__init__()
        self.channels = int(channels)
        self.visual_name = canonical_visual_name(visual)
        self.text_name = canonical_text_name(text)
        self.visual = make_visual_op(self.visual_name, channels, alpha_init=alpha_init)
        self.text = make_text_op(self.text_name, channels, alpha_init=alpha_init)

    def forward(
        self,
        x: torch.Tensor,
        sent_emb: torch.Tensor | None = None,
        words_embs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.visual(x)
        return self.text(x, sent_emb, words_embs, mask)

    def genotype(self) -> Dict[str, str]:
        return {"visual": self.visual_name, "text": self.text_name}


def normalize_arch(arch: Mapping[str, Mapping[str, str]]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for stage in SEARCH_STAGES:
        row = arch.get(stage) or arch.get(stage.upper()) or arch.get(STAGE_LABELS[stage])
        if row is None:
            raise ValueError(f"Missing DARTS arch stage: {stage}")
        out[stage] = {
            "visual": canonical_visual_name(row["visual"]),
            "text": canonical_text_name(row["text"]),
        }
    return out


def arch_display(arch: Mapping[str, Mapping[str, str]]) -> Dict[str, Dict[str, str]]:
    norm = normalize_arch(arch)
    return {
        stage: {
            "visual": VISUAL_DISPLAY[norm[stage]["visual"]],
            "text": TEXT_DISPLAY[norm[stage]["text"]],
        }
        for stage in SEARCH_STAGES
    }


def arch_key(arch: Mapping[str, Mapping[str, str]]) -> str:
    norm = normalize_arch(arch)
    parts: List[str] = []
    for stage in SEARCH_STAGES:
        parts.append(norm[stage]["visual"])
        parts.append(norm[stage]["text"])
    return "|".join(parts)

