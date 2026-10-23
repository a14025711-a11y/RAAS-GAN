from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping
import torch
import torch.nn.functional as F
from darts.search_ops import TEXT_DISPLAY, TEXT_OPS, VISUAL_DISPLAY, VISUAL_OPS
from darts.search_units import SEARCH_STAGES, arch_display
from darts.train_utils import save_json
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Export Top-k fixed architectures from DARTS alpha")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", default="")

    p.add_argument("--history", default="", help="alpha_history.jsonl; defaults to ckpt_dir/alpha_history.jsonl")
    p.add_argument("--ngf", type=int, default=64, help="Generator channel base used for parameter estimation.")
    return p.parse_args()
def linear(in_f: int, out_f: int, bias: bool = True) -> int:
    return int(in_f) * int(out_f) + (int(out_f) if bias else 0)
def conv2d(in_c: int, out_c: int, k: int, bias: bool = True, groups: int = 1) -> int:
    return int(out_c) * (int(in_c) // int(groups)) * int(k) * int(k) + (int(out_c) if bias else 0)
def conv1d(in_c: int, out_c: int, k: int, bias: bool = True) -> int:
    return int(out_c) * int(in_c) * int(k) + (int(out_c) if bias else 0)
def affine_params(num_features: int) -> int:
    # fc_gamma and fc_beta, each Linear(256,256)+Linear(256,num_features).
    return 2 * (linear(256, 256) + linear(256, num_features))
def g_block_params(in_c: int, out_c: int) -> int:
    total = 0
    total += conv2d(in_c, out_c, 3)
    total += conv2d(out_c, out_c, 3)
    total += 3 * affine_params(in_c) + 3 * affine_params(out_c)
    total += linear(in_c, 256) + linear(out_c, 256)
    total += 1
    if in_c != out_c:
        total += conv2d(in_c, out_c, 1)
    return total
def mog_lstm_params() -> int:
    hidden = 256
    inp = 256
    total = 0
    total += inp * (hidden * 4)
    total += hidden * (hidden * 4)
    total += hidden * 4
    total += hidden * inp
    total += inp * hidden
    total += linear(100, hidden)
    total += linear(100, hidden)
    return total
def baseline_g_params(ngf: int) -> int:
    total = 0
    total += linear(100, ngf * 8 * 4 * 4)
    for in_c, out_c in (
        (ngf * 8, ngf * 8),
        (ngf * 8, ngf * 8),
        (ngf * 8, ngf * 8),
        (ngf * 8, ngf * 8),
        (ngf * 8, ngf * 4),
        (ngf * 4, ngf * 2),
        (ngf * 2 * 3, ngf),
        (ngf, ngf),
    ):
        total += g_block_params(in_c, out_c)
    total += conv2d(256, ngf * 2, 1, bias=False)
    total += conv2d(256, 128 * 128, 1, bias=False)
    total += conv2d(ngf, 3, 3)
    total += mog_lstm_params()
    return total
def visual_params(name: str, channels: int) -> int:
    c = int(channels)
    if name == "identity":
        return 0
    if name in {"plain_conv", "resblock"}:
        return 1 + conv2d(c, c, 3) + conv2d(c, c, 3)
    if name == "glu_block":
        return 1 + conv2d(c, 2 * c, 3) + conv2d(c, c, 1)
    if name == "light_resblock":
        return 1 + conv2d(c, c, 3, groups=c) + conv2d(c, c, 1)
    raise ValueError(f"Unsupported visual op: {name}")
def text_params(name: str, channels: int) -> int:
    c = int(channels)
    if name == "no_fusion":
        return 0
    if name == "additive_fusion":
        return 1 + linear(256, c)
    if name == "film_fusion":
        return 1 + linear(256, 256) + linear(256, 2 * c) + conv2d(c, c, 3) + conv2d(c, c, 1)
    if name == "gated_film":
        return (
            1
            + linear(256, 256)
            + linear(256, 2 * c)
            + linear(256, 256)
            + linear(256, c)
            + conv2d(c, c, 1)
            + conv2d(c, c, 3)
        )
    if name == "light_attention_fusion":
        att_dim = 96
        return (
            1
            + conv2d(c, att_dim, 1)
            + conv1d(256, att_dim, 1)
            + conv1d(256, att_dim, 1)
            + conv2d(c, c, 3, groups=c)
            + conv2d(c, c, 1)
            + conv2d(c + att_dim, c, 1)
            + conv2d(c, c, 3)
            + linear(256, c)
        )
    raise ValueError(f"Unsupported text op: {name}")
def arch_extra_params(arch: Mapping[str, Mapping[str, str]], ngf: int) -> int:
    channels = {"s1": ngf * 2, "s2": ngf * 2 * 3, "s3": ngf}
    total = 0
    for stage in SEARCH_STAGES:
        row = arch[stage]
        total += visual_params(str(row["visual"]), channels[stage])
        total += text_params(str(row["text"]), channels[stage])
    return total
def load_log_probs(ckpt_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["netG"] if isinstance(ckpt, dict) and "netG" in ckpt else ckpt
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for stage in SEARCH_STAGES:
        prefix = f"search_{stage}."
        av = state[prefix + "alpha_visual"].float()
        at = state[prefix + "alpha_text"].float()
        out[stage] = {
            "visual": F.log_softmax(av, dim=0),
            "text": F.log_softmax(at, dim=0),
        }
    return out
def load_history_log_probs(path: Path, window: int) -> List[Dict[str, Dict[str, Dict[str, float]]]]:
    if not path.is_file() or window == 0:
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("phase") == "search":
                rows.append(row)
    if window > 0:
        rows = rows[-window:]
    out: List[Dict[str, Dict[str, Dict[str, float]]]] = []
    for row in rows:
        alpha = row.get("alpha", {})
        cur: Dict[str, Dict[str, Dict[str, float]]] = {}
        for stage in SEARCH_STAGES:
            cur[stage] = {"visual": {}, "text": {}}
            for kind, names in (("visual", VISUAL_OPS), ("text", TEXT_OPS)):
                probs = alpha.get(stage, {}).get(kind, {})
                for name in names:
                    p = float(probs.get(name, 0.0) or 0.0)
                    cur[stage][kind][name] = math.log(max(p, 1.0e-12))
        out.append(cur)
    return out
def history_genotypes(path: Path, window: int) -> List[Dict[str, Dict[str, str]]]:
    if not path.is_file() or window == 0:
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("phase") == "search":
                rows.append(row)
    if window > 0:
        rows = rows[-window:]
    out: List[Dict[str, Dict[str, str]]] = []
    for row in rows:
        genotype = row.get("alpha", {}).get("genotype")
        if isinstance(genotype, dict):
            out.append(genotype)
    return out
def arch_key(arch: Mapping[str, Mapping[str, str]]) -> str:
    parts: List[str] = []
    for stage in SEARCH_STAGES:
        parts.append(str(arch[stage]["visual"]))
        parts.append(str(arch[stage]["text"]))
    return "|".join(parts)
def stage_diff(a: Mapping[str, Mapping[str, str]], b: Mapping[str, Mapping[str, str]]) -> int:
    diff = 0
    for stage in SEARCH_STAGES:
        if a[stage] != b[stage]:
            diff += 1
    return diff
def choice_margin(log_probs: torch.Tensor, idx: int) -> float:
    if log_probs.numel() <= 1:
        return 0.0
    mask = torch.ones(log_probs.numel(), dtype=torch.bool)
    mask[idx] = False
    return float(log_probs[idx] - torch.max(log_probs[mask]))
def arch_score_from_log_probs(arch: Mapping[str, Mapping[str, str]], log_probs: Mapping[str, Mapping[str, Any]]) -> float:
    score = 0.0
    for stage in SEARCH_STAGES:
        vi = VISUAL_OPS.index(str(arch[stage]["visual"]))
        ti = TEXT_OPS.index(str(arch[stage]["text"]))
        v = log_probs[stage]["visual"]
        t = log_probs[stage]["text"]
        score += float(v[vi] if isinstance(v, torch.Tensor) else v[VISUAL_OPS[vi]])
        score += float(t[ti] if isinstance(t, torch.Tensor) else t[TEXT_OPS[ti]])
    return score
def arch_margin(arch: Mapping[str, Mapping[str, str]], log_probs: Mapping[str, Mapping[str, torch.Tensor]]) -> float:
    margin = 0.0
    for stage in SEARCH_STAGES:
        vi = VISUAL_OPS.index(str(arch[stage]["visual"]))
        ti = TEXT_OPS.index(str(arch[stage]["text"]))
        margin += choice_margin(log_probs[stage]["visual"], vi)
        margin += choice_margin(log_probs[stage]["text"], ti)
    return margin
def build_rows(
    log_probs: Dict[str, Dict[str, torch.Tensor]],
    history: List[Dict[str, Dict[str, Dict[str, float]]]],
    genotype_votes: List[Dict[str, Dict[str, str]]],
    ngf: int,
) -> List[Dict[str, Any]]:
    base_params = baseline_g_params(ngf)
    rows: List[Dict[str, Any]] = []
    per_stage_choices = list(itertools.product(range(len(VISUAL_OPS)), range(len(TEXT_OPS))))
    for combo in itertools.product(per_stage_choices, repeat=len(SEARCH_STAGES)):
        arch: Dict[str, Dict[str, str]] = {}
        for stage, (vi, ti) in zip(SEARCH_STAGES, combo):
            arch[stage] = {"visual": VISUAL_OPS[vi], "text": TEXT_OPS[ti]}

        latest_score = arch_score_from_log_probs(arch, log_probs)
        margin = arch_margin(arch, log_probs)
        if history:
            hist_scores = [arch_score_from_log_probs(arch, h) for h in history]
            stability_score = sum(hist_scores) / len(hist_scores)
        else:
            stability_score = latest_score
        if genotype_votes:
            key = arch_key(arch)
            exact_vote = sum(1 for g in genotype_votes if arch_key(g) == key) / float(len(genotype_votes))
        else:
            exact_vote = 0.0

        extra = arch_extra_params(arch, ngf)
        total = base_params + extra
        rows.append(
            {
                "score": latest_score,
                "arch": arch,
                "objectives": {
                    "alpha_score": latest_score,
                    "alpha_margin": margin,
                    "stability_score": stability_score,
                    "exact_vote": exact_vote,
                    "params_total": total,
                    "params_extra": extra,
                    "param_ratio": total / float(base_params),
                },
            }
        )
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["alpha_rank"] = i
    return rows
def assign_ranks(rows: List[Dict[str, Any]]) -> None:
    def rank_by(key: str, reverse: bool) -> Dict[str, int]:
        ordered = sorted(rows, key=lambda r: float(r["objectives"][key]), reverse=reverse)
        return {arch_key(r["arch"]): i for i, r in enumerate(ordered, start=1)}

    alpha_rank = rank_by("alpha_score", True)
    margin_rank = rank_by("alpha_margin", True)
    stability_rank = rank_by("stability_score", True)
    params_rank = rank_by("params_total", False)
    for row in rows:
        key = arch_key(row["arch"])
        row["ranks"] = {
            "alpha": alpha_rank[key],
            "margin": margin_rank[key],
            "stability": stability_rank[key],
            "params": params_rank[key],
        }
def apply_raas_score(rows: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    assign_ranks(rows)
    for row in rows:
        ranks = row["ranks"]
        exact_vote = float(row["objectives"].get("exact_vote", 0.0))
        vote_bonus = exact_vote * len(rows)
        raas = (
            float(1.0) * ranks["alpha"]
            + float(0.5) * ranks["margin"]
            + float(1.0) * ranks["stability"]
            + float(0.25) * ranks["params"]
            - float(1.0) * vote_bonus
        )
        row["raas_score"] = float(raas)
def select_rows(rows: List[Dict[str, Any]], topk: int, min_stage_diff: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in rows:
        arch = row["arch"]
        if all(stage_diff(arch, prev["arch"]) >= int(min_stage_diff) for prev in selected):
            selected.append(row)
        if len(selected) >= int(topk):
            break
    return selected
def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        suffix = "topk_archs_raas" if 'raas' == "raas" else "topk_archs"
        out_dir = ckpt_path.parent / suffix
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Use an empty candidate output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    history_path = Path(args.history) if args.history else ckpt_path.parent / "alpha_history.jsonl"
    log_probs = load_log_probs(str(ckpt_path))
    history = load_history_log_probs(history_path, int(5))
    votes = history_genotypes(history_path, int(5))
    if len(history) < 5 or len(votes) < 5:
        raise ValueError("RAAS requires alpha history from the last five search epochs")
    rows = build_rows(log_probs, history, votes, int(args.ngf))

    selection_rows = rows
    selection_rows = rows[: int(50)]

    apply_raas_score(selection_rows, args)
    selection_rows = sorted(selection_rows, key=lambda r: float(r["raas_score"]))
    selected = select_rows(selection_rows, int(5), int(1))

    summary = {
        "source_ckpt": str(ckpt_path),
        "history": str(history_path) if history_path.is_file() else "",
        "selection": 'raas',
        "topk": len(selected),
        "candidate_pool": len(selection_rows),
        "stability_window": int(5),
        "weights": {
            "alpha": float(1.0),
            "margin": float(0.5),
            "stability": float(1.0),
            "params": float(0.25),
        },
        "ngf": int(args.ngf),
        "baseline_params": baseline_g_params(int(args.ngf)),
        "visual_ops": VISUAL_OPS,
        "text_ops": TEXT_OPS,
        "display_names": {"visual": VISUAL_DISPLAY, "text": TEXT_DISPLAY},
        "selected": selected,
    }
    save_json(out_dir / "topk_summary.json", summary)
    save_json(out_dir / "raas_candidates.json", {"rows": selection_rows})

    for i, row in enumerate(selected, start=1):
        arch = row["arch"]
        payload = {
            "rank": i,
            "selection": 'raas',
            "score": float(row["score"]),
            "raas_score": row.get("raas_score", None),
            "objectives": row.get("objectives", {}),
            "ranks": row.get("ranks", {}),
            "arch": arch,
            "display": arch_display(arch),
        }
        # JSON is valid YAML 1.2, so these files stay dependency-free.
        save_json(out_dir / f"arch_top{i}.yaml", payload)
        obj = row["objectives"]
        print(
            f"[TOP{i}][RAAS] raas={float(row['raas_score']):.3f} "
            f"alpha={float(row['score']):.6f} margin={obj['alpha_margin']:.3f} "
            f"stable={obj['stability_score']:.6f} vote={obj['exact_vote']:.2f} "
            f"params={obj['params_total'] / 1e6:.2f}M arch={arch}",
            flush=True,
        )
    print(f"[DONE] out_dir={out_dir}", flush=True)
if __name__ == "__main__":
    main()
