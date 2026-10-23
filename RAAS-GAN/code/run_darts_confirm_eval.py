from __future__ import annotations
import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

def parse_args():
    p = argparse.ArgumentParser("RAAS candidate confirmation (metric implementations excluded)")
    p.add_argument('--cfg', required=True)
    p.add_argument('--topk_dir', required=True)
    p.add_argument('--out_root', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=223)
    p.add_argument('--train_bs', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--val_fraction', type=float, default=0.1)
    p.add_argument('--split_seed', type=int, default=2026)
    p.add_argument('--scores_dir', default='',
                   help='Selection pass only: external top1.json ... top5.json, each containing {"fid": value}.')
    return p.parse_args()

def resolve(path: str | Path, base: Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (base / p).resolve()


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def run_logged(cmd: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[RUN] " + " ".join(cmd), flush=True)
    print(f"[LOG] {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8", errors="ignore") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with code {rc}; see {log_path}")


def find_archs(topk_dir: Path, topk: int) -> List[Path]:
    archs = sorted(
        topk_dir.glob("arch_top*.yaml"),
        key=lambda p: int(re.search(r"arch_top(\d+)", p.stem).group(1)) if re.search(r"arch_top(\d+)", p.stem) else 999,
    )
    return archs[:topk]


def rank_from_arch(path: Path) -> int:
    m = re.search(r"arch_top(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def main():
    args = parse_args()
    code_dir = Path(__file__).resolve().parent
    topk_dir = resolve(args.topk_dir, code_dir)
    out_root = resolve(args.out_root, code_dir)
    archs = find_archs(topk_dir, 5)
    if len(archs) != 5 or {rank_from_arch(p) for p in archs} != {1, 2, 3, 4, 5}:
        raise ValueError('Confirmation requires arch_top1.yaml through arch_top5.yaml')
    from darts.search_units import arch_key, normalize_arch
    keys = [arch_key(normalize_arch(read_json(p).get('arch', read_json(p)))) for p in archs]
    if len(set(keys)) != 5:
        raise ValueError('The five candidate architectures must be distinct')
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for arch_path in archs:
        rank = rank_from_arch(arch_path)
        tag = f'top{rank}'
        calib_dir = out_root / 'calibration' / tag
        netgema = calib_dir / 'netGEMA_final.pth'
        if not args.scores_dir:
            cmd = [sys.executable, 'train_fixed_arch.py', '--cfg', args.cfg,
                   '--arch', str(arch_path), '--gpu', str(args.gpu),
                   '--batch_size', str(args.train_bs), '--workers', str(args.workers),
                   '--seed', str(args.seed), '--val_fraction', str(args.val_fraction),
                   '--split_seed', str(args.split_seed), '--out_dir', str(calib_dir)]
            run_logged(cmd, code_dir, out_root / 'logs' / f'{tag}_train.log')
        if not netgema.is_file():
            raise FileNotFoundError(f'Missing confirmed candidate EMA: {netgema}')
        payload = read_json(arch_path)
        row = dict(rank=rank, arch=payload.get('arch', payload), arch_path=str(arch_path),
                   checkpoint=str(netgema), fid=None)
        if args.scores_dir:
            score_file = resolve(args.scores_dir, code_dir) / f'{tag}.json'
            if not score_file.is_file():
                raise FileNotFoundError(f'All five external proxy-FID results are required: {score_file}')
            score = float(read_json(score_file)['fid'])
            if not math.isfinite(score) or score < 0:
                raise ValueError(f'Invalid external proxy FID in {score_file}')
            row['fid'] = score
        rows.append(row)
    # Do not select an architecture before ALL five confirmation scores exist.
    best = min(rows, key=lambda r: r['fid']) if args.scores_dir else None
    selected = out_root / 'selected_architecture.yaml'
    if best is not None and selected.exists():
        raise FileExistsError(f'Refusing to overwrite selected architecture: {selected}')
    write_json(out_root / 'confirm_eval_summary.json',
               dict(rows=rows, best_by_fid=best, scores_complete=best is not None))
    if best is not None:
        write_json(selected, {'arch': best['arch']})
        print(f"[SELECTED] top{best['rank']} proxy_FID={best['fid']} arch={selected}", flush=True)
    else:
        print('[CONFIRMATION_TRAINING_DONE] Evaluate the five EMA models externally, then rerun with --scores_dir.', flush=True)
        print('[NOT_SELECTED] No metric implementation or fabricated proxy score is included.', flush=True)


if __name__ == '__main__':
    main()
