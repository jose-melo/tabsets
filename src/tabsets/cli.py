"""``python -m tabsets <command>``

    python -m tabsets manifest  [--cache DIR] [--out manifest.parquet]
    python -m tabsets summarize [--block F1] [--alpha 0.10] [--scores lac,aps]
                                [--workers 7] [--out runs.parquet]

``manifest`` indexes a directory of cells. ``summarize`` writes one row per
(cell, score, level) with the metrics of the article, which is the table every figure
and table is computed from.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

SCORES = ("lac", "aps", "raps", "top_k")


def _write(df: pd.DataFrame, out: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    if out.endswith(".parquet"):
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"{len(df):,} rows -> {out}")


def cmd_manifest(args) -> int:
    from . import cache
    root = args.cache or __import__("tabsets.blocks", fromlist=["x"]).cache_root()
    paths = cache.find_npz(root)
    if not paths:
        print(f"no cells under {root}", file=sys.stderr)
        return 1
    _write(cache.manifest(cache.load_dir(root)), args.out)
    return 0


def _row(path):
    """One cell, every score and level. Top level so the worker processes can import it."""
    from . import blocks, metrics, sets
    scores = os.environ.get("TABSETS_SCORES", "lac").split(",")
    alphas = [float(a) for a in os.environ.get("TABSETS_ALPHAS", "0.10").split(",")]
    cell = blocks.load(path)
    out = []
    for score in scores:
        for alpha in alphas:
            try:
                r = sets.sets_for_cell(cell, score, alpha=alpha)
            except (ValueError, NotImplementedError):
                continue          # binary targets refuse the adaptive scores
            d = metrics.decomposition(cell.y_test, r.sets)
            d.update(path=path, conformity_score=score, alpha=alpha, qhat=r.qhat,
                     implementation=r.implementation, guarantee=r.guarantee,
                     coverage=metrics.coverage(cell.y_test, r.sets),
                     width=metrics.mean_width(r.sets),
                     singleton_rate=metrics.singleton_rate(r.sets),
                     accuracy=metrics.accuracy(cell.y_test, cell.p_test))
            out.append(d)
    return out


def cmd_summarize(args) -> int:
    from . import blocks
    os.environ["TABSETS_SCORES"] = args.scores
    os.environ["TABSETS_ALPHAS"] = args.alpha
    blk = blocks.block(args.block)
    print(f"block {args.block}: {len(blk):,} cells, "
          f"{blk.dataset.nunique()} datasets, {blk.model.nunique()} models")
    _write(blocks.map_cells(_row, blk, workers=args.workers), args.out)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tabsets", description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="index a directory of cells")
    m.add_argument("--cache", default=None)
    m.add_argument("--out", default="manifest.parquet")
    m.set_defaults(fn=cmd_manifest)

    s = sub.add_parser("summarize", help="one row per cell, score and level")
    s.add_argument("--block", default="F1")
    s.add_argument("--alpha", default="0.10", help="comma-separated levels")
    s.add_argument("--scores", default=",".join(SCORES), help="comma-separated scores")
    s.add_argument("--workers", type=int, default=7)
    s.add_argument("--out", default="runs.parquet")
    s.set_defaults(fn=cmd_summarize)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
