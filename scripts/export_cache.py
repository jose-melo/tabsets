#!/usr/bin/env python3
"""Pack the probability cells into parquet shards for distribution.

    python3 scripts/export_cache.py --source DIR --out DIR [--target-mb 400]
    python3 scripts/export_cache.py --source DIR --out DIR --verify
    python3 scripts/export_cache.py --source DIR --out DIR --dry-run

One file per cell is the right shape for a machine writing them one at a time and the
wrong shape for handing 45,000 of them to someone else. Shards are packed dataset by
dataset, so a reader who wants one dataset reads one shard, and an index says which.

Nothing is rounded. The probabilities stay float64 because a prediction set is a
comparison against a threshold, and a point sitting on that threshold would move.
``--verify`` reads the shards back and compares every array against its source file.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tabsets import cache  # noqa: E402


def find_sources(source: str) -> dict:
    """basename -> path, for every cell under ``source``."""
    out = {}
    for path in cache.find_npz(source):
        out[os.path.basename(path)[:-4]] = path
    return out


def plan(manifest: pd.DataFrame, sources: dict, target_bytes: int) -> tuple:
    """Group datasets into shards of roughly ``target_bytes``, keeping a dataset whole."""
    wanted = [n for n in (p[:-4] if p.endswith(".npz") else p for p in manifest.path)]
    missing = [n for n in wanted if n not in sources]
    by_dataset = {}
    for name in wanted:
        if name in sources:
            by_dataset.setdefault(name.split("__", 1)[0], []).append(name)

    sized = sorted(((sum(os.path.getsize(sources[n]) for n in names), ds, names)
                    for ds, names in by_dataset.items()), reverse=True)
    shards, sizes = [], []
    for size, _ds, names in sized:
        i = next((j for j, s in enumerate(sizes) if s + size <= target_bytes), None)
        if i is None:
            shards.append(list(names))
            sizes.append(size)
        else:
            shards[i].extend(names)
            sizes[i] += size
    return shards, missing


def export(manifest_path: str, source: str, out: str, target_mb: int, dry_run: bool = False) -> int:
    man = pd.read_parquet(manifest_path) if manifest_path.endswith(".parquet") \
        else pd.read_csv(manifest_path, low_memory=False)
    sources = find_sources(source)
    shards, missing = plan(man, sources, target_mb * 1_000_000)
    if missing:
        print(f"WARNING: {len(missing):,} cells in the manifest are absent from {source}",
              file=sys.stderr)
        for n in missing[:5]:
            print(f"    {n}", file=sys.stderr)

    if dry_run:
        sizes = [sum(os.path.getsize(sources[n]) for n in names) for names in shards]
        print(f"{sum(len(x) for x in shards):,} cells -> {len(shards)} shards, "
              f"{sum(sizes) / 1e9:.2f} GB, largest {max(sizes) / 1e6:.0f} MB")
        return 1 if missing else 0

    os.makedirs(out, exist_ok=True)
    index, total = [], 0
    for i, names in enumerate(shards):
        shard = f"cells-{i:03d}.parquet"
        cells = [cache.load_npz(sources[n]) for n in names]
        nbytes = cache.write_shard(cells, os.path.join(out, shard))
        total += nbytes
        index.extend(dict(name=n, shard=shard, dataset=c.dataset, model=c.model, seed=c.seed)
                     for n, c in zip(names, cells))
        print(f"  {shard}  {len(names):>5} cells  {nbytes / 1e6:7.1f} MB")
    pd.DataFrame(index).to_parquet(os.path.join(out, "index.parquet"), index=False)
    print(f"{len(index):,} cells in {len(shards)} shards, {total / 1e9:.2f} GB -> {out}")
    return 0


def verify(source: str, out: str, sample: int, seed: int) -> int:
    """Re-read the shards and compare every array against the file it came from."""
    index = pd.read_parquet(os.path.join(out, "index.parquet"))
    sources = find_sources(source)
    rng = np.random.default_rng(seed)
    pick = index.iloc[rng.choice(len(index), size=min(sample, len(index)), replace=False)]
    bad = 0
    for shard, rows in pick.groupby("shard"):
        cells = {c.name: c for c in cache.read_shard(os.path.join(out, shard), names=rows.name)}
        for name in rows.name:
            got, want = cells[name], cache.load_npz(sources[name])
            for field in cache.ROW_ARRAYS:
                a, b = getattr(got, field), getattr(want, field)
                if not (a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)):
                    print(f"MISMATCH {name}.{field}", file=sys.stderr)
                    bad += 1
            if got.meta != want.meta or got.x_cal_sha1 != want.x_cal_sha1:
                print(f"MISMATCH {name}.meta", file=sys.stderr)
                bad += 1
    print(f"verified {len(pick):,} cells, {bad} mismatches")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--source", required=True, help="directory holding the .npz cells")
    ap.add_argument("--out", required=True, help="where the shards go")
    ap.add_argument("--manifest", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reproduce", "data", "manifest.parquet"))
    ap.add_argument("--target-mb", type=int, default=400)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the plan and any missing cells; write nothing")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.verify:
        return verify(a.source, a.out, a.sample, a.seed)
    return export(a.manifest, a.source, a.out, a.target_mb, dry_run=a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
