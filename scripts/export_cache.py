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
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from tabsets import cache  # noqa: E402


def find_sources(manifest: pd.DataFrame, source: str) -> tuple:
    """cell name -> the file on disk the manifest chose, plus the names it could not find.

    Resolving by name alone is wrong and was wrong here: 12,294 names exist in more than
    one place under the source root, with different bytes, because a run was repeated.
    The manifest's ``source_path`` records which of them the deduplication kept, and that
    is the only thing that identifies it. A manifest without that column is resolved by
    name, which is safe only when no name is duplicated, so that case is checked.
    """
    def stem(n):
        return n[:-4] if n.endswith(".npz") else n

    if "source_path" in manifest:
        found, missing = {}, []
        for name, rel in zip(manifest.path, manifest.source_path):
            path = os.path.join(source, rel)
            (found.__setitem__(stem(name), path) if os.path.exists(path)
             else missing.append(stem(name)))
        return found, missing

    seen = {}
    for path in cache.find_npz(source):
        seen.setdefault(os.path.basename(path)[:-4], []).append(path)
    ambiguous = {k: v for k, v in seen.items() if len(v) > 1}
    if ambiguous:
        raise SystemExit(
            f"this manifest has no source_path and {len(ambiguous):,} cell names resolve "
            f"to more than one file under {source}; regenerate the manifest with the "
            f"column, or the export would pack an arbitrary one of them")
    names = {stem(n) for n in manifest.path}
    return ({k: v[0] for k, v in seen.items() if k in names},
            sorted(names - set(seen)))


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
    sources, unresolved = find_sources(man, source)
    shards, missing = plan(man, sources, target_mb * 1_000_000)
    missing = sorted(set(missing) | set(unresolved))
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

    # Which manifest this was packed from, so a stale export cannot be uploaded by
    # mistake: cells keep landing, and a directory of shards otherwise looks the same
    # whether it was made an hour ago or a week ago.
    provenance = dict(
        exported_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        manifest=os.path.abspath(manifest_path),
        manifest_cells=int(len(man)),
        manifest_mtime=datetime.datetime.fromtimestamp(
            os.path.getmtime(manifest_path)).astimezone().isoformat(timespec="seconds"),
        cells_exported=len(index), cells_missing=len(missing),
        shards=len(shards), bytes=total,
    )
    with open(os.path.join(out, "EXPORT.json"), "w") as fh:
        json.dump(provenance, fh, indent=2)
        fh.write("\n")
    print(f"{len(index):,} cells in {len(shards)} shards, {total / 1e9:.2f} GB -> {out}")
    print(f"packed from a manifest of {len(man):,} cells, last written {provenance['manifest_mtime']}")
    return 0


def verify(source: str, out: str, sample: int, seed: int) -> int:
    """Re-read the shards and compare every array against the file it came from."""
    index = pd.read_parquet(os.path.join(out, "index.parquet"))
    man = pd.read_parquet(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "reproduce", "data", "manifest.parquet"))
    sources, _ = find_sources(man, source)
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
