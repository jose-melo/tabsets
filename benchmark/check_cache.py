#!/usr/bin/env python
"""Rebuild every metric from the probability cache and diff it against the CSV.

    python check_cache.py <out-dir> [--tol 1e-6] [--data-path data] [--rebuild-manifest]

``<out-dir>`` is scanned recursively for ``cache/*.npz`` and for the per-job
``talent_benchmark__*.csv`` files. For every npz the four conformity scores
are recomputed through ``cache_io.conformal_from_cache`` (the same MAPIE path
the benchmark used) and every numeric column of the matching CSV rows -
keyed by ``(dataset, task_type, model, tag, seed, conformity_score)`` - is
compared (``tag`` is the ``--tag`` sweep suffix, empty for base cells). Exit code 1 on any |diff| > tol, on a cell without CSV rows, on a
CSV row without a cell, on duplicate keys, or (with ``--data-path``) on a
feature-hash mismatch against the arrays on disk.

Also prints the completeness matrix and flags: ``budget_bound``
(best_epoch >= max_epoch - 1), ``hpo_ran == False`` on a model whose HPO space
is non-empty, ``mock_run``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache_io import (  # noqa: E402
    MANIFEST_COLUMNS,
    append_manifest,
    conformal_from_cache,
    parse_cache_name,
    read_cache,
    sha1_array,
)

KEY = ["dataset", "task_type", "model", "tag", "seed", "conformity_score"]
MODELS_WITHOUT_HPO = {"tabpfn", "PFN-v2", "tabicl", "mitra", "tabiclv2", "tabpfn3", "tabpfn25"}


def find_files(root):
    npz = sorted(glob.glob(os.path.join(glob.escape(root), "**", "cache", "*.npz"), recursive=True))
    csvs = sorted(
        p
        for p in glob.glob(os.path.join(glob.escape(root), "**", "talent_benchmark__*.csv"), recursive=True)
        if not p.endswith("__errors.csv")
    )
    return npz, csvs


def load_rows(csvs):
    frames = []
    for p in csvs:
        try:
            df = pd.read_csv(p, on_bad_lines="error")
        except Exception as exc:  # noqa: BLE001
            print(f"CSV UNREADABLE {p}: {exc!r}")
            continue
        df["_csv"] = p
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=KEY)
    rows = pd.concat(frames, ignore_index=True)
    rows["seed"] = rows["seed"].astype(int)
    # CSVs written before the --tag flag have no column: those rows are base cells
    if "tag" not in rows.columns:
        rows["tag"] = ""
    rows["tag"] = rows["tag"].fillna("").astype(str)
    return rows


def rebuild_metrics(d, score):
    from utils import evaluate_classification

    y_set, qhat = conformal_from_cache(d, score)
    p_test = d["p_test"]
    m = evaluate_classification(p_test.argmax(1), d["y_test"], y_set, p_test)
    return m, qhat


def check_data_hash(d, data_path):
    """Recompute the feature hashes from TALENT's on-disk arrays."""
    from model.lib.data import get_dataset
    from utils import concat_features

    meta = d["meta"]
    (N_tv, C_tv, _), (N_te, C_te, _), _ = get_dataset(meta["dataset"], data_path)
    N_pool = np.concatenate([N_tv["train"], N_tv["val"]]) if N_tv else None
    C_pool = np.concatenate([C_tv["train"], C_tv["val"]]) if C_tv else None
    X_pool = concat_features(C_pool, N_pool)
    X_test = concat_features(C_te["test"] if C_te else None, N_te["test"] if N_te else None)
    return (
        sha1_array(X_pool[d["idx_cal"]]) == d["x_cal_sha1"],
        sha1_array(X_test[d["idx_test"]]) == d["x_test_sha1"],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--data-path", default=None, help="verify x_*_sha1 against data/<dataset>")
    ap.add_argument("--rebuild-manifest", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    npz_files, csv_files = find_files(args.out_dir)
    rows = load_rows(csv_files)
    print(f"{len(npz_files)} npz, {len(csv_files)} csv, {len(rows)} rows under {args.out_dir}")
    if not npz_files:
        print("FAIL: no cache files found")
        return 1

    dup = rows.duplicated(KEY, keep=False) if len(rows) else pd.Series(dtype=bool)
    failures = 0
    if dup.any():
        print(f"FAIL: {int(dup.sum())} duplicate CSV rows on {KEY}")
        print(rows.loc[dup, KEY + ["_csv"]].to_string())
        failures += 1

    rows_idx = rows.set_index(KEY) if len(rows) else None
    seen_keys = set()
    flags = []
    cells = []
    n_checked = 0
    max_abs = 0.0
    manifest_rows = []
    for path in npz_files:
        d = read_cache(path)
        meta = d["meta"]
        name = parse_cache_name(path)
        ident = (meta["dataset"], meta["task_type"], meta["model"], meta.get("tag") or "", int(meta["seed"]))
        if (name["dataset"], name["task_type"], name["model"], name["tag"], name["seed"]) != ident:
            print(f"FAIL: file name {os.path.basename(path)} disagrees with meta {ident}")
            failures += 1
        cells.append(dict(zip(["dataset", "task_type", "model", "tag", "seed"], ident), mock=meta.get("mock_run")))

        if not (np.allclose(d["p_cal"].sum(1), 1, atol=1e-4) and np.allclose(d["p_test"].sum(1), 1, atol=1e-4)):
            print(f"FAIL: probabilities do not sum to 1 in {path}")
            failures += 1
        if meta.get("mock_run"):
            flags.append(("mock_run", ident))
        be, me = meta.get("best_epoch"), meta.get("max_epoch")
        if be is not None and me is not None and be >= me - 1:
            flags.append(("budget_bound", ident + (be, me)))
        if not meta.get("hpo_ran") and meta["model"] not in MODELS_WITHOUT_HPO and not meta.get("mock_run"):
            flags.append(("hpo_not_run", ident + (meta.get("hpo_skipped_reason"),)))

        if args.data_path:
            ok_cal, ok_test = check_data_hash(d, args.data_path)
            if not (ok_cal and ok_test):
                print(f"FAIL: feature hash mismatch (cal={ok_cal}, test={ok_test}) for {path}")
                failures += 1

        for score in meta["conformity_scores"]:
            key = ident + (score,)
            seen_keys.add(key)
            m, qhat = rebuild_metrics(d, score)
            if score in d["qhat"] and qhat is not None and not np.allclose(qhat, d["qhat"][score], atol=args.tol):
                print(f"FAIL: q-hat drift for {key}: stored {d['qhat'][score]} rebuilt {qhat}")
                failures += 1
            if rows_idx is None or key not in rows_idx.index:
                print(f"FAIL: no CSV row for {key}")
                failures += 1
                continue
            row = rows_idx.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            worst = 0.0
            for col, val in m.items():
                if col not in row.index:
                    print(f"FAIL: column {col} missing from CSV for {key}")
                    failures += 1
                    continue
                ref = float(row[col])
                if np.isnan(ref) and np.isnan(val):
                    continue
                diff = abs(float(val) - ref)
                worst = max(worst, diff)
                if not diff <= args.tol:
                    print(f"FAIL: {key} {col}: csv={ref!r} cache={val!r} |diff|={diff:.3g}")
                    failures += 1
            max_abs = max(max_abs, worst)
            n_checked += 1
            if not args.quiet:
                print(f"ok   {key} max|diff|={worst:.2e}")

        manifest_rows.append(
            {
                **meta,
                "cache_file": os.path.basename(path),
                "x_cal_sha1": d["x_cal_sha1"],
                "x_test_sha1": d["x_test_sha1"],
            }
        )

    if rows_idx is not None:
        orphan = [k for k in rows_idx.index if tuple(k) not in seen_keys]
        if orphan:
            print(f"FAIL: {len(orphan)} CSV rows without a cache cell, e.g. {orphan[:3]}")
            failures += 1

    if args.rebuild_manifest:
        cache_dirs = sorted({os.path.dirname(p) for p in npz_files})
        for cd in cache_dirs:
            mpath = os.path.join(cd, "manifest.csv")
            if os.path.exists(mpath):
                os.remove(mpath)
            for r in manifest_rows:
                if os.path.exists(os.path.join(cd, r["cache_file"])):
                    append_manifest(mpath, r)
        print(f"manifest rebuilt in {len(cache_dirs)} cache dir(s) ({', '.join(MANIFEST_COLUMNS[:5])}, ...)")

    cells_df = pd.DataFrame(cells)
    if len(cells_df):
        cells_df["model"] = [m + (f"@{t}" if t else "") for m, t in zip(cells_df["model"], cells_df["tag"])]
        mat = cells_df.groupby(["dataset", "task_type", "model"])["seed"].nunique().unstack("model", fill_value=0)
        print("\ncompleteness (seeds per dataset x model):")
        print(mat.to_string())
    for kind, what in flags:
        print(f"flag {kind}: {what}")

    print(f"\nchecked {n_checked} (cell, score) pairs; max |diff| = {max_abs:.3g}; failures = {failures}")
    if failures:
        print("RESULT: FAIL")
        return 1
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
