"""The cell index of the release, and the blocks the article computes on.

A *cell* is one run: one model, on one dataset, at one seed, with its calibration and
test probabilities. The manifest lists every cell; a *block* is a subset of it on which
every model is complete at ten seeds, so that a paired comparison across models has the
same datasets on both sides.

Five things here are corrections, each of which produced a wrong number once:

* the model key comes from the manifest, never from the file name: cells written with a
  checkpoint tag record ``mitra`` in their own header where the run is ``mitra@v2``;
* families come from the roster (:mod:`tabsets.stats`), which the manifest is checked
  against on load;
* the number of classes comes from the dataset covariates, because the manifest records
  ``K = 1`` on five datasets whose cells hold two columns of probabilities, and the
  index's own ``n_classes`` repeats that error rather than contradicting it;
* sets are built with the cell's own calibration seed, never with a fixed one, or every
  model in a split shares a tie-break it did not draw;
* averaging runs seed -> cell -> dataset -> family, and never pools runs directly.

Blocks are recomputed from the manifest on every call rather than frozen, so a block
grows as the queue drains. The article names the block behind every table for that
reason, and every pass of :mod:`reproduce` takes a ``--block`` argument.
"""
from __future__ import annotations

import os
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd

from .stats import FAMILY_OF_MODEL, ORDER, roster

PKG_DATA = os.path.join(os.path.dirname(__file__), "data")

#: Models excluded from the many-class comparison block, which only seven models reach.
DK10_MODELS = ["tabpfn3", "tabiclv2", "catboost", "lightgbm", "xgboost", "LogReg", "knn"]

#: Dropped from the main block: three datasets whose conference-era splits cannot be
#: reproduced from the recorded indices.
DROP_F1 = ["Fitness_Club_c", "VulNoneVul", "ringnorm"]

TFM = [m for m in ORDER if FAMILY_OF_MODEL.get(m) == "TFM"]
GBDT = [m for m in ORDER if FAMILY_OF_MODEL.get(m) == "GBDT"]
DEEP = [m for m in ORDER if FAMILY_OF_MODEL.get(m) == "deep"]
CLASSIC = [m for m in ORDER if FAMILY_OF_MODEL.get(m) == "classic"]

#: The core foundation-model cluster: the thirteen without the two first-generation
#: checkpoints, which sit apart from the rest on every statistic of the article.
TFM_CORE = [m for m in TFM if m not in ("tabpfn", "mitra")]

_MAN = None
_COV = None


def cache_root() -> str:
    """Where the probability cells live. ``TABSETS_CACHE``, or a cache under ``$HOME``."""
    return os.path.expanduser(os.environ.get("TABSETS_CACHE", "~/.cache/tabsets/cells"))


def data_root() -> str:
    """Where the manifest lives. ``TABSETS_DATA``, or the copy shipped with the source."""
    env = os.environ.get("TABSETS_DATA")
    if env:
        return os.path.expanduser(env)
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "reproduce", "data")
    return here if os.path.isdir(here) else PKG_DATA


def covariates() -> pd.DataFrame:
    """One row per dataset: size, number of classes, class balance, and the unit maps."""
    global _COV
    if _COV is None:
        _COV = pd.read_csv(os.path.join(PKG_DATA, "dataset_covariates.csv"), low_memory=False)
    return _COV.copy()


def manifest(path=None) -> pd.DataFrame:
    """Every cell of the release, with ``path`` resolved against :func:`cache_root`.

    Raises if the manifest's family column disagrees with the roster, which is the check
    that would have caught the family map drifting away from the model table.
    """
    global _MAN
    if _MAN is None or path is not None:
        src = path
        if src is None:
            for name in ("manifest.parquet", "manifest.csv"):
                cand = os.path.join(data_root(), name)
                if os.path.exists(cand):
                    src = cand
                    break
        if src is None:
            raise FileNotFoundError(
                f"no manifest in {data_root()}; set TABSETS_DATA or pass path=")
        m = pd.read_parquet(src) if src.endswith(".parquet") else pd.read_csv(src, low_memory=False)

        cov = covariates()[["dataset", "K", "is_binary"]].rename(columns={"K": "K_true"})
        m = m.merge(cov, on="dataset", how="left")
        m["K"] = m["K_true"].fillna(m["K"]).astype(int)
        m = m.drop(columns=["K_true"])
        # ``n_classes`` carries the same defect as ``K`` and from the same source, so
        # correcting one and not the other leaves the index disagreeing with itself:
        # a reader filtering on n_classes would drop the cells whose K was just fixed.
        # The cells themselves are right either way, since a Cell reads its class count
        # off the width of its probability matrix rather than off any metadata.
        if "n_classes" in m:
            m["n_classes"] = m["K"]
        m["seed_int"] = m["seed"].astype(str).str.replace("seed", "", regex=False).astype(np.int64)

        bad = m.loc[m["family"] != m["model"].map(FAMILY_OF_MODEL), "model"].unique()
        if len(bad):
            raise ValueError(f"manifest family disagrees with the roster for {sorted(bad)}")

        if not m["path"].iloc[0].startswith("/"):
            m["path"] = [os.path.join(cache_root(), p) for p in m["path"]]
        if path is not None:
            return m
        _MAN = m
    return _MAN.copy()


def units(kind="strict") -> dict:
    """dataset -> unit id, for collapsing near-duplicate datasets before a paired test.

    ``strict`` groups datasets that share a source and differ only in preprocessing or
    in the target column; ``orig`` is the coarser grouping the conference version used.
    """
    cov = covariates()
    col = "family_strict" if kind == "strict" else "family_orig"
    return dict(zip(cov.dataset, cov[col]))


def block(name: str, man: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The cells of a named block: the datasets on which every model of that block has
    all ten seeds.

    A block is defined by completeness, not by the roster. It is not a claim that the
    benchmark is a full grid, and it is not one: eleven datasets carry more than ten
    classes and the in-context models measured on them cannot enter them, one foundation
    model is measured on the original datasets only, and runs are still landing. A block
    is the largest rectangle inside that, which is what a paired comparison needs.

    ``F1``    every model on the real datasets inside the conference envelope
    ``allA``  every model on the published conference datasets alone
    ``C39``   every model on the synthetic datasets
    ``D29``   the models complete on the new real datasets outside the envelope
    ``D7``    the many-class subset's seven models on those datasets
    ``DK10``  the same seven, restricted to more than ten classes
    """
    man = manifest() if man is None else man
    m = man[man.complete10]
    if name == "F1":
        b = m[m.universe.isin(["A_published112", "B_tierA_real4"]) & ~m.dataset.isin(DROP_F1)]
        return _complete(b, ORDER)
    if name == "allA":
        return _complete(m[m.universe == "A_published112"], ORDER)
    if name == "C39":
        b = m[(m.universe == "C_synthetic40") & (m.dataset != "synthetic_gbdt_favorable_016")]
        return _complete(b, ORDER)
    if name == "D29":
        return _complete(m[m.universe == "D_new62"], [x for x in ORDER if x != "exaone"])
    if name == "D7":
        return _complete(m[m.universe == "D_new62"], DK10_MODELS)
    if name == "DK10":
        return _complete(m[(m.universe == "D_new62") & (m.K > 10)], DK10_MODELS)
    raise ValueError(f"unknown block {name!r}")


def _complete(b: pd.DataFrame, models) -> pd.DataFrame:
    b = b[b.model.isin(models)]
    t = b.groupby(["dataset", "model"]).seed.nunique().unstack().reindex(columns=list(models))
    ok = (t == 10).all(axis=1)
    return b[b.dataset.isin(ok[ok].index)].copy()


_SHARDS = None


def shard_index():
    """name -> shard, when the cache is the released one. ``None`` for a directory of files.

    The release ships the cells packed into a few parquet shards with an index beside
    them; a machine that wrote them writes one file per cell instead. Both are read.
    """
    global _SHARDS
    if _SHARDS is None:
        idx = os.path.join(cache_root(), "index.parquet")
        _SHARDS = pd.read_parquet(idx).set_index("name")["shard"].to_dict() if os.path.exists(idx) else {}
    return _SHARDS or None


def load(path):
    """One cell, from a file of its own or from the shard that holds it."""
    from . import cache

    shards = shard_index()
    if shards is None:
        return cache.load_npz(path)
    name = os.path.basename(path)
    name = name[:-4] if name.endswith(".npz") else name
    shard = shards.get(name)
    if shard is None:
        raise KeyError(f"{name} is not in the released cache index")
    got = cache.read_shard(os.path.join(cache_root(), shard), names=[name])
    if not got:
        raise KeyError(f"{name} is indexed to {shard} but absent from it")
    return got[0]


def map_cells(fn, blk: pd.DataFrame, workers: int = 7, chunksize: int = 16,
              key_cols=("path", "dataset", "task_type", "universe", "model", "family", "seed",
                        "K", "is_binary", "n_cal", "n_test")) -> pd.DataFrame:
    """Apply ``fn(path) -> dict | list[dict]`` to every cell of a block, in parallel.

    ``fn`` has to be importable at module level: the worker processes are spawned, not
    forked, on macOS. The rows come back joined to the manifest keys on ``path``.
    """
    paths = list(blk.path)
    with Pool(workers) as pool:
        out = list(pool.imap_unordered(fn, paths, chunksize=chunksize))
    rows = []
    for o in out:
        if o is None:
            continue
        rows.extend(o if isinstance(o, list) else [o])
    df = pd.DataFrame(rows)
    keys = [c for c in key_cols if c in blk.columns]
    return blk[keys].merge(df, on="path", how="inner")


def split_jobs(blk: pd.DataFrame):
    """Yield ``(dataset, seed, {model: path})`` for work that compares models point by point.

    Every model in one job was fitted on the same split, so their test rows line up.
    """
    for (d, s), g in blk.groupby(["dataset", "seed"]):
        yield d, s, dict(zip(g.model, g.path))


def check_split(cells: dict):
    """Assert that the models of one job really do share their calibration and test labels."""
    ys = [c.y_test for c in cells.values()]
    yc = [c.y_cal for c in cells.values()]
    if not all(np.array_equal(ys[0], y) for y in ys):
        raise AssertionError("y_test differs across the models of one split")
    if not all(np.array_equal(yc[0], y) for y in yc):
        raise AssertionError("y_cal differs across the models of one split")


def roster_of(block_name: str) -> pd.DataFrame:
    """The roster rows for the models present in a block."""
    present = set(block(block_name).model)
    r = roster()
    return r[r.model.isin(present)]
