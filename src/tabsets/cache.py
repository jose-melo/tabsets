"""Read the probability cache written by ``cache_io.write_cache``.

One ``.npz`` per ``(dataset, task_type, model[@tag], seed, machine)`` with
arrays ``p_cal, y_cal, p_test, y_test, idx_cal, idx_test, classes``, the
feature hashes ``x_cal_sha1 / x_test_sha1``, a JSON ``meta`` blob and one
``qhat_<score>`` array per score the benchmark ran. The layout is owned by
``cache_io.py`` at the repository root; this module only *reads* it and is
kept dependency-free so the package works outside the benchmark tree.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

TAG_SEP = "@"  # same as cache_io.TAG_SEP

# Manifest columns, in order. Everything after ``n_test`` is read from meta.
MANIFEST_COLUMNS = [
    "cache_file", "dataset", "task_type", "model", "tag", "seed", "machine",
    "git_sha", "n_classes", "n_cal", "n_test", "n_train", "env_tag", "mock_run",
    "hpo_ran", "n_trials", "n_trials_ok", "hpo_constant_trials", "hpo_best_value", "hpo_objective", "t_hpo_s", "t_fit_s", "t_pred_cal_s",
    "t_pred_test_s", "best_epoch", "max_epoch", "val_frac", "n_estimators",
    "softmax_temperature", "model_version", "checkpoint", "confidence_level",
    "mapie_random_state", "conformity_scores", "qhat_scores", "schema_version",
    "written_at", "x_cal_sha1", "x_test_sha1",
]


@dataclass
class Cell:
    """One cached (dataset, model, seed) cell: the fitted model's output."""

    path: str
    dataset: str
    task_type: str
    model: str
    tag: str
    seed: int
    machine: str
    p_cal: np.ndarray
    y_cal: np.ndarray
    p_test: np.ndarray
    y_test: np.ndarray
    idx_cal: np.ndarray
    idx_test: np.ndarray
    classes: np.ndarray
    x_cal_sha1: str
    x_test_sha1: str
    meta: dict
    qhat: dict = field(default_factory=dict)

    # -- derived -------------------------------------------------------- #
    @property
    def n_classes(self) -> int:
        return int(self.p_cal.shape[1])

    @property
    def n_cal(self) -> int:
        return int(self.p_cal.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.p_test.shape[0])

    @property
    def is_binary(self) -> bool:
        return self.n_classes == 2

    @property
    def confidence_level(self) -> float:
        return float(self.meta.get("confidence_level", 0.9))

    @property
    def alpha(self) -> float:
        return 1.0 - self.confidence_level

    @property
    def mapie_random_state(self) -> int:
        rs = self.meta.get("mapie_random_state")
        return int(self.seed if rs is None else rs)

    @property
    def key(self) -> tuple:
        return (self.dataset, self.task_type, self.model, self.tag, self.seed)

    @property
    def name(self) -> str:
        seg = f"{self.model}{TAG_SEP}{self.tag}" if self.tag else self.model
        return f"{self.dataset}__{self.task_type}__{seg}__seed{self.seed}__{self.machine}"

    def id_fields(self) -> dict:
        return dict(
            dataset=self.dataset, task_type=self.task_type, n_classes=self.n_classes,
            model=self.model, tag=self.tag, seed=self.seed, machine=self.machine,
            git_sha=self.meta.get("git_sha"), env_tag=self.meta.get("env_tag"),
            mock_run=bool(self.meta.get("mock_run", False)), n_cal=self.n_cal, n_test=self.n_test,
        )


def parse_name(path: str) -> dict:
    """``<dataset>__<task_type>__<model[@tag]>__seed<seed>__<machine>.npz`` -> dict.
    Mirrors ``cache_io.parse_cache_name`` (dataset names may contain ``_``,
    hence the ``rsplit`` from the right)."""
    stem = os.path.basename(path)
    if stem.endswith(".npz"):
        stem = stem[:-4]
    dataset, task_type, seg, seed, machine = stem.rsplit("__", 4)
    model, _, tag = seg.partition(TAG_SEP)
    return dict(dataset=dataset, task_type=task_type, model=model, tag=tag,
                seed=int(seed[len("seed"):]), machine=machine)


def load_npz(path: str) -> Cell:
    """Read one cell. Labels are returned as ``int64``; probabilities untouched
    (float64 as written)."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays.pop("meta")))
    qhat = {k[len("qhat_"):]: np.asarray(arrays.pop(k), dtype=np.float64)
            for k in list(arrays) if k.startswith("qhat_")}
    name = parse_name(path)
    return Cell(
        path=os.path.abspath(path),
        dataset=meta.get("dataset", name["dataset"]),
        task_type=meta.get("task_type", name["task_type"]),
        model=meta.get("model", name["model"]),
        tag=str(meta.get("tag") or name["tag"] or ""),
        seed=int(meta.get("seed", name["seed"])),
        machine=str(meta.get("machine", name["machine"])),
        p_cal=np.asarray(arrays["p_cal"], dtype=np.float64),
        y_cal=np.asarray(arrays["y_cal"]).astype(np.int64),
        p_test=np.asarray(arrays["p_test"], dtype=np.float64),
        y_test=np.asarray(arrays["y_test"]).astype(np.int64),
        idx_cal=np.asarray(arrays["idx_cal"]),
        idx_test=np.asarray(arrays["idx_test"]),
        classes=np.asarray(arrays["classes"]).astype(np.int64),
        x_cal_sha1=str(arrays.get("x_cal_sha1", "")),
        x_test_sha1=str(arrays.get("x_test_sha1", "")),
        meta=meta,
        qhat=qhat,
    )


def find_npz(cache_dir: str, recursive: bool = True) -> list:
    """All ``*.npz`` under ``cache_dir`` (recursively: any ``cache/`` subdir
    too, so a results root with several ``<job>/cache/`` dirs works)."""
    root = glob.escape(os.path.abspath(cache_dir))
    files = set(glob.glob(os.path.join(root, "*.npz")))
    if recursive:
        files |= set(glob.glob(os.path.join(root, "**", "*.npz"), recursive=True))
    return sorted(f for f in files if not f.endswith(".tmp.npz"))


def load_dir(cache_dir: str, recursive: bool = True, mock: bool = False) -> list:
    """Every cell under ``cache_dir`` as a list of :class:`Cell`, sorted by
    file name. ``mock=False`` drops ``mock_run`` cells (the benchmark's smoke
    mode) - they carry random probabilities and must never enter a table."""
    cells = [load_npz(p) for p in find_npz(cache_dir, recursive=recursive)]
    if not mock:
        cells = [c for c in cells if not c.meta.get("mock_run")]
    return cells


def manifest(cells) -> pd.DataFrame:
    """One row per cell with the identifying fields and the meta fields the
    paper reports (HPO status, timings, epochs, TFM settings, versions)."""
    rows = []
    for c in cells:
        m = c.meta
        row = {k: m.get(k) for k in MANIFEST_COLUMNS}
        row.update(dict(
            cache_file=os.path.basename(c.path), dataset=c.dataset, task_type=c.task_type,
            model=c.model, tag=c.tag, seed=c.seed, machine=c.machine, n_classes=c.n_classes,
            n_cal=c.n_cal, n_test=c.n_test, x_cal_sha1=c.x_cal_sha1, x_test_sha1=c.x_test_sha1,
            conformity_scores="|".join(m.get("conformity_scores", []) or []),
            qhat_scores="|".join(sorted(c.qhat)),
        ))
        # package versions, flattened
        for pkg, v in (m.get("versions") or {}).items():
            row[f"v_{pkg}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["dataset", "task_type", "model", "tag", "seed"]).reset_index(drop=True)
    return df


def completeness(cells) -> pd.DataFrame:
    """Seeds per (dataset, task_type) x model, as ``check_cache.py`` prints it."""
    if not cells:
        return pd.DataFrame()
    df = pd.DataFrame([dict(dataset=c.dataset, task_type=c.task_type,
                            model=c.model + (f"@{c.tag}" if c.tag else ""), seed=c.seed) for c in cells])
    return df.groupby(["dataset", "task_type", "model"])["seed"].nunique().unstack("model", fill_value=0)


class CachedProbaClassifier(BaseEstimator, ClassifierMixin):
    """A fitted "classifier" whose predict_proba is a table lookup.

    MAPIE's ``SplitConformalClassifier(prefit=True)`` only ever calls
    ``predict_proba``/``predict`` and reads ``classes_``. ``X`` is an
    ``(n, 1)`` integer array of row indices into ``table``.
    """

    def __init__(self, table, classes):
        self.table = np.asarray(table)
        self.classes_ = np.asarray(classes)
        self._is_fitted = True

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return self.table[np.asarray(X, dtype=np.int64).ravel()]

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]

    def __sklearn_is_fitted__(self):
        return True


# --------------------------------------------------------------------------- #
# the released layout: many cells to a parquet shard
# --------------------------------------------------------------------------- #
# One npz per cell is the right shape for a machine that writes them one at a
# time and the wrong shape for distributing 45,000 of them. A shard holds many
# cells as rows, with each array flattened into a list column. Probabilities
# stay float64: a set is a comparison against a threshold, and rounding the
# thing being compared would move sets that sit on it.

ROW_ARRAYS = ("p_cal", "p_test", "y_cal", "y_test", "idx_cal", "idx_test", "classes")


def cell_to_row(cell: "Cell") -> dict:
    """One cell as a flat record. Reversed exactly by :func:`row_to_cell`."""
    return dict(
        name=cell.name, dataset=cell.dataset, task_type=cell.task_type, model=cell.model,
        tag=cell.tag, seed=int(cell.seed), machine=cell.machine,
        n_cal=cell.n_cal, n_test=cell.n_test, n_classes=cell.n_classes,
        p_cal=cell.p_cal.ravel().tolist(), p_test=cell.p_test.ravel().tolist(),
        y_cal=cell.y_cal.tolist(), y_test=cell.y_test.tolist(),
        idx_cal=cell.idx_cal.tolist(), idx_test=cell.idx_test.tolist(),
        classes=cell.classes.tolist(),
        idx_dtype=str(cell.idx_cal.dtype),
        x_cal_sha1=cell.x_cal_sha1, x_test_sha1=cell.x_test_sha1,
        meta=json.dumps(cell.meta, sort_keys=True),
        qhat=json.dumps({k: np.asarray(v).tolist() for k, v in cell.qhat.items()}, sort_keys=True),
    )


def row_to_cell(row, path: str = "") -> "Cell":
    """A shard row back into a cell, with the arrays as :func:`load_npz` returns them."""
    K = int(row["n_classes"])
    idx_dtype = np.dtype(row.get("idx_dtype") or "int64")
    return Cell(
        path=path or f"{row['name']}.npz",
        dataset=row["dataset"], task_type=row["task_type"], model=row["model"],
        tag=row["tag"] or "", seed=int(row["seed"]), machine=row["machine"],
        p_cal=np.asarray(row["p_cal"], dtype=np.float64).reshape(int(row["n_cal"]), K),
        y_cal=np.asarray(row["y_cal"], dtype=np.int64),
        p_test=np.asarray(row["p_test"], dtype=np.float64).reshape(int(row["n_test"]), K),
        y_test=np.asarray(row["y_test"], dtype=np.int64),
        idx_cal=np.asarray(row["idx_cal"], dtype=idx_dtype),
        idx_test=np.asarray(row["idx_test"], dtype=idx_dtype),
        classes=np.asarray(row["classes"], dtype=np.int64),
        x_cal_sha1=row["x_cal_sha1"], x_test_sha1=row["x_test_sha1"],
        meta=json.loads(row["meta"]),
        qhat={k: np.asarray(v, dtype=np.float64) for k, v in json.loads(row["qhat"]).items()},
    )


def write_shard(cells, path: str) -> int:
    """Write cells to one parquet shard. Returns the file size in bytes."""
    pd.DataFrame([cell_to_row(c) for c in cells]).to_parquet(
        path, index=False, compression="zstd")
    return os.path.getsize(path)


def read_shard(path: str, names=None):
    """Cells from a shard, or only those named.

    Naming them pushes the filter into the file, so only the row groups that hold them
    are read. Without that, asking for one cell costs the whole shard, which is hundreds
    of megabytes, and a loop over cells becomes quadratic in the shard size.
    """
    if names is None:
        df = pd.read_parquet(path)
    else:
        names = list(dict.fromkeys(names))
        df = pd.read_parquet(path, filters=[("name", "in", set(names))])
    return [row_to_cell(r, path=f"{path}::{r['name']}") for _, r in df.iterrows()]
