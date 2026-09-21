"""Probability cache for the TALENT/HPLR benchmark.

This is the only module that knows the ``.npz`` layout. One file per
``(dataset, task_type, model, seed)`` holds the class probabilities the fitted
model produced on the calibration and on the test split, plus the labels, the
row indices into TALENT's on-disk arrays, and a JSON ``meta`` blob. Every
conformal set (any score, any confidence level) and every calibration metric
is a function of these arrays, so the expensive fit never has to be repeated.

``CachedProbaClassifier`` + ``conformal_from_cache`` run MAPIE's
``SplitConformalClassifier(prefit=True)`` on the stored probabilities through
exactly the code path the live benchmark uses; the CSV rows written by
``talent_benchmark.py`` are computed from the cache read back from disk, so
``check_cache.py`` can rebuild them bit for bit.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import io
import json
import os
import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

SCHEMA_VERSION = 1

# Arrays stored in every npz, in this order.
ARRAY_KEYS = ("p_cal", "y_cal", "p_test", "y_test", "idx_cal", "idx_test", "classes")

MANIFEST_COLUMNS = [
    "cache_file", "dataset", "task_type", "n_classes", "model", "tag", "seed", "machine",
    "env_tag", "git_sha", "n_train", "n_cal", "n_test", "x_cal_sha1", "x_test_sha1",
    "mock_run", "hpo_ran", "n_trials", "n_trials_ok", "t_hpo_s", "t_fit_s",
    "t_pred_cal_s", "t_pred_test_s", "best_epoch", "max_epoch", "n_estimators",
    "softmax_temperature", "model_version", "checkpoint", "written_at",
]

# A cache tag (``--tag ne8``) rides in the model segment of the file name as
# ``<model>@<tag>``: one character that no model name uses, so the five
# ``__``-separated segments and every glob / rsplit below stay as they are.
TAG_SEP = "@"


def model_segment(model, tag=None) -> str:
    if tag:
        tag = str(tag)
        assert TAG_SEP not in tag and "__" not in tag and "/" not in tag, tag
        return f"{model}{TAG_SEP}{tag}"
    return str(model)


def split_model_segment(segment) -> tuple:
    model, _, tag = str(segment).partition(TAG_SEP)
    return model, tag


# --------------------------------------------------------------------------- #
# hashing / naming
# --------------------------------------------------------------------------- #
def sha1_array(X) -> str:
    """sha1 of the feature matrix, stable across dtype/contiguity.

    Numeric arrays are hashed as C-contiguous float64 bytes. TALENT feature
    matrices that mix categorical strings and numbers arrive as ``object``
    arrays; those are hashed through their ``str`` rendering, row by row.
    """
    X = np.asarray(X)
    if X.dtype.kind in "biuf":
        return hashlib.sha1(
            np.ascontiguousarray(X, dtype=np.float64).tobytes()
        ).hexdigest()
    h = hashlib.sha1()
    for row in X.astype(str):
        h.update("\x1f".join(row).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def cache_name(dataset, task_type, model, seed, machine, tag=None) -> str:
    # ``task_type`` is part of the key: TALENT ships two datasets called
    # ``eye_movements`` (OpenML 1044 multiclass / 44130 binclass), and the same
    # for ``house_16H`` and ``pol``. ``tag`` (a sweep run) becomes ``model@tag``.
    seg = model_segment(model, tag)
    return f"{dataset}__{task_type}__{seg}__seed{int(seed)}__{machine}.npz"


def cache_path(cache_dir, dataset, task_type, model, seed, machine, tag=None) -> str:
    return os.path.join(
        cache_dir, cache_name(dataset, task_type, model, seed, machine, tag=tag)
    )


def cache_exists(cache_dir, dataset, task_type, model, seed, tag=None):
    """Resume check: a cell is done if an npz exists under *any* machine tag.
    A tagged cell (``--tag``) and the untagged base cell are different cells."""
    # dataset names contain "(", "[" and similar glob metacharacters
    seg = model_segment(model, tag)
    prefix = glob.escape(f"{dataset}__{task_type}__{seg}__seed{int(seed)}__")
    hits = sorted(glob.glob(os.path.join(glob.escape(cache_dir), prefix + "*.npz")))
    return hits[0] if hits else None


def parse_cache_name(path) -> dict:
    stem = os.path.basename(path)
    if stem.endswith(".npz"):
        stem = stem[:-4]
    dataset, task_type, seg, seed, machine = stem.rsplit("__", 4)
    model, tag = split_model_segment(seg)
    return {
        "dataset": dataset,
        "task_type": task_type,
        "model": model,
        "tag": tag,
        "seed": int(seed[len("seed"):]),
        "machine": machine,
    }


# --------------------------------------------------------------------------- #
# read / write
# --------------------------------------------------------------------------- #
def write_cache(
    path,
    *,
    p_cal,
    y_cal,
    p_test,
    y_test,
    idx_cal,
    idx_test,
    classes,
    X_cal,
    X_test,
    meta: dict,
    qhat: dict | None = None,
) -> str:
    """Atomically write one cell. ``qhat`` maps score name -> MAPIE quantiles_.

    Probabilities are stored as float64 on purpose: the CSV rows are computed
    from the read-back arrays, so a float32 cast would put the cache one
    rounding away from the model's own output. The whole 112-dataset grid is
    < 1 GB at float64.
    """
    p_cal = np.asarray(p_cal, dtype=np.float64)
    p_test = np.asarray(p_test, dtype=np.float64)
    y_cal = np.asarray(y_cal, dtype=np.int16)
    y_test = np.asarray(y_test, dtype=np.int16)
    classes = np.asarray(classes, dtype=np.int16)
    assert np.array_equal(classes, np.arange(len(classes))), (
        "classes must be positional (0..K-1 = column order of p_*); raw label "
        "values belong in meta['class_labels']"
    )
    assert p_cal.ndim == 2 and p_test.ndim == 2, "p_* must be (n, K)"
    assert p_cal.shape[1] == p_test.shape[1] == len(classes), "K mismatch"
    assert p_cal.shape[0] == len(y_cal) == len(idx_cal), "calibration size mismatch"
    assert p_test.shape[0] == len(y_test) == len(idx_test), "test size mismatch"
    assert np.allclose(p_cal.sum(1), 1, atol=1e-4), "p_cal rows do not sum to 1"
    assert np.allclose(p_test.sum(1), 1, atol=1e-4), "p_test rows do not sum to 1"

    meta = {
        **meta,
        "schema_version": SCHEMA_VERSION,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "qhat_scores": sorted(qhat.keys()) if qhat else [],
    }
    arrays = dict(
        p_cal=p_cal,
        y_cal=y_cal,
        p_test=p_test,
        y_test=y_test,
        idx_cal=np.asarray(idx_cal, dtype=np.int32),
        idx_test=np.asarray(idx_test, dtype=np.int32),
        classes=classes,
        x_cal_sha1=np.array(sha1_array(X_cal)),
        x_test_sha1=np.array(sha1_array(X_test)),
        meta=np.array(json.dumps(meta, default=_json_default)),
    )
    for score, q in (qhat or {}).items():
        arrays[f"qhat_{score}"] = np.asarray(q, dtype=np.float64)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)  # a killed job never leaves a half-written npz
    return path


def read_cache(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    d["x_cal_sha1"] = str(d["x_cal_sha1"])
    d["x_test_sha1"] = str(d["x_test_sha1"])
    d["qhat"] = {
        k[len("qhat_"):]: d.pop(k) for k in list(d) if k.startswith("qhat_")
    }
    return d


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# MAPIE on stored probabilities
# --------------------------------------------------------------------------- #
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


def conformal_from_arrays(p_cal, y_cal, p_test, classes, score, confidence_level, seed):
    """One score from probabilities. Same MAPIE path as the live benchmark.

    Returns ``(y_set, qhat)`` with ``y_set`` in MAPIE's own
    ``(n_test, K, n_alpha)`` boolean layout and ``qhat`` MAPIE's ``quantiles_``.
    """
    from mapie.classification import SplitConformalClassifier

    p_cal = np.asarray(p_cal)
    p_test = np.asarray(p_test)
    n_cal, n_test = len(p_cal), len(p_test)
    est = CachedProbaClassifier(np.vstack([p_cal, p_test]), classes)
    mc = SplitConformalClassifier(
        estimator=est,
        confidence_level=confidence_level,
        prefit=True,
        conformity_score=score,
        random_state=seed,
    )
    mc.conformalize(np.arange(n_cal)[:, None], np.asarray(y_cal))
    _, y_set = mc.predict_set(np.arange(n_cal, n_cal + n_test)[:, None])
    qhat = getattr(mc._mapie_classifier, "quantiles_", None)
    return y_set, (None if qhat is None else np.asarray(qhat, dtype=np.float64))


def conformal_from_cache(d, score, confidence_level=None, seed=None):
    """``d`` is a ``read_cache`` dict. Defaults come from ``meta``."""
    if confidence_level is None:
        confidence_level = d["meta"]["confidence_level"]
    if seed is None:
        seed = d["meta"]["mapie_random_state"]
    return conformal_from_arrays(
        d["p_cal"], d["y_cal"], d["p_test"], d["classes"], score, confidence_level, seed
    )


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def append_manifest(manifest_csv, row: dict) -> None:
    """One line per npz, written with a single ``write()`` so that concurrent
    jobs appending to the same manifest do not tear lines."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
    w.writerow({k: row.get(k) for k in MANIFEST_COLUMNS})
    os.makedirs(os.path.dirname(os.path.abspath(manifest_csv)), exist_ok=True)
    new = not os.path.exists(manifest_csv) or os.path.getsize(manifest_csv) == 0
    with open(manifest_csv, "a") as f:
        f.write((",".join(MANIFEST_COLUMNS) + "\n" if new else "") + buf.getvalue())


def package_versions() -> dict:
    import importlib
    import platform

    out = {"python": platform.python_version()}
    for name in (
        "numpy", "scipy", "sklearn", "mapie", "optuna", "torch", "tabpfn", "tabicl",
        "tabm", "catboost", "lightgbm", "xgboost", "rtdl_num_embeddings", "skrub",
    ):
        try:
            mod = importlib.import_module(name)
            v = getattr(mod, "__version__", None)
        except Exception:  # noqa: BLE001
            out[name] = None
            continue
        if v is None:  # tabicl 2.x exposes no __version__
            try:
                import importlib.metadata as md

                v = md.version(name)
            except Exception:  # noqa: BLE001
                v = "?"
        out[name] = str(v)
    return out
