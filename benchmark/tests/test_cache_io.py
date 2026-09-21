"""Cache round-trip and MAPIE parity tests.

Run with ``pytest tests/`` or plainly ``python tests/test_cache_io.py``.
No model is trained: probabilities are random Dirichlet draws with a fixed
seed, and the "live" estimator is a stub whose ``predict_proba(X)`` returns
the rows of X itself, so the live MAPIE path and the cached MAPIE path can be
compared set for set.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cache_io import (  # noqa: E402
    CachedProbaClassifier,
    append_manifest,
    cache_exists,
    cache_name,
    cache_path,
    conformal_from_arrays,
    conformal_from_cache,
    parse_cache_name,
    read_cache,
    sha1_array,
    write_cache,
)

SCORES = ("lac", "aps", "raps", "top_k")


class IdentityProba(BaseEstimator, ClassifierMixin):
    """The 'fitted model': its features ARE its probabilities."""

    def __init__(self, classes):
        self.classes_ = np.asarray(classes)

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return np.asarray(X)

    def predict(self, X):
        return self.classes_[np.asarray(X).argmax(1)]

    def __sklearn_is_fitted__(self):
        return True


def _synthetic(seed=0, n_cal=300, n_test=200, k=4):
    rng = np.random.default_rng(seed)
    p_cal = rng.dirichlet(np.full(k, 0.5), size=n_cal)
    p_test = rng.dirichlet(np.full(k, 0.5), size=n_test)
    y_cal = np.array([rng.choice(k, p=p) for p in p_cal])
    y_test = np.array([rng.choice(k, p=p) for p in p_test])
    return p_cal, y_cal, p_test, y_test, np.arange(k)


def _live_sets(p_cal, y_cal, p_test, classes, score, cl=0.9, seed=123):
    from mapie.classification import SplitConformalClassifier

    mc = SplitConformalClassifier(
        estimator=IdentityProba(classes),
        confidence_level=cl,
        prefit=True,
        conformity_score=score,
        random_state=seed,
    )
    mc.conformalize(p_cal, y_cal)
    _, y_set = mc.predict_set(p_test)
    return y_set


def test_cached_classifier_matches_live_path_for_all_scores():
    p_cal, y_cal, p_test, y_test, classes = _synthetic(seed=7)
    for score in SCORES:
        live = _live_sets(p_cal, y_cal, p_test, classes, score)
        cached, qhat = conformal_from_arrays(p_cal, y_cal, p_test, classes, score, 0.9, 123)
        assert live.shape == cached.shape, score
        assert np.array_equal(live, cached), f"{score}: sets differ in {np.sum(live != cached)} entries"
        assert qhat is not None and np.all(np.isfinite(qhat)), score
    # the lookup table is index-addressed; a shuffled query must return the same rows
    est = CachedProbaClassifier(p_test, classes)
    idx = np.random.default_rng(1).permutation(len(p_test))
    assert np.array_equal(est.predict_proba(idx[:, None]), p_test[idx])
    assert np.array_equal(est.predict(idx[:, None]), classes[p_test[idx].argmax(1)])


def test_npz_round_trip_and_parity_from_disk():
    p_cal, y_cal, p_test, y_test, classes = _synthetic(seed=11)
    X_cal = np.random.default_rng(2).normal(size=(len(p_cal), 5))
    X_test = np.random.default_rng(3).normal(size=(len(p_test), 5))
    qhat = {s: conformal_from_arrays(p_cal, y_cal, p_test, classes, s, 0.9, 5)[1] for s in SCORES}
    meta = dict(
        dataset="toy(1)", task_type="multiclass", n_classes=4, model="stub", seed=5,
        mapie_random_state=5, confidence_level=0.9, conformity_scores=list(SCORES),
        machine="test", best_params={"a": np.float64(1.5)}, hpo_ran=True,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = cache_path(tmp, "toy(1)", "multiclass", "stub", 5, "test")
        assert os.path.basename(path) == cache_name("toy(1)", "multiclass", "stub", 5, "test")
        assert cache_exists(tmp, "toy(1)", "multiclass", "stub", 5) is None
        write_cache(
            path, p_cal=p_cal, y_cal=y_cal, p_test=p_test, y_test=y_test,
            idx_cal=np.arange(len(p_cal)), idx_test=np.arange(len(p_test)) + 1000,
            classes=classes, X_cal=X_cal, X_test=X_test, meta=meta, qhat=qhat,
        )
        assert not os.path.exists(path + ".tmp.npz")
        assert cache_exists(tmp, "toy(1)", "multiclass", "stub", 5) == path
        assert parse_cache_name(path) == {
            "dataset": "toy(1)", "task_type": "multiclass", "model": "stub", "tag": "", "seed": 5, "machine": "test",
        }

        d = read_cache(path)
        assert np.array_equal(d["p_cal"], p_cal) and np.array_equal(d["p_test"], p_test)
        assert d["p_cal"].dtype == np.float64
        assert np.array_equal(d["y_cal"], y_cal) and np.array_equal(d["y_test"], y_test)
        assert np.array_equal(d["classes"], classes)
        assert d["idx_test"][0] == 1000
        assert d["x_cal_sha1"] == sha1_array(X_cal) and d["x_test_sha1"] == sha1_array(X_test)
        assert d["meta"]["dataset"] == "toy(1)" and d["meta"]["best_params"] == {"a": 1.5}
        assert d["meta"]["schema_version"] == 1 and sorted(d["qhat"]) == sorted(SCORES)

        for score in SCORES:
            live = _live_sets(p_cal, y_cal, p_test, classes, score, seed=5)
            from_disk, q = conformal_from_cache(d, score)
            assert np.array_equal(live, from_disk), score
            assert np.allclose(q, d["qhat"][score]), score

        manifest = os.path.join(tmp, "manifest.csv")
        append_manifest(manifest, {**meta, "cache_file": os.path.basename(path)})
        append_manifest(manifest, {**meta, "cache_file": "second.npz"})
        lines = open(manifest).read().splitlines()
        assert len(lines) == 3 and lines[0].startswith("cache_file,dataset,task_type")


def test_cache_name_round_trips_with_and_without_tag():
    name = cache_name("credit-g", "binclass", "PFN-v2", 42, "mac")
    assert name == "credit-g__binclass__PFN-v2__seed42__mac.npz"
    assert parse_cache_name(name) == {
        "dataset": "credit-g", "task_type": "binclass", "model": "PFN-v2",
        "tag": "", "seed": 42, "machine": "mac",
    }
    tagged = cache_name("credit-g", "binclass", "PFN-v2", 42, "mac", tag="ne8")
    assert tagged == "credit-g__binclass__PFN-v2@ne8__seed42__mac.npz"
    assert parse_cache_name(tagged)["model"] == "PFN-v2"
    assert parse_cache_name(tagged)["tag"] == "ne8"
    # dataset names with "__"-free but glob-hostile characters survive
    ugly = "Contaminant-(Urbinati)[x]"
    assert parse_cache_name(cache_name(ugly, "binclass", "tabicl", 1, "jz", tag="t1.0"))["dataset"] == ugly
    with tempfile.TemporaryDirectory() as d:
        for t in (None, "ne8"):
            open(cache_path(d, ugly, "binclass", "tabicl", 1, "jz", tag=t), "wb").close()
        assert cache_exists(d, ugly, "binclass", "tabicl", 1).endswith("__tabicl__seed1__jz.npz")
        assert cache_exists(d, ugly, "binclass", "tabicl", 1, tag="ne8").endswith("__tabicl@ne8__seed1__jz.npz")
        assert cache_exists(d, ugly, "binclass", "tabicl", 1, tag="ne2") is None


def test_sha1_array_handles_object_arrays_and_dtypes():
    a = np.array([[1, 2.5], [3, 4]])
    assert sha1_array(a) == sha1_array(a.astype(np.float32).astype(np.float64))
    assert sha1_array(a) == sha1_array(np.asfortranarray(a))
    obj = np.array([["red", 1.0], ["blue", 2.0]], dtype=object)
    assert sha1_array(obj) == sha1_array(obj.copy())
    assert sha1_array(obj) != sha1_array(np.array([["red", 1.0], ["blue", 3.0]], dtype=object))


def test_write_cache_rejects_bad_probabilities():
    p_cal, y_cal, p_test, y_test, classes = _synthetic(seed=1, n_cal=20, n_test=10)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            write_cache(
                os.path.join(tmp, "x.npz"), p_cal=p_cal * 2, y_cal=y_cal, p_test=p_test,
                y_test=y_test, idx_cal=np.arange(20), idx_test=np.arange(10), classes=classes,
                X_cal=np.zeros((20, 1)), X_test=np.zeros((10, 1)), meta={},
            )
        except AssertionError:
            pass
        else:
            raise AssertionError("rows not summing to 1 were accepted")
        assert not os.listdir(tmp), "a rejected write must leave nothing behind"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
