"""Constant-classifier guard (A1 audit, 2026-08-30) and the Table-4 gate.

Run with ``pytest tests/`` or plainly ``python tests/test_constant_guard.py``.
No model is trained: the guard is exercised on synthetic probability
matrices and the scanner on a temporary cache written by ``cache_io``.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hplr_testlib import run_module  # noqa: E402

from cache_io import cache_path, write_cache  # noqa: E402
from hplr import constant_cells as CC  # noqa: E402
from model.utils import (  # noqa: E402
    CONSTANT_ATOL,
    LOGLOSS_TIEBREAK,
    _probabilities_from_logits,
    is_constant_prediction,
)


def test_is_constant_prediction_rows():
    p = np.tile([0.9, 0.1], (50, 1))
    assert is_constant_prediction(p)
    assert is_constant_prediction(p, argmax=False)
    # identical within atol still constant; beyond atol not (unless argmax constant)
    q = p.copy()
    q[3, 0] += CONSTANT_ATOL / 2
    assert is_constant_prediction(q, argmax=False)
    q[3, 0] += 1e-6
    assert not is_constant_prediction(q, argmax=False)
    assert is_constant_prediction(q)  # argmax still class 0 everywhere


def test_is_constant_prediction_informative():
    rng = np.random.default_rng(0)
    p1 = rng.dirichlet(np.ones(3), size=100)
    assert not is_constant_prediction(p1)
    assert not is_constant_prediction(p1, argmax=False)
    # a single differing row breaks constancy
    p = np.tile([0.9, 0.1], (50, 1))
    p[7] = [0.2, 0.8]
    assert not is_constant_prediction(p)
    # empty / 1-D inputs
    assert not is_constant_prediction(np.zeros((0, 2)))
    assert is_constant_prediction(np.full(10, 0.3), argmax=False)


def test_probabilities_from_logits():
    logits = np.array([[2.0, -1.0], [0.0, 0.0]])
    p = _probabilities_from_logits(logits)
    assert np.allclose(p.sum(axis=1), 1) and p[1, 0] == 0.5
    q = np.array([[0.3, 0.7], [0.6, 0.4]])
    assert np.array_equal(_probabilities_from_logits(q), q)


def test_tiebreak_is_lexicographic():
    # accuracy steps on any TALENT validation split (n_val < 2 000) exceed the
    # largest possible log-loss penalty (sklearn clips at 1e-15 -> 34.54)
    max_penalty = LOGLOSS_TIEBREAK * -np.log(1e-15)
    assert max_penalty < 1 / 2000 / 2


def _write_cell(cache_dir, dataset, model, seed, p_test, y_test, meta_extra=None):
    K = p_test.shape[1]
    rng = np.random.default_rng(seed)
    p_cal = rng.dirichlet(np.ones(K), size=40)
    y_cal = rng.integers(0, K, size=40)
    meta = dict(dataset=dataset, task_type="binclass" if K == 2 else "multiclass",
                model=model, seed=seed, machine="test", mock_run=False, n_train=100,
                confidence_level=0.9, mapie_random_state=seed, conformity_scores=["lac"])
    meta.update(meta_extra or {})
    path = cache_path(cache_dir, dataset, meta["task_type"], model, seed, "test")
    write_cache(path, p_cal=p_cal, y_cal=y_cal, p_test=p_test, y_test=y_test,
                idx_cal=np.arange(40), idx_test=np.arange(len(y_test)), classes=np.arange(K),
                X_cal=p_cal, X_test=p_test, meta=meta, qhat={})
    return path


def test_scanner_flags_constant_and_auc_half():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=60)
    with tempfile.TemporaryDirectory() as d:
        _write_cell(d, "dsA", "xgboost", 1, np.tile([0.88, 0.12], (60, 1)), y,
                    {"best_params": {"model_min_child_weight": 19085.0}, "hpo_constant_trials": 1})
        p_good = np.zeros((60, 2)); p_good[:, 1] = np.clip(y * 0.6 + rng.uniform(0, 0.4, 60), 0, 1)
        p_good[:, 0] = 1 - p_good[:, 1]
        _write_cell(d, "dsA", "xgboost", 2, p_good, y,
                    {"best_params": {"model_min_child_weight": 0.24}, "hpo_constant_trials": 1})
        _write_cell(d, "dsA", "lightgbm", 1, p_good, y)
        # mock cells are ignored by default
        _write_cell(d, "dsM", "xgboost", 9, np.tile([0.5, 0.5], (60, 1)), y, {"mock_run": True})

        df = CC.scan([d])
        assert len(df) == 3
        flagged = df[df["flagged"]]
        assert list(flagged["seed"]) == [1] and list(flagged["model"]) == ["xgboost"]
        assert bool(flagged["p_test_constant"].iloc[0]) and float(flagged["auc"].iloc[0]) == 0.5
        assert bool(flagged["mcw_over_n_train"].iloc[0])
        counts = CC.per_model_counts(df).set_index("model")
        assert counts.loc["xgboost", "flagged"] == 1 and counts.loc["lightgbm", "flagged"] == 0
        assert CC.main([d]) == 1
        assert CC.main([d, "--all"]) == 1

        df_mock = CC.scan([d], mock=True)
        assert len(df_mock) == 4


def test_scanner_clean_dir_exits_zero():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 3, size=90)
    p = rng.dirichlet(np.ones(3), size=90)
    p[np.arange(90), y] += 1.0
    p /= p.sum(axis=1, keepdims=True)
    with tempfile.TemporaryDirectory() as d:
        _write_cell(d, "dsB", "catboost", 3, p, y)
        assert CC.main([d]) == 0


if __name__ == "__main__":
    run_module(globals())
