"""Parity of tabsets.sets with MAPIE 1.0.1 SplitConformalClassifier(prefit=True).

Run: ``~/miniforge3/envs/uq/bin/python tests/test_sets.py`` (or pytest).
Needs the smoke caches under journal_results/smoke_*/cache/ and mapie 1.0.1.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testlib import require, run_module, smoke_cells  # noqa: E402

from tabsets import sets as S  # noqa: E402

SCORES = ("lac", "aps", "raps", "top_k")


def _mapie_sets(cell, score, include_last_label=True, alpha=None, seed=None):
    require("mapie")
    from mapie.classification import SplitConformalClassifier

    from tabsets.cache import CachedProbaClassifier

    alpha = cell.alpha if alpha is None else alpha
    seed = cell.mapie_random_state if seed is None else seed
    est = CachedProbaClassifier(np.vstack([cell.p_cal, cell.p_test]), cell.classes)
    mc = SplitConformalClassifier(estimator=est, confidence_level=1 - alpha, prefit=True,
                                  conformity_score=score, random_state=seed)
    mc.conformalize(np.arange(cell.n_cal)[:, None], cell.y_cal)
    params = {"include_last_label": include_last_label} if score in ("aps", "raps") else None
    _, ys = mc.predict_set(np.arange(cell.n_cal, cell.n_cal + cell.n_test)[:, None], conformity_score_params=params)
    return ys[:, :, 0], float(mc._mapie_classifier.quantiles_[0]), mc


def test_mapie_version():
    require("mapie")
    import mapie

    assert mapie.__version__ == "1.0.1", mapie.__version__


def test_multiclass_parity_all_scores_default():
    """LAC / APS / RAPS / top-k sets and q-hat equal MAPIE's, byte for byte,
    at the cell's own level and seed (the benchmark's setting)."""
    cells = [c for c in smoke_cells() if not c.is_binary]
    assert cells, "no multiclass smoke cells"
    for c in cells:
        for sc in SCORES:
            ref, q_ref, _ = _mapie_sets(c, sc)
            r = S.sets_for_cell(c, sc)
            assert r.implementation == "mapie-parity", (c.name, sc)
            assert np.array_equal(ref, r.sets), f"{c.name} {sc}: {(ref != r.sets).sum()} cells differ"
            assert q_ref == r.qhat, (c.name, sc, q_ref, r.qhat)
            if sc in c.qhat:
                assert float(c.qhat[sc][0]) == r.qhat, (c.name, sc, "stored q-hat differs")


def test_multiclass_parity_include_last_label_modes():
    """APS and RAPS under include_last_label in {True, False, 'randomized'}."""
    for c in [c for c in smoke_cells() if not c.is_binary]:
        for sc in ("aps", "raps"):
            for ill in (True, False, "randomized"):
                ref, q_ref, _ = _mapie_sets(c, sc, include_last_label=ill)
                r = S.build_sets(sc, c.p_cal, c.y_cal, c.p_test, c.alpha, seed=c.mapie_random_state,
                                 include_last_label=ill)
                assert np.array_equal(ref, r.sets), (c.name, sc, ill, int((ref != r.sets).sum()))
                assert q_ref == r.qhat


def test_multiclass_parity_other_levels_and_seeds():
    for c in [c for c in smoke_cells() if not c.is_binary][:3]:
        for alpha in (0.2, 0.05):
            for seed in (0, 7):
                for sc in SCORES:
                    ref, q_ref, _ = _mapie_sets(c, sc, alpha=alpha, seed=seed)
                    r = S.build_sets(sc, c.p_cal, c.y_cal, c.p_test, alpha, seed=seed)
                    assert np.array_equal(ref, r.sets), (c.name, sc, alpha, seed)
                    assert q_ref == r.qhat


def test_binary_lac_parity_and_own_scores():
    """Binary: LAC equals MAPIE; APS/RAPS/top-k are refused by MAPIE and are
    ours (implementation == 'tabsets')."""
    cells = smoke_cells(K=2)
    for c in cells:
        ref, q_ref, _ = _mapie_sets(c, "lac")
        r = S.sets_for_cell(c, "lac")
        assert r.implementation == "mapie-parity"
        assert np.array_equal(ref, r.sets) and q_ref == r.qhat
        for sc in ("aps", "raps", "top_k"):
            try:
                _mapie_sets(c, sc)
                raise AssertionError("MAPIE 1.0.1 accepted a non-LAC score on a binary target")
            except ValueError as exc:
                assert "binary" in str(exc)
            r = S.sets_for_cell(c, sc)
            assert r.implementation == "tabsets"
            assert r.sets.shape == (c.n_test, 2)
            cov = r.sets[np.arange(c.n_test), c.y_test].mean()
            assert cov >= 1 - c.alpha - 0.05, (c.name, sc, cov)


def test_raps_size_raps_is_reset_to_sklearn_default():
    """MAPIE 1.0.1 overwrites size_raps with None -> sklearn's 0.1 share.
    Our default reproduces it; size_raps=0.2 is a different (documented)
    estimator."""
    c = [c for c in smoke_cells() if not c.is_binary][0]
    _, _, mc = _mapie_sets(c, "raps")
    cs = mc._mapie_classifier.conformity_score_function_
    assert cs.size_raps is None
    n_raps_mapie = len(cs.y_raps)
    r = S.sets_for_cell(c, "raps")
    assert r.extra["n_raps"] == n_raps_mapie
    assert abs(r.extra["size_raps"] - 0.1) < 0.01, r.extra["size_raps"]
    r2 = S.sets_for_cell(c, "raps", size_raps=0.2)
    assert r2.implementation == "tabsets" and abs(r2.extra["size_raps"] - 0.2) < 0.01


def test_quantile_methods_agree_except_documented_edge():
    """MAPIE's np.quantile('higher') at (n+1)(1-a)/n vs the ceil((n+1)(1-a))
    order statistic: equal for the benchmark's n and alpha = 0.1; off by one
    order statistic when (n+1)(1-alpha) is an integer < n (e.g. n = 19)."""
    rng = np.random.default_rng(0)
    for n in (240, 1544, 2625, 100, 1000):
        s = rng.random(n)
        assert S.conformal_quantile(s, 0.1, "mapie") == S.conformal_quantile(s, 0.1, "textbook"), n
    s = np.sort(rng.random(19))
    q_m = S.conformal_quantile(s, 0.1, "mapie")
    q_t = S.conformal_quantile(s, 0.1, "textbook")
    assert q_t == s[17] and q_m == s[18], (q_m, q_t)  # 18th vs 19th order statistic


def test_alpha_guard():
    c = smoke_cells()[0]
    try:
        S.lac_sets(c.p_cal, c.y_cal, c.p_test, alpha=1.0 / (c.n_cal + 5))
        raise AssertionError("expected ValueError for alpha < 1/n")
    except ValueError:
        pass


def test_lac_empty_iff_max_prob_below_threshold():
    for c in smoke_cells():
        r = S.sets_for_cell(c, "lac")
        empty = r.sets.sum(1) == 0
        below = c.p_test.max(1) < 1 - r.qhat - S.EPSILON
        assert np.array_equal(empty, below), c.name


def _knn_like(K, n, k, rng):
    """Probabilities that are multiples of 1/k (kNN votes), many exact ties."""
    votes = rng.multinomial(k, np.ones(K) / K, size=n)
    return votes / k


def test_tied_probabilities_parity():
    """kNN-like probabilities (multiples of 1/k, ties at the max in most
    rows): every score, every include_last_label mode, equals MAPIE."""
    require("mapie")
    from mapie.classification import SplitConformalClassifier

    from tabsets.cache import CachedProbaClassifier

    rng = np.random.default_rng(11)
    for K, k in ((4, 5), (7, 3), (3, 10)):
        n_cal, n_test = 400, 300
        p_cal, p_test = _knn_like(K, n_cal, k, rng), _knn_like(K, n_test, k, rng)
        y_cal = np.array([rng.choice(K, p=p) for p in p_cal])
        tied_rows = np.mean([len(np.unique(row)) < K for row in p_test])
        assert tied_rows > 0.3, tied_rows  # ties are real (K=3, k=10 has the fewest)
        classes = np.arange(K)
        est = CachedProbaClassifier(np.vstack([p_cal, p_test]), classes)
        for score in ("lac", "aps", "raps", "top_k"):
            mc = SplitConformalClassifier(estimator=est, confidence_level=0.9, prefit=True,
                                          conformity_score=score, random_state=5)
            mc.conformalize(np.arange(n_cal)[:, None], y_cal)
            modes = (True, False, "randomized") if score in ("aps", "raps") else (True,)
            for ill in modes:
                params = {"include_last_label": ill} if score in ("aps", "raps") else None
                _, ys = mc.predict_set(np.arange(n_cal, n_cal + n_test)[:, None], conformity_score_params=params)
                kw = {"include_last_label": ill} if score in ("aps", "raps") else {}
                r = S.build_sets(score, p_cal, y_cal, p_test, 0.1, seed=5, **kw)
                assert np.array_equal(ys[:, :, 0], r.sets), (K, k, score, ill, int((ys[:, :, 0] != r.sets).sum()))
                assert float(mc._mapie_classifier.quantiles_[0]) == r.qhat


def test_tied_probabilities_parity_on_ruche_knn_cells():
    """The two Ruche kNN cells the Tier-1 driver flagged: tabsets equals MAPIE
    recomputed from the cached probabilities on this machine. The Ruche CSV
    row (and the stored q-hat) disagree with both because numpy's argsort
    orders exact ties differently on x86/AVX-512 than on arm64 - see the
    tabsets.sets module docstring; this is a platform property, not a rule
    tabsets can reproduce."""
    import glob

    from tabsets.cache import load_npz

    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          "journal_results", "ruche", "cache", "*__multiclass__knn__*.npz")))
    files = [f for f in files if os.path.basename(f).startswith(("allrep__", "steel_plates_faults__"))]
    if not files:
        print("ruche kNN cells not on this machine; skipped")
        return
    for f in files:
        c = load_npz(f)
        for sc in ("lac", "aps", "raps", "top_k"):
            ref, q_ref, _ = _mapie_sets(c, sc)
            r = S.sets_for_cell(c, sc)
            assert np.array_equal(ref, r.sets) and q_ref == r.qhat, (c.name, sc)


def test_lac_jitter():
    """Jitter changes nothing without ties, breaks ties with them, keeps the
    level, and is reproducible from its seed."""
    rng = np.random.default_rng(2)
    K, n = 4, 500
    p_cal, p_test = rng.dirichlet(np.ones(K), n), rng.dirichlet(np.ones(K), n)
    y_cal = np.array([rng.choice(K, p=p) for p in p_cal])
    a = S.lac_sets(p_cal, y_cal, p_test, 0.1)
    b = S.lac_sets(p_cal, y_cal, p_test, 0.1, jitter=True)
    assert np.array_equal(a.sets, b.sets) and b.implementation == "tabsets" and b.extra["jitter"] == 1e-6
    pk_cal, pk_test = _knn_like(K, n, 5, rng), _knn_like(K, n, 5, rng)
    yk = np.array([rng.choice(K, p=p) for p in pk_cal])
    a = S.lac_sets(pk_cal, yk, pk_test, 0.1)
    b = S.lac_sets(pk_cal, yk, pk_test, 0.1, jitter=1e-6, jitter_seed=3)
    b2 = S.lac_sets(pk_cal, yk, pk_test, 0.1, jitter=1e-6, jitter_seed=3)
    assert np.array_equal(b.sets, b2.sets)
    assert b.sets.sum() < a.sets.sum(), "jitter must remove the wholesale-admitted tied labels"
    s_cal = np.take_along_axis(1 - pk_cal, yk.reshape(-1, 1), axis=1)
    assert np.mean(np.abs(s_cal - a.qhat) <= S.EPSILON) > 0.05  # many calibration scores sit exactly at qhat
    cov_j = b.sets[np.arange(n), np.array([rng.choice(K, p=p) for p in pk_test])].mean()
    assert cov_j > 0.8


if __name__ == "__main__":
    run_module(globals())
