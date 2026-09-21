"""tabsets.metrics: parity with MAPIE's metrics, the empty-set identities, the
three accountings, and the ECE regression from apostila checa_mapie.py."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testlib import require, run_module, smoke_cells  # noqa: E402

from tabsets import metrics as M  # noqa: E402
from tabsets import sets as S  # noqa: E402


def _all_sets():
    for c in smoke_cells():
        for sc in ("lac", "aps", "raps", "top_k", "aps_rand"):
            yield c, S.sets_for_cell(c, sc)


def test_coverage_width_sscs_match_mapie():
    require("mapie")
    from mapie.metrics.classification import (classification_coverage_score, classification_mean_width_score,
                                              classification_ssc_score)

    for c, r in _all_sets():
        ys = r.as_mapie()
        assert M.coverage(c.y_test, r.sets) == float(classification_coverage_score(c.y_test, ys)[0])
        assert M.mean_width(r.sets) == float(classification_mean_width_score(ys)[0])
        assert M.sscs(c.y_test, r.sets) == float(classification_ssc_score(c.y_test, ys)[0]), (c.name, r.score)


def test_empty_set_identities():
    """any empty set => SSCS = 0; eps_pt <= 1 - coverage exactly; SSCS+ >= SSCS."""
    seen_empty = False
    for c, r in _all_sets():
        eps = M.empty_rate(r.sets, c.y_test)  # asserts eps <= 1 - coverage
        ss = M.sscs(c.y_test, r.sets)
        sp = M.sscs_plus(c.y_test, r.sets)
        if eps > 0:
            seen_empty = True
            assert ss == 0.0, (c.name, r.score, ss)
            assert M.eps_run(r.sets) == 1
        assert sp >= ss or np.isnan(sp)
        assert M.min_stratum_count(r.sets) >= 1
    assert seen_empty, "no empty set in the smoke cells - the identity test is vacuous"


def test_ssc_strata_conventions():
    c = [c for c in smoke_cells() if not c.is_binary][0]
    r = S.sets_for_cell(c, "lac")
    st = M.ssc_strata(c.y_test, r.sets)
    assert len(st) == c.n_classes + 1 and st["n"].sum() == c.n_test
    pooled = M.ssc_strata(c.y_test, r.sets, "pooled")
    assert list(pooled["size_min"]) == [0, 2] and pooled["n"].sum() == c.n_test
    # min_stratum_count sensitivity: strata below the floor are dropped
    st10 = M.ssc_strata(c.y_test, r.sets, min_stratum_count=10)
    assert st10.loc[st10["n"] < 10, "coverage"].isna().all()
    assert M.sscs(c.y_test, r.sets, min_stratum_count=10**9) != M.sscs(c.y_test, r.sets) or np.isnan(
        M.sscs(c.y_test, r.sets, min_stratum_count=10**9))


def test_three_accountings():
    for c, r in _all_sets():
        acc = M.three_accountings(c.y_test, r.sets, c.p_test).set_index("accounting")
        assert set(acc.index) == {"miss", "abstain", "top1"}
        if acc.loc["miss", "empty_rate"] > 0:
            assert acc.loc["abstain", "coverage"] > acc.loc["miss", "coverage"]
            assert acc.loc["top1", "width"] > acc.loc["miss", "width"]
            assert acc.loc["top1", "coverage"] >= acc.loc["miss", "coverage"]
            assert acc.loc["abstain", "sscs"] == acc.loc["miss", "sscs_plus"]
        else:
            assert acc.loc["abstain", "coverage"] == acc.loc["miss", "coverage"]
            assert acc.loc["top1", "width"] == acc.loc["miss", "width"]
        # fallback sets are never empty
        assert (M.fallback_top1(r.sets, c.p_test).sum(1) > 0).all()


def test_confident_wrong_and_selective_error_binary_budget():
    """For sizes in {0, 1}: 1 - coverage = eps_pt + confident-wrong exactly,
    and selective error = ws / (1 - eps_pt)."""
    for c in smoke_cells():
        r = S.sets_for_cell(c, "lac")
        if r.sets.sum(1).max() > 1:
            continue
        eps, ws = M.empty_rate(r.sets), M.confident_wrong_singleton_rate(c.y_test, r.sets)
        assert abs((1 - M.coverage(c.y_test, r.sets)) - (eps + ws)) < 1e-12
        assert abs(M.selective_error(c.y_test, r.sets) - ws / (1 - eps)) < 1e-12


def test_ece_regression_checa_mapie():
    """apostila checa_mapie.py: on perfectly calibrated synthetic data the
    benchmark's (n, 2) call returns ~0.279; the correct top-label / 1-D call
    returns ~0.01."""
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-2 * x))
    y = (rng.random(n) < p).astype(int)
    prob = np.c_[1 - p, p]
    require("mapie")   # ece_legacy_mapie imports it inside the package
    legacy = M.ece_legacy_mapie(y, prob)
    assert 0.27 < legacy < 0.29, legacy
    assert M.top_label_ece(y, prob, 15, "equal_mass") < 0.02
    assert M.top_label_ece(y, prob, 15, "uniform") < 0.02
    assert M.ece_binary(y, p, 15, "equal_mass") < 0.02
    # MAPIE's own correct 1-D call, for the record
    from mapie.metrics.calibration import expected_calibration_error

    assert abs(float(expected_calibration_error(y, p)) - 0.0149) < 0.003


def test_equal_mass_bins_are_equal_mass():
    ids = M._bin_ids(np.random.default_rng(1).random(1000), 15, "equal_mass")
    counts = np.bincount(ids, minlength=15)
    assert counts.max() - counts.min() <= 1 and counts.sum() == 1000


def test_proper_scores_match_benchmark_definitions():
    from sklearn.metrics import log_loss, roc_auc_score

    for c in smoke_cells():
        assert M.brier(c.y_test, c.p_test) == float(np.mean(np.sum(
            (c.p_test - (c.y_test[:, None] == np.arange(c.n_classes)[None, :])) ** 2, axis=1)))
        assert M.log_loss(c.y_test, c.p_test) == float(log_loss(c.y_test, c.p_test, labels=np.arange(c.n_classes)))
        ref = roc_auc_score(c.y_test, c.p_test[:, 1]) if c.is_binary else roc_auc_score(
            c.y_test, c.p_test, multi_class="ovo", average="weighted")
        assert M.auc(c.y_test, c.p_test) == float(ref)
        dec = M.brier_decomposition(c.y_test, c.p_test)
        assert abs(dec["brier"] - (dec["reliability"] - dec["resolution"] + dec["uncertainty"] + dec["residual"])) < 1e-12


def test_aurc_bounds_and_oracle():
    # The bound holds when the confidence ranks the errors, which is what a calibrated
    # score does. It is not an identity: under a ranking no better than chance, refusing
    # the least confident points does not lower the risk and the area can exceed it.
    for c in smoke_cells(label_noise=None):
        a = M.aurc(c.y_test, c.p_test)
        assert 0 <= a["aurc"] <= a["top1_error"] + 1e-12
        assert a["e_aurc"] >= -1e-12
    # perfect ranking -> AURC equals the oracle -> E-AURC = 0
    y = np.array([0] * 90 + [1] * 10)
    p = np.c_[np.linspace(0.99, 0.51, 100), 1 - np.linspace(0.99, 0.51, 100)]
    a = M.aurc(y, p)
    assert abs(a["e_aurc"]) < 1e-12


def test_mondrian_by_class_guarantee_direction():
    """Class-conditional LAC never uses a smaller per-class threshold than
    the level asks; classes are covered at >= 1-alpha on the calibration
    fold by construction (checked on calibration data, where it is exact)."""
    for c in smoke_cells():
        r = M.mondrian_lac_by_class(c.p_cal, c.y_cal, c.p_cal, c.alpha)
        cw = M.classwise_metrics(c.y_cal, r.sets)
        for k, cov, n in zip(cw["class_id"], cw["coverage"], cw["n"]):
            if k in r.extra["classes_underfilled"]:
                assert cov == 1.0
            else:
                assert cov >= 1 - c.alpha - 1e-12, (c.name, k, cov)


def test_sharpness_diagnostic_equals_lac_empty_rate():
    for c in smoke_cells():
        d = M.sharpness_diagnostic(c.p_cal, c.y_cal, c.p_test, c.alpha)
        assert d["mass_below_threshold"] == d["empty_rate"]


if __name__ == "__main__":
    run_module(globals())
