"""tabsets.labelfree: the threshold solved without labels, and what it does not promise."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testlib import make_cell, run_module  # noqa: E402

from tabsets import labelfree as LF  # noqa: E402
from tabsets import sets as S  # noqa: E402


def _G(p, t):
    """The predicted error mass strictly below ``t``, per point."""
    p = np.asarray(p, dtype=float)
    return float((p * (p < t)).sum() / p.shape[0])


def _probs(n=400, K=4, conc=1.0, seed=0):
    return np.random.default_rng(seed).dirichlet(np.full(K, conc), size=n)


def test_threshold_solves_its_defining_equation():
    """``tau`` is a pooled probability value at which the predicted error reaches alpha."""
    for seed in range(8):
        p = _probs(seed=seed)
        for alpha in (0.01, 0.05, 0.10, 0.20):
            tau = LF.labelfree_threshold(p, alpha)
            assert tau in set(p.ravel().tolist())
            assert _G(p, tau) >= alpha - 1e-12
            below = p.ravel()[p.ravel() < tau]
            if below.size:
                assert _G(p, below.max()) < alpha + 1e-12


def test_threshold_is_monotone_in_the_level():
    p = _probs(seed=1)
    taus = [LF.labelfree_threshold(p, a) for a in (0.01, 0.05, 0.10, 0.20, 0.30)]
    assert all(b >= a for a, b in zip(taus, taus[1:]))


def test_variants_differ_by_at_most_one_tie_block():
    for seed in range(10):
        p = _probs(seed=seed)
        v = np.unique(p.ravel())
        for alpha in (0.05, 0.10, 0.20):
            a = LF.labelfree_threshold(p, alpha, variant="block")
            b = LF.labelfree_threshold(p, alpha, variant="inclusive")
            assert abs(int(np.searchsorted(v, a)) - int(np.searchsorted(v, b))) <= 1


def test_sets_are_the_threshold_applied_and_carry_no_guarantee():
    p = _probs(seed=2)
    r = LF.labelfree_sets(p, alpha=0.10)
    assert np.array_equal(r.sets, p >= r.qhat)
    assert r.guarantee is None
    assert r.extra["solved_on"] == "test"
    assert r.score == "labelfree"


def test_split_conformal_still_declares_its_guarantee():
    """The contrast is the point of the field: one of these rests on labels, the other does not."""
    c = make_cell(K=4, label_noise=0.02, flat_share=0.05)
    conformal = S.sets_for_cell(c, "lac", alpha=0.10)
    free = LF.labelfree_sets(c.p_test, alpha=0.10)
    assert conformal.guarantee is not None and "finite-sample" in conformal.guarantee
    assert free.guarantee is None


def test_solving_on_calibration_records_where_it_was_solved():
    c = make_cell(K=4)
    r = LF.labelfree_sets(c.p_test, alpha=0.10, p_solve=c.p_cal)
    assert r.extra["solved_on"] == "calibration" and r.n_cal == c.n_cal
    assert r.qhat == LF.labelfree_threshold(c.p_cal, 0.10)


def test_constant_threshold_is_one_minus_the_level():
    p = _probs(seed=3)
    r = LF.constant_threshold_sets(p, alpha=0.10)
    assert r.qhat == 0.9 and np.array_equal(r.sets, p >= 0.9)
    assert r.guarantee is None


def test_a_calibrated_scale_delivers_close_to_the_level():
    """Labels drawn from the probabilities make the scale canonically calibrated, which is
    the condition the threshold is derived under. Delivered coverage should then land near
    the requested level; this is a property of the scale, not a guarantee of the method."""
    errs = []
    for seed in range(12):
        c = make_cell(dataset=f"d{seed}", seed=seed, n_test=4000, K=4, concentration=1.0)
        r = LF.labelfree_sets(c.p_test, alpha=0.10)
        errs.append(abs(r.sets[np.arange(len(c.y_test)), c.y_test].mean() - 0.90))
    assert float(np.mean(errs)) < 0.02


def test_threshold_rejects_bad_input():
    for bad in (dict(alpha=0.0), dict(alpha=1.0), dict(variant="nope")):
        try:
            LF.labelfree_threshold(_probs(), **{"alpha": 0.1, **bad})
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass


if __name__ == "__main__":
    run_module(globals())
