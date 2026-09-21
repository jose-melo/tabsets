"""Prediction sets built without labels.

The split-conformal threshold is the empirical quantile of a score computed on labelled
calibration data. This module solves for a threshold using the probabilities alone, on
the identity that a canonically calibrated score satisfies ``E[e_Y | p(X)] = p(X)``: the
mass a model assigns below a cut is what it predicts its own error rate to be below that
cut. Setting that predicted error to the requested level and solving gives

    tau = min { t : G(t) >= alpha },   G(t) = (1/n) sum_i sum_k p_ik 1{p_ik < t}

and the set of a point is ``{k : p_k >= tau}``. Nothing is calibrated on a label, so
nothing here has a coverage guarantee, and every result carries ``guarantee=None``.
Whether the level is nonetheless delivered is a property of the model's probability
scale, which is the measurement the article reports rather than a claim it makes.

The threshold can be solved on the calibration probabilities and applied to the test
ones, or solved on the test probabilities themselves, which uses no labels either and is
the transductive variant.
"""
from __future__ import annotations

import numpy as np

from .sets import SetResult


def labelfree_threshold(p, alpha: float = 0.10, variant: str = "block") -> float:
    """Solve ``tau`` on a probability matrix ``p`` of shape ``(n, K)``.

    ``block`` is the article's reading: ``G`` counts the mass strictly below ``t``, so the
    solution is the first pooled value whose whole tie block sits above a mass of
    ``alpha * n``. ``inclusive`` takes the first entry whose cumulative mass reaches
    ``alpha * n`` instead, one tie block lower. The two never differ by more than one tie
    block and agree on delivered coverage to about 1e-3; both are kept because the earlier
    probes of this work used the second and their numbers stay auditable.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 2:
        raise ValueError("p must be (n, K)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    n = p.shape[0]
    v = np.sort(p.ravel())
    cm = np.cumsum(v)
    if variant == "inclusive":
        j = min(int(np.searchsorted(cm / n, alpha, side="left")), len(v) - 1)
        return float(v[j])
    if variant != "block":
        raise ValueError(f"unknown variant {variant!r}")
    j0 = int(np.searchsorted(cm, alpha * n, side="left"))
    if j0 >= len(v):
        # the whole matrix carries less than alpha * n mass, which a simplex cannot do
        return float(v[-1])
    j = min(int(np.searchsorted(v, v[j0], side="right")), len(v) - 1)
    return float(v[j])


def labelfree_sets(p_test, alpha: float = 0.10, p_solve=None, variant: str = "block") -> SetResult:
    """Sets ``{k : p_k >= tau}`` for ``p_test``, with ``tau`` solved without labels.

    ``p_solve`` defaults to ``p_test``, the transductive variant. Passing the calibration
    probabilities instead solves the threshold before the test points are seen, which is
    the weaker but more deployable of the two.
    """
    p_test = np.asarray(p_test, dtype=float)
    src = p_test if p_solve is None else np.asarray(p_solve, dtype=float)
    tau = labelfree_threshold(src, alpha, variant=variant)
    return SetResult(
        score="labelfree",
        sets=p_test >= tau,
        qhat=tau,
        alpha=alpha,
        n_cal=0 if p_solve is None else len(src),
        implementation="tabsets",
        extra=dict(solved_on="test" if p_solve is None else "calibration", variant=variant),
        guarantee=None,
    )


def constant_threshold_sets(p_test, alpha: float = 0.10) -> SetResult:
    """Sets ``{k : p_k >= 1 - alpha}``, the threshold a perfectly calibrated scale implies.

    The reference the solved threshold is measured against: it uses neither labels nor
    the probabilities of the evaluation set, so any coverage it delivers comes entirely
    from the scale being what it claims to be.
    """
    p_test = np.asarray(p_test, dtype=float)
    return SetResult(
        score="constant", sets=p_test >= 1 - alpha, qhat=1 - alpha, alpha=alpha, n_cal=0,
        implementation="tabsets", extra=dict(solved_on="none"), guarantee=None,
    )


def threshold_density(p, tau: float, h: float = 0.01) -> float:
    """Share of entries within ``h`` of ``tau``, divided by ``2h``.

    How sharply the delivered level reacts to an error in the threshold. A high density
    at the cut means a small misplacement moves many points in or out of their set.
    """
    v = np.asarray(p, dtype=float).ravel()
    return float(np.mean((v >= tau - h) & (v <= tau + h)) / (2 * h))
