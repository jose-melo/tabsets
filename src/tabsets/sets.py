"""Conformal prediction sets from cached probabilities, in plain numpy.

Every constructor takes ``(p_cal, y_cal, p_test, alpha)`` and returns a
:class:`SetResult` with the boolean set matrix ``sets`` of shape
``(n_test, K)`` and the calibrated quantile ``qhat``.

Parity with MAPIE 1.0.1
-----------------------
The code paths below are a line-by-line transcription of
``mapie.conformity_scores.sets.{lac,aps,raps,topk}`` for the
``SplitConformalClassifier(prefit=True)`` case that produced the benchmark
CSVs (``utils.py`` / ``cache_io.conformal_from_arrays``). Three MAPIE
choices are reproduced on purpose, because parity is the point:

* **the quantile** is ``np.quantile(s, (n+1)(1-alpha)/n, method="higher")``,
  which is *not always* the ``ceil((n+1)(1-alpha))``-th order statistic of
  the textbook: ``np.quantile`` interpolates on ``(n-1)*q`` and ``"higher"``
  rounds that up, so when ``(n+1)(1-alpha)`` is an integer ``k < n`` MAPIE
  returns the ``(k+1)``-th order statistic instead of the ``k``-th (one
  order statistic more conservative). :func:`conformal_quantile` exposes both
  (``method="mapie"`` default, ``method="textbook"``); every set constructor
  takes ``quantile_method``.
* **argsort and ties.** Every ranking is ``np.argsort(p, axis=1)`` with
  numpy's default ``kind="quicksort"`` followed by ``fliplr`` / ``flip``:
  MAPIE has no tie rule of its own - tied probabilities (kNN: multiples of
  1/k) are ordered however numpy's introsort orders them, and tabsets calls
  the same functions in the same order, so on one machine the two agree
  byte for byte (``tests/test_sets.py::test_tied_probabilities_parity``).
  **That order is platform-dependent**: numpy >= 1.25 on x86 with AVX-512
  uses a vectorised, non-stable ``argsort`` (x86-simd-sort) whose tie
  permutation differs from the scalar introsort of an arm64 Mac (numpy
  1.26.4 on both). Measured on the Ruche grid (2026-08-30): of 16 203 rows
  every one reproduces on the Mac except two kNN cells with exact ties
  (``allrep`` seed 1727351912 APS, ``steel_plates_faults`` seed 1502757397
  top-k - width 5.12 vs 3.27); their *stored* q-hat (written on Ruche)
  differs from the one MAPIE itself recomputes from the stored
  probabilities on the Mac, and for steel_plates the Ruche value is
  recovered exactly by reversing the tie order. Consequence for the paper:
  rows of tie-producing models (kNN) are reproducible only up to the tie
  permutation of the CPU that ran them; every tie-free model is exact. The
  jittered-LAC control (``lac_sets(jitter=...)``, E-Mond) removes the
  dependence. Within one platform the APS rule on a tie is: the *first*
  position (``argmin`` over the masked cumsum) whose cumulated mass reaches
  the threshold is the last included label, and the set is ``p >= p_last -
  1e-8``, so every label tied with it enters as well (top-k likewise: "in
  case two probabilities are equal, both are taken").
* **APS calibration scores are always randomised** (``u * p(y)`` is
  subtracted, ``u ~ U(0,1)`` from ``RandomState(seed)``) even when the test
  side is deterministic (``include_last_label=True``, MAPIE's default and the
  benchmark's setting). ``randomize_calibration=False`` gives the textbook
  deterministic APS of Romano et al. (2020, eq. 5 without U).
* **RAPS's held-out share is 10 %, not the documented 20 %.** MAPIE's
  ``RAPSConformityScore(size_raps=0.2)`` is overwritten to ``None`` by
  ``_MapieClassifier._check_fit_parameter`` (``set_external_attributes``
  is called without ``size_raps``), and ``StratifiedShuffleSplit
  (test_size=None)`` then falls back to scikit-learn's default ``0.1``.
  This happens whatever the user passes, so every RAPS row of the benchmark
  used a 10 % share. ``size_raps=None`` (default here) reproduces that;
  pass ``0.2`` for the documented behaviour (``implementation="tabsets"``).
* **RAPS's lambda search regularises with ``k_reg = lambda``** (MAPIE
  ``raps.py::_add_regularization`` in the non-prediction phase does
  ``k_star = cast(int, lambda_)``, a no-op cast). The regularised *scores*
  used for the quantile are correct; only the set widths that pick
  ``lambda_star`` are computed with the wrong ``k``. ``mapie_lambda_bug=True``
  (default) reproduces it; ``False`` uses ``k_star``. The paper states
  ``size_raps = 0.2`` and this choice.

Comparisons use MAPIE's ``EPSILON = 1e-8`` exactly where MAPIE does.

Binary targets: MAPIE 1.0.1 raises ``Invalid conformity score for binary
target`` for anything but LAC (``label_binarize`` returns one column for two
classes). The constructors here work for K = 2 by using an explicit one-hot;
:func:`build_sets` reports ``implementation="tabsets"`` in that case and
``"mapie-parity"`` otherwise.

References: Sadinle, Lei & Wasserman (2019) LAC; Romano, Sesia & Candès
(2020) APS; Angelopoulos, Bates, Jordan & Malik (2021) RAPS / top-k.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPSILON = np.float64(1e-8)  # mapie._machine_precision.EPSILON
SCORES = ("lac", "aps", "raps", "top_k")
RAPS_LAMBDAS = (0.001, 0.01, 0.1, 0.2, 0.5)  # Angelopoulos et al. 2021, as in MAPIE


SPLIT_CONFORMAL = "split conformal: marginal, finite-sample, exchangeable calibration"


@dataclass
class SetResult:
    score: str
    sets: np.ndarray          # (n_test, K) bool
    qhat: float               # calibrated quantile (int position for top_k)
    alpha: float
    n_cal: int
    implementation: str = "tabsets"   # "mapie-parity" when MAPIE would produce the same sets
    extra: dict = field(default_factory=dict)
    guarantee: str | None = SPLIT_CONFORMAL
    """What the level of this set rests on, or ``None`` when it rests on nothing.

    Every constructor in this module calibrates the threshold on held-out labels and so
    keeps the default. The label-free sets of :mod:`tabsets.labelfree` set it to ``None``:
    they are built without labels, carry no finite-sample statement, and the article
    reports how often they hold the requested level rather than asserting that they do.
    """

    @property
    def sizes(self) -> np.ndarray:
        return self.sets.sum(axis=1)

    def as_mapie(self) -> np.ndarray:
        """MAPIE's ``(n_test, K, n_alpha=1)`` layout."""
        return self.sets[:, :, None]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _check_inputs(p_cal, y_cal, p_test):
    p_cal = np.asarray(p_cal, dtype=np.float64)
    p_test = np.asarray(p_test, dtype=np.float64)
    y_cal = np.asarray(y_cal).astype(np.int64).ravel()
    if p_cal.ndim != 2 or p_test.ndim != 2 or p_cal.shape[1] != p_test.shape[1]:
        raise ValueError("p_cal and p_test must be (n, K) with the same K")
    if len(y_cal) != len(p_cal):
        raise ValueError("y_cal length must match p_cal")
    K = p_cal.shape[1]
    if y_cal.min() < 0 or y_cal.max() >= K:
        raise ValueError("y_cal must be positional labels in 0..K-1")
    # mapie.conformity_scores.sets.utils.check_proba_normalized (rtol 1e-5)
    np.testing.assert_allclose(p_cal.sum(1), 1, rtol=1e-5, err_msg="p_cal rows must sum to one")
    np.testing.assert_allclose(p_test.sum(1), 1, rtol=1e-5, err_msg="p_test rows must sum to one")
    return p_cal, y_cal, p_test


def check_alpha_n(alpha: float, n: int) -> None:
    """MAPIE's ``_check_alpha_and_n_samples``: n must be >= max(1/alpha, 1/(1-alpha))."""
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n < np.max([1 / alpha, 1 / (1 - alpha)]):
        raise ValueError(
            f"Number of calibration samples ({n}) is too low for alpha={alpha}: "
            "1/alpha and 1/(1-alpha) must be lower than n."
        )


def min_alpha(n: int) -> float:
    """Smallest alpha MAPIE accepts for ``n`` calibration points (1/n)."""
    return 1.0 / n


def conformal_quantile(scores, alpha: float, method: str = "mapie"):
    """Finite-sample (1-alpha) quantile of the calibration scores.

    ``method="mapie"``  : ``np.quantile(s, (n+1)(1-alpha)/n, method="higher")``
                          (``mapie.utils._compute_quantiles``; parity default).
    ``method="textbook"``: the ``ceil((n+1)(1-alpha))``-th smallest score
                          (Vovk; Angelopoulos & Bates 2021 eq. 2). Equal to
                          the MAPIE value except in the edge cases described
                          in the module docstring.
    Returns a numpy scalar with the dtype of ``scores`` (int for top-k).
    """
    s = np.asarray(scores).ravel()
    n = len(s)
    check_alpha_n(alpha, n)
    if method == "mapie":
        return np.quantile(s, ((n + 1) * (1 - alpha)) / n, method="higher")
    if method == "textbook":
        k = int(np.ceil((n + 1) * (1 - alpha)))
        k = min(max(k, 1), n)
        return np.sort(s)[k - 1]
    raise ValueError(f"unknown quantile method {method!r}")


def enforced_level(n_cal: int, alpha: float) -> float:
    """The coverage the split-conformal quantile actually enforces:
    ``ceil((n+1)(1-alpha)) / (n+1)`` (Vovk 2012; the marginal guarantee is
    ``>=`` this and ``<`` this ``+ 1/(n+1)``)."""
    return float(np.ceil((n_cal + 1) * (1 - alpha)) / (n_cal + 1))


def _one_hot(y, K):
    return (np.asarray(y)[:, None] == np.arange(K)[None, :]).astype(np.int64)


def _true_label_cumsum(y, p, K):
    """``APSConformityScore.get_true_label_cumsum_proba`` with an explicit
    one-hot (MAPIE's ``label_binarize`` has one column for K = 2)."""
    y_true = _one_hot(y, K)
    index_sorted = np.fliplr(np.argsort(p, axis=1))
    p_sorted = np.take_along_axis(p, index_sorted, axis=1)
    y_sorted = np.take_along_axis(y_true, index_sorted, axis=1)
    cumsum = np.cumsum(p_sorted, axis=1)
    cutoff = np.argmax(y_sorted, axis=1)
    tl = np.take_along_axis(cumsum, cutoff.reshape(-1, 1), axis=1)
    cutoff = cutoff + 1
    return tl, cutoff


def _true_label_position(p, y):
    """``mapie.conformity_scores.sets.utils.get_true_label_position`` -> (n, 1)."""
    index = np.argsort(np.fliplr(np.argsort(p, axis=1)))
    return np.take_along_axis(index, np.asarray(y).reshape(-1, 1), axis=1)


def _last_index_included(cumsum, threshold, include_last_label):
    """``get_last_index_included``; ``cumsum`` (n, K, n_alpha), ``threshold`` (n_alpha,)."""
    if include_last_label or include_last_label == "randomized":
        idx = np.ma.masked_less(cumsum - threshold[np.newaxis, :], -EPSILON).argmin(axis=1)
    else:
        max_threshold = np.maximum(threshold[np.newaxis, :], np.min(cumsum, axis=1))
        idx = np.argmax(np.ma.masked_greater(cumsum - max_threshold[:, np.newaxis, :], EPSILON), axis=1)
    return idx[:, np.newaxis, :]


def _last_included_proba(p3, thresholds, include_last_label, lambda_=None, k_reg=None):
    """``NaiveConformityScore._get_last_included_proba`` (+ RAPS regularisation
    when ``lambda_`` is given). ``p3`` is (n, K, n_alpha)."""
    index_sorted = np.flip(np.argsort(p3, axis=1), axis=1)
    p_sorted = np.take_along_axis(p3, index_sorted, axis=1)
    cumsum_sorted = np.cumsum(p_sorted, axis=1)
    if lambda_ is not None:
        cumsum_sorted = cumsum_sorted + lambda_ * np.maximum(
            0, np.cumsum(np.ones(cumsum_sorted.shape), axis=1) - k_reg
        )
    cumsum = np.take_along_axis(cumsum_sorted, np.argsort(index_sorted, axis=1), axis=1)
    idx_last = _last_index_included(cumsum, thresholds, include_last_label)
    p_last = np.take_along_axis(p3, idx_last, axis=1)
    zeros = p_last <= EPSILON
    if np.sum(zeros) > 0:
        p_last[zeros] = np.expand_dims(
            np.min(np.ma.masked_less(p3, EPSILON).filled(fill_value=np.inf), axis=1), axis=1
        )[zeros]
    return cumsum, idx_last, p_last


def _random_tie_breaking(sets, idx_last, cumsum, p_last, threshold, seed, lambda_=None, k_reg=None):
    """``APSConformityScore._add_random_tie_breaking`` (RAPS variant of V when
    ``lambda_`` is given)."""
    cumsum_last = np.squeeze(np.take_along_axis(cumsum, idx_last, axis=1), axis=1)
    if lambda_ is None:
        v = (cumsum_last - threshold.reshape(1, -1)) / p_last[:, 0, :]
    else:
        L = np.sum(sets, axis=1)
        v = (cumsum_last - threshold.reshape(1, -1)) / (
            p_last[:, 0, :] - lambda_ * np.maximum(0, L - k_reg) + lambda_ * (L > k_reg)
        )
    u = np.random.RandomState(seed).uniform(size=(sets.shape[0], 1))
    keep = np.less_equal(v - u, EPSILON)
    np.put_along_axis(sets, idx_last, keep[:, np.newaxis, :], axis=1)
    return sets


# --------------------------------------------------------------------------- #
# LAC
# --------------------------------------------------------------------------- #
def lac_sets(p_cal, y_cal, p_test, alpha=0.1, quantile_method="mapie", jitter=None, jitter_seed=0) -> SetResult:
    """LAC / "score" (Sadinle et al. 2019): s = 1 - p(y); label k in the set
    iff ``1 - p_k <= qhat + 1e-8``. The only score that can return an empty
    set: it does so exactly when ``max_k p_k < 1 - qhat``.

    ``jitter``: tie-breaking control for models whose probabilities are
    discrete (kNN: multiples of 1/k, so many calibration scores are *equal*
    to qhat and every one of them is admitted -> over-coverage, 0.9088 vs
    0.9033 in the reconstruction, SYNTHESIS s4 E-Mond). ``jitter=scale``
    adds iid ``U(0, scale)`` noise from ``RandomState(jitter_seed)`` to every
    calibration score and to every (test row, label) score, and the
    comparison becomes exact (``<=`` without the 1e-8 tolerance, which
    would swallow the noise). Any ``scale`` far below the smallest gap
    between distinct probabilities (1/k for kNN) leaves non-tied decisions
    unchanged; default when enabled: ``1e-6``. Noise on the scores is the
    textbook randomised conformal threshold (Vovk et al. 2005, "smoothed
    conformal") realised on the score scale. ``implementation="tabsets"``."""
    p_cal, y_cal, p_test = _check_inputs(p_cal, y_cal, p_test)
    s = np.take_along_axis(1 - p_cal, y_cal.reshape(-1, 1), axis=1)
    if jitter is None or jitter is False:
        q = conformal_quantile(s, alpha, quantile_method)
        sets = np.less_equal((1 - p_test) - q, EPSILON)
        return SetResult("lac", sets, float(q), alpha, len(y_cal), "mapie-parity",
                         extra=dict(threshold=float(1 - q), jitter=0.0))
    scale = 1e-6 if jitter is True else float(jitter)
    rng = np.random.RandomState(jitter_seed)
    s = s + rng.uniform(0, scale, size=s.shape)
    s_test = (1 - p_test) + rng.uniform(0, scale, size=p_test.shape)
    q = conformal_quantile(s, alpha, quantile_method)
    sets = np.less_equal(s_test, q)
    return SetResult("lac", sets, float(q), alpha, len(y_cal), "tabsets",
                     extra=dict(threshold=float(1 - q), jitter=scale, jitter_seed=jitter_seed))


# --------------------------------------------------------------------------- #
# APS
# --------------------------------------------------------------------------- #
def aps_sets(p_cal, y_cal, p_test, alpha=0.1, seed=0, include_last_label=True,
             randomize_calibration=True, quantile_method="mapie") -> SetResult:
    """APS (Romano et al. 2020). ``include_last_label`` follows MAPIE:
    ``True`` (default, the benchmark): include the label whose cumulated mass
    crosses qhat; ``False``: exclude it (never below one label);
    ``"randomized"``: drop it with probability V (Romano eq. 5) using
    ``RandomState(seed)`` - the only setting under which APS can be empty.
    ``randomize_calibration=True`` subtracts ``u * p(y)`` from the calibration
    scores exactly as MAPIE does under every ``include_last_label``."""
    p_cal, y_cal, p_test = _check_inputs(p_cal, y_cal, p_test)
    K = p_cal.shape[1]
    s, _ = _true_label_cumsum(y_cal, p_cal, K)
    if randomize_calibration:
        p_true = np.take_along_axis(p_cal, y_cal.reshape(-1, 1), axis=1)
        u = np.random.RandomState(seed).uniform(size=len(p_cal)).reshape(-1, 1)
        s = s - u * p_true
    q = conformal_quantile(s, alpha, quantile_method)
    thresholds = np.array([q], dtype=np.float64)
    p3 = np.repeat(p_test[:, :, np.newaxis], 1, axis=2)
    cumsum, idx_last, p_last = _last_included_proba(p3, thresholds, include_last_label)
    sets = np.greater_equal(p3 - p_last, -EPSILON)
    if include_last_label == "randomized":
        sets = _random_tie_breaking(sets, idx_last, cumsum, p_last, thresholds, seed)
    impl = "mapie-parity" if (randomize_calibration and K > 2) else "tabsets"
    return SetResult("aps", sets[:, :, 0], float(q), alpha, len(y_cal), impl,
                     extra=dict(include_last_label=include_last_label,
                                randomize_calibration=randomize_calibration))


# --------------------------------------------------------------------------- #
# RAPS
# --------------------------------------------------------------------------- #
def _raps_split(y_cal, size_raps, seed):
    """MAPIE ``RAPSConformityScore.split_data``: ``StratifiedShuffleSplit
    (n_splits=1, test_size=size_raps, random_state=seed)`` on the calibration
    labels; the held-out part chooses ``k_reg`` and ``lambda``. ``size_raps=None``
    is what MAPIE 1.0.1 actually passes (see module docstring) and means
    scikit-learn's default share of 0.1."""
    from sklearn.model_selection import StratifiedShuffleSplit

    _, counts = np.unique(y_cal, return_counts=True)
    if counts.min() < 2:
        # MAPIE (via StratifiedShuffleSplit) refuses; we follow it - no fallback.
        raise ValueError(
            "RAPS cannot be built: the least populated calibration class has "
            f"{int(counts.min())} member (StratifiedShuffleSplit needs >= 2); MAPIE skips it too"
        )
    sss = StratifiedShuffleSplit(n_splits=1, test_size=size_raps, random_state=seed)
    X_dummy = np.zeros((len(y_cal), 1))
    train_idx, val_idx = next(sss.split(X_dummy, y_cal))
    return train_idx, val_idx


def _regularize(k_star, lambda_, conf, cutoff):
    """``RAPSConformityScore._regularize_conformity_score``: (n,1)->(n,1,n_alpha)."""
    conf = np.repeat(conf[:, :, np.newaxis], len(k_star), axis=2)
    cutoff = np.repeat(cutoff[:, np.newaxis], len(k_star), axis=1)
    conf = conf + np.maximum(np.expand_dims(lambda_ * (cutoff - k_star), axis=1), 0)
    return conf


def _quantiles_3d(vec3, alpha_np, method):
    """``_compute_quantiles`` on a (n, 1, n_alpha) array -> (n_alpha,)."""
    return np.stack([conformal_quantile(vec3[:, :, i], a, method) for i, a in enumerate(alpha_np)])


def raps_sets(p_cal, y_cal, p_test, alpha=0.1, seed=0, include_last_label=True,
              size_raps=None, mapie_lambda_bug=True, quantile_method="mapie") -> SetResult:
    """RAPS (Angelopoulos et al. 2021) as MAPIE 1.0.1 runs it with
    ``prefit=True``: a stratified ``size_raps`` share of the calibration set
    (``None`` = MAPIE's effective 10 %, see module docstring; ``RandomState
    (seed)``) picks ``k_reg`` = 1 + the (1-alpha) quantile of
    the true-label rank and ``lambda`` from {.001,.01,.1,.2,.5} by smallest
    mean set size on that share; the rest calibrates the regularised APS
    score. See the module docstring for ``mapie_lambda_bug``."""
    p_cal, y_cal, p_test = _check_inputs(p_cal, y_cal, p_test)
    K = p_cal.shape[1]
    alpha_np = np.array([alpha], dtype=np.float64)
    train_idx, val_idx = _raps_split(y_cal, size_raps, seed)
    p_tr, y_tr = p_cal[train_idx], y_cal[train_idx]
    p_raps, y_raps = p_cal[val_idx], y_cal[val_idx]
    check_alpha_n(alpha, len(y_raps))

    # calibration scores (randomised, like APS) + cutoff on the train part
    s, cutoff = _true_label_cumsum(y_tr, p_tr, K)
    p_true = np.take_along_axis(p_tr, y_tr.reshape(-1, 1), axis=1)
    u = np.random.RandomState(seed).uniform(size=len(p_tr)).reshape(-1, 1)
    s = s - u * p_true

    # k_star and lambda_star on the RAPS share
    position = _true_label_position(p_raps, y_raps)
    k_star = np.array([conformal_quantile(position, alpha, quantile_method)]) + 1
    p_raps3 = np.repeat(p_raps[:, :, np.newaxis], 1, axis=2)
    lambda_star = np.zeros(1)
    best_sizes = np.full(1, np.finfo(np.float64).max)
    for lam in RAPS_LAMBDAS:
        tl, cut = _true_label_cumsum(y_raps, p_raps3[:, :, 0], K)
        reg = _regularize(k_star, lam, tl, cut)
        q_lam = _quantiles_3d(reg, alpha_np, quantile_method)
        k_reg_search = lam if mapie_lambda_bug else k_star
        _, _, p_last = _last_included_proba(p_raps3, q_lam, include_last_label,
                                            lambda_=lam, k_reg=k_reg_search)
        y_ps = np.greater_equal(p_raps3 - p_last, -EPSILON)
        sizes = y_ps.sum(axis=1).mean(axis=0)
        improve = sizes < best_sizes - EPSILON
        lambda_star = improve * lam + (1 - improve) * lambda_star
        best_sizes = improve * sizes + (1 - improve) * best_sizes
    lambda_star = float(lambda_star[0])

    reg_scores = _regularize(k_star, lambda_star, s, cutoff)
    q = _quantiles_3d(reg_scores, alpha_np, quantile_method)

    p3 = np.repeat(p_test[:, :, np.newaxis], 1, axis=2)
    cumsum, idx_last, p_last = _last_included_proba(p3, q, include_last_label,
                                                    lambda_=lambda_star, k_reg=k_star)
    sets = np.greater_equal(p3 - p_last, -EPSILON)
    if include_last_label == "randomized":
        sets = _random_tie_breaking(sets, idx_last, cumsum, p_last, q, seed,
                                    lambda_=lambda_star, k_reg=k_star)
    impl = "mapie-parity" if (K > 2 and mapie_lambda_bug and size_raps is None) else "tabsets"
    return SetResult("raps", sets[:, :, 0], float(q[0]), alpha, len(y_cal), impl,
                     extra=dict(k_reg=int(k_star[0]), lambda_=lambda_star,
                                size_raps=len(val_idx) / len(y_cal),
                                n_cal_scores=len(train_idx), n_raps=len(val_idx),
                                include_last_label=include_last_label))


# --------------------------------------------------------------------------- #
# top-k
# --------------------------------------------------------------------------- #
def topk_sets(p_cal, y_cal, p_test, alpha=0.1, quantile_method="mapie") -> SetResult:
    """Top-k (Angelopoulos et al. 2021, "naive" fixed-size baseline): score =
    rank of the true label (0 = top); qhat is the (1-alpha) quantile of the
    ranks; the set is the ``qhat + 1`` most probable labels (ties at the
    boundary all included, MAPIE semantics). Never empty."""
    p_cal, y_cal, p_test = _check_inputs(p_cal, y_cal, p_test)
    K = p_cal.shape[1]
    position = _true_label_position(p_cal, y_cal)
    q = int(conformal_quantile(position, alpha, quantile_method))
    index_sorted = np.fliplr(np.argsort(p_test, axis=1))
    idx_last = index_sorted[:, q]
    p_last = np.take_along_axis(p_test, idx_last.reshape(-1, 1), axis=1)
    sets = np.greater_equal(p_test - p_last, -EPSILON)
    return SetResult("top_k", sets, float(q), alpha, len(y_cal),
                     "mapie-parity" if K > 2 else "tabsets", extra=dict(k=q + 1))


# --------------------------------------------------------------------------- #
# dispatcher
# --------------------------------------------------------------------------- #
def build_sets(score: str, p_cal, y_cal, p_test, alpha=0.1, seed=0, **kw) -> SetResult:
    """``score`` in {"lac", "aps", "aps_rand", "raps", "raps_rand", "top_k"}.
    ``aps_rand`` / ``raps_rand`` are the ``include_last_label="randomized"``
    variants. Keyword arguments go to the constructor."""
    if score == "lac":
        return lac_sets(p_cal, y_cal, p_test, alpha, **kw)
    if score == "aps":
        return aps_sets(p_cal, y_cal, p_test, alpha, seed=seed, **kw)
    if score == "aps_rand":
        r = aps_sets(p_cal, y_cal, p_test, alpha, seed=seed, include_last_label="randomized", **kw)
        r.score = "aps_rand"
        return r
    if score == "raps":
        return raps_sets(p_cal, y_cal, p_test, alpha, seed=seed, **kw)
    if score == "raps_rand":
        r = raps_sets(p_cal, y_cal, p_test, alpha, seed=seed, include_last_label="randomized", **kw)
        r.score = "raps_rand"
        return r
    if score == "top_k":
        return topk_sets(p_cal, y_cal, p_test, alpha, **kw)
    raise ValueError(f"unknown score {score!r}")


def sets_for_cell(cell, score: str, alpha=None, seed=None, **kw) -> SetResult:
    """Build sets for a :class:`tabsets.cache.Cell` with its own level and
    MAPIE seed unless overridden."""
    alpha = cell.alpha if alpha is None else alpha
    seed = cell.mapie_random_state if seed is None else seed
    return build_sets(score, cell.p_cal, cell.y_cal, cell.p_test, alpha, seed=seed, **kw)
