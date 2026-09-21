"""Metrics of set-valued classification, empty sets included.

Conventions
-----------
``sets`` is a boolean matrix ``(n, K)``; ``y`` positional labels ``0..K-1``;
``p`` the probability matrix ``(n, K)``. Every function is plain numpy.

The empty-set layer, and the identities it rests on (BRIEF.md, "What is
already measured"):

* **eps_pt** = share of test points whose set is empty. An empty set never
  covers, so ``eps_pt <= 1 - coverage`` holds *exactly in every run*
  (:func:`empty_rate` asserts it). The bound ``eps_pt <= alpha`` holds only
  in expectation over calibration draws (marginal validity); per run it can
  fail by finite-sample noise, so it is a flag (:func:`empty_bound_flag`),
  not an assertion.
* **any empty set in a run => SSCS = 0** under MAPIE's convention (the
  size-0 stratum has coverage 0 and the score is the min over occupied
  strata). 7 377 / 7 377 published runs obey it; the converse fails in 336.
  :func:`sscs` reproduces ``mapie.metrics.classification.classification_ssc_score``
  (``num_bins=None``: one stratum per size ``0..K``, ``nanmin``, no minimum
  count); :func:`sscs_plus` is the same min over the strata of size ``>= 1``
  (**stratum-level** SSCS+, distinct from the run-level "SSCS over non-empty
  runs" used in analysis/tier0.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import sets as _sets

# --------------------------------------------------------------------------- #
# basic set metrics
# --------------------------------------------------------------------------- #
def _as_sets(sets):
    s = np.asarray(sets)
    if s.ndim == 3:  # MAPIE layout (n, K, n_alpha) -> first level
        s = s[:, :, 0]
    return s.astype(bool)


def sizes(sets) -> np.ndarray:
    return _as_sets(sets).sum(axis=1)


def covered(y, sets) -> np.ndarray:
    s = _as_sets(sets)
    return np.take_along_axis(s, np.asarray(y).reshape(-1, 1), axis=1)[:, 0]


def coverage(y, sets) -> float:
    """Marginal coverage, ``classification_coverage_score`` (nanmean of the
    indicator; identical for a boolean matrix)."""
    return float(np.nanmean(covered(y, sets)))


def mean_width(sets) -> float:
    return float(sizes(sets).mean())


def empty_rate(sets, y=None, check=True) -> float:
    """eps_pt = P(set is empty). With ``y`` and ``check``, asserts the exact
    per-run identity ``eps_pt <= 1 - coverage``."""
    eps = float(np.mean(sizes(sets) == 0))
    if y is not None and check:
        miss = 1.0 - coverage(y, sets)
        assert eps <= miss + 1e-12, f"eps_pt={eps} > 1-coverage={miss}: an empty set covered?"
    return eps


def eps_run(sets) -> int:
    """1 if the run has at least one empty set (the ESANN-era 'empty run')."""
    return int((sizes(sets) == 0).any())


def empty_bound_flag(eps_pt: float, alpha: float, n_test: int, z: float = 2.0) -> bool:
    """True when ``eps_pt`` exceeds ``alpha`` by more than ``z`` binomial SDs
    - the in-expectation bound ``E[eps_pt] <= alpha`` looks violated. A
    diagnostic, not a proof: the guarantee is over calibration draws."""
    sd = np.sqrt(alpha * (1 - alpha) / max(n_test, 1))
    return bool(eps_pt > alpha + z * sd)


def singleton_rate(sets) -> float:
    return float(np.mean(sizes(sets) == 1))


def confident_wrong_singleton_rate(y, sets) -> float:
    """Share of test points with a singleton set that misses the label -
    the confident wrong answers (``ws`` in tier0.py; for K = 2 with sizes in
    {0, 1}: ``1 - coverage = eps_pt + ws`` exactly)."""
    sz = sizes(sets)
    return float(np.mean((sz == 1) & ~covered(y, sets)))


def selective_error(y, sets) -> float:
    """P(y not in S | S non-empty): the error of the classifier that abstains
    on empty sets. NaN if every set is empty."""
    ne = sizes(sets) > 0
    if not ne.any():
        return float("nan")
    return float(np.mean(~covered(y, sets)[ne]))


# --------------------------------------------------------------------------- #
# size-stratified coverage
# --------------------------------------------------------------------------- #
def _strata_bounds(K: int, convention: str):
    if convention == "mapie":
        return [(k, k) for k in range(K + 1)]
    if convention == "pooled":
        # Angelopoulos & Bates (2021) style pooled bins for small K: {0,1}, {2..K}
        return [(0, 1), (2, K)] if K >= 2 else [(0, 1)]
    raise ValueError(f"unknown SSC convention {convention!r}")


def ssc_strata(y, sets, convention="mapie", min_stratum_count=None) -> pd.DataFrame:
    """Coverage per set-size stratum. Columns ``size_min, size_max, n,
    coverage``; ``coverage`` is NaN for an empty stratum and, when
    ``min_stratum_count`` is given, for strata with fewer points (sensitivity
    variant - MAPIE has no minimum)."""
    s = _as_sets(sets)
    y = np.asarray(y)
    sz = s.sum(axis=1)
    cov = covered(y, s)
    rows = []
    for lo, hi in _strata_bounds(s.shape[1], convention):
        m = (sz >= lo) & (sz <= hi)
        n = int(m.sum())
        c = float(cov[m].mean()) if n else float("nan")
        if min_stratum_count is not None and n < min_stratum_count:
            c = float("nan")
        rows.append(dict(size_min=lo, size_max=hi, n=n, coverage=c))
    return pd.DataFrame(rows)


def sscs(y, sets, convention="mapie", min_stratum_count=None) -> float:
    """Size-stratified coverage score = min over occupied strata (MAPIE
    ``classification_ssc_score`` when ``convention="mapie"``). Any empty set
    => 0. NaN only if no stratum qualifies."""
    st = ssc_strata(y, sets, convention, min_stratum_count)
    return float(np.nanmin(st["coverage"].values)) if st["coverage"].notna().any() else float("nan")


def sscs_plus(y, sets, convention="mapie", min_stratum_count=None) -> float:
    """SSCS over the strata of size >= 1 (stratum-level SSCS+): the
    conditional-coverage failure that is *not* the empty-set identity.
    NaN when every set is empty."""
    st = ssc_strata(y, sets, convention, min_stratum_count)
    st = st[st["size_max"] >= 1]
    if convention == "pooled":  # the {0,1} stratum restricted to size 1
        s = _as_sets(sets)
        sz = s.sum(1)
        m = sz == 1
        n1 = int(m.sum())
        c1 = float(covered(y, s)[m].mean()) if n1 else float("nan")
        if min_stratum_count is not None and n1 < min_stratum_count:
            c1 = float("nan")
        st = pd.concat([pd.DataFrame([dict(size_min=1, size_max=1, n=n1, coverage=c1)]),
                        st[st["size_min"] >= 2]])
    return float(np.nanmin(st["coverage"].values)) if st["coverage"].notna().any() else float("nan")


def min_stratum_count(sets) -> int:
    """Smallest occupied stratum (MAPIE convention): the number of points the
    SSCS minimum may rest on."""
    _, counts = np.unique(sizes(sets), return_counts=True)
    return int(counts.min())


def n_strata_occupied(sets) -> int:
    return int(len(np.unique(sizes(sets))))


def p_sscs_zero(values) -> float:
    """Share of runs with SSCS == 0 (aggregation helper)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.mean(v == 0)) if len(v) else float("nan")


# --------------------------------------------------------------------------- #
# three accountings of an empty set
# --------------------------------------------------------------------------- #
def fallback_top1(sets, p) -> np.ndarray:
    """Replace every empty set by the argmax singleton."""
    s = _as_sets(sets).copy()
    empty = s.sum(1) == 0
    if empty.any():
        s[np.where(empty)[0], np.asarray(p)[empty].argmax(1)] = True
    return s


def three_accountings(y, sets, p) -> pd.DataFrame:
    """coverage / width / SSCS / SSCS+ under the three readings of an empty set:
    ``miss`` (it is a miss; the ESANN reading), ``abstain`` (the point is
    excluded from every denominator), ``top1`` (the argmax singleton stands
    in). Also the accuracy of the fallback on the formerly-empty points."""
    s = _as_sets(sets)
    y = np.asarray(y)
    p = np.asarray(p)
    empty = s.sum(1) == 0
    ne = ~empty
    rows = [dict(accounting="miss", n_eval=len(y), coverage=coverage(y, s), width=mean_width(s),
                 sscs=sscs(y, s), sscs_plus=sscs_plus(y, s), empty_rate=float(empty.mean()))]
    if ne.any():
        rows.append(dict(accounting="abstain", n_eval=int(ne.sum()), coverage=coverage(y[ne], s[ne]),
                         width=mean_width(s[ne]), sscs=sscs(y[ne], s[ne]), sscs_plus=sscs_plus(y[ne], s[ne]),
                         empty_rate=0.0))
    else:
        rows.append(dict(accounting="abstain", n_eval=0, coverage=np.nan, width=np.nan, sscs=np.nan,
                         sscs_plus=np.nan, empty_rate=0.0))
    s1 = fallback_top1(s, p)
    fb_acc = float((p[empty].argmax(1) == y[empty]).mean()) if empty.any() else float("nan")
    rows.append(dict(accounting="top1", n_eval=len(y), coverage=coverage(y, s1), width=mean_width(s1),
                     sscs=sscs(y, s1), sscs_plus=sscs_plus(y, s1), empty_rate=0.0, fallback_accuracy=fb_acc))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# selective classification
# --------------------------------------------------------------------------- #
def risk_coverage_curve(y, p) -> pd.DataFrame:
    """Risk-coverage curve of the max-probability selector (Geifman &
    El-Yaniv 2017): points sorted by confidence, ``risk(c)`` = error rate of
    the ``c``-fraction most confident. One row per point."""
    y = np.asarray(y)
    p = np.asarray(p)
    conf = p.max(1)
    err = (p.argmax(1) != y).astype(float)
    order = np.argsort(-conf, kind="stable")
    n = len(y)
    cum_err = np.cumsum(err[order])
    k = np.arange(1, n + 1)
    return pd.DataFrame(dict(coverage=k / n, risk=cum_err / k, confidence=conf[order]))


def aurc(y, p) -> dict:
    """Area under the risk-coverage curve (mean risk over the n coverage
    levels; Geifman, Uziel & El-Yaniv 2019) and the excess E-AURC over the
    oracle selector with the same error rate."""
    rc = risk_coverage_curve(y, p)
    a = float(rc["risk"].mean())
    n = len(rc)
    r = float(rc["risk"].iloc[-1])  # overall error
    n_err = int(round(r * n))
    # oracle: all correct first, then the errors
    k = np.arange(1, n + 1)
    oracle = np.maximum(0, k - (n - n_err)) / k
    return dict(aurc=a, e_aurc=a - float(oracle.mean()), top1_error=r)


# --------------------------------------------------------------------------- #
# proper scores and calibration
# --------------------------------------------------------------------------- #
def accuracy(y, p) -> float:
    return float(np.mean(np.asarray(p).argmax(1) == np.asarray(y)))


def f1_weighted(y, p) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, np.asarray(p).argmax(1), average="weighted"))


def auc(y, p) -> float:
    """Binary: ROC AUC of column 1. Multiclass: one-vs-one, weighted (the
    benchmark's ``evaluate_classification``)."""
    from sklearn.metrics import roc_auc_score

    p = np.asarray(p)
    if p.shape[1] == 2:
        return float(roc_auc_score(y, p[:, 1]))
    return float(roc_auc_score(y, p, multi_class="ovo", average="weighted"))


def brier(y, p) -> float:
    """Multiclass Brier score = mean over rows of sum_k (p_k - 1[y=k])^2."""
    p = np.asarray(p, dtype=float)
    onehot = (np.asarray(y)[:, None] == np.arange(p.shape[1])[None, :]).astype(float)
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def log_loss(y, p) -> float:
    """sklearn's ``log_loss`` (clipping included) so the CSV column reproduces."""
    from sklearn.metrics import log_loss as _ll

    p = np.asarray(p)
    return float(_ll(y, p, labels=np.arange(p.shape[1])))


def _bin_ids(score, n_bins, strategy):
    """Bin id per point. ``equal_mass``: sorted points split into n_bins
    (near-)equal groups; ``uniform``: n_bins equal-width bins on [0, 1]."""
    score = np.asarray(score, dtype=float)
    n = len(score)
    if strategy == "equal_mass":
        ids = np.empty(n, dtype=int)
        order = np.argsort(score, kind="stable")
        for b, chunk in enumerate(np.array_split(order, n_bins)):
            ids[chunk] = b
        return ids
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        return np.clip(np.digitize(score, edges[1:-1], right=True), 0, n_bins - 1)
    raise ValueError(f"unknown binning strategy {strategy!r}")


def top_label_ece(y, p, n_bins=15, strategy="equal_mass") -> float:
    """Top-label ECE: bins of the max probability, |accuracy - confidence|
    weighted by bin mass. ``equal_mass`` (default, 15 bins) is the
    recommended estimator (Nixon et al. 2019); ``uniform`` is the classic
    Guo et al. binning. This is the *correct* call for binary tasks too - the
    benchmark's ESANN-era ``ece`` column fed MAPIE the (n, 2) matrix and
    binned the max probability against the class-1 frequency; see
    :func:`ece_legacy_mapie` and tests/test_metrics.py."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    conf = p.max(1)
    correct = (p.argmax(1) == y).astype(float)
    ids = _bin_ids(conf, n_bins, strategy)
    n = len(y)
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def ece_binary(y, p1, n_bins=15, strategy="equal_mass") -> float:
    """Classical binary ECE on the class-1 probability against the class-1
    frequency (the 1-D call of ``mapie.metrics.calibration
    .expected_calibration_error``)."""
    y = np.asarray(y).astype(float)
    p1 = np.asarray(p1, dtype=float)
    ids = _bin_ids(p1, n_bins, strategy)
    n = len(y)
    ece = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            ece += m.sum() / n * abs(y[m].mean() - p1[m].mean())
    return float(ece)


def ece_legacy_mapie(y, p) -> float:
    """The benchmark's ``ece`` column exactly (``utils.evaluate_classification``):
    binary -> ``expected_calibration_error(y, p)`` with the full (n, 2) matrix
    (MAPIE takes ``max`` over columns and compares it with ``y``, i.e. the
    class-1 frequency: wrong by construction); multiclass ->
    ``top_label_ece(y, p)`` with 50 uniform bins. Requires MAPIE."""
    from mapie.metrics.calibration import expected_calibration_error, top_label_ece as _tle

    p = np.asarray(p)
    if p.shape[1] == 2:
        return float(expected_calibration_error(np.asarray(y), p))
    return float(_tle(np.asarray(y), p))


def ece_top_label_15_mapie(y, p) -> float:
    """The benchmark's ``ece_top_label_15`` column: MAPIE ``top_label_ece``
    with 15 uniform bins (per-predicted-label ECE averaged over labels)."""
    from mapie.metrics.calibration import top_label_ece as _tle

    return float(_tle(np.asarray(y), np.asarray(p), num_bins=15))


def brier_decomposition(y, p, n_bins=15, strategy="equal_mass") -> dict:
    """Murphy (1973) calibration-refinement decomposition of the multiclass
    Brier score, one-vs-rest per class and summed: ``reliability``
    (calibration), ``resolution``, ``uncertainty``; ``residual`` is
    ``brier - (rel - res + unc)`` = within-bin variance minus twice the
    within-bin covariance (Stephenson, Coelho & Jolliffe 2008), the terms
    the binned identity leaves out (0 when forecasts equal their bin
    means; can be negative)."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    K = p.shape[1]
    n = len(y)
    rel = res = unc = 0.0
    for k in range(K):
        o = (y == k).astype(float)
        f = p[:, k]
        obar = o.mean()
        unc += obar * (1 - obar)
        ids = _bin_ids(f, n_bins, strategy)
        for b in range(n_bins):
            m = ids == b
            nb = m.sum()
            if nb:
                rel += nb / n * (f[m].mean() - o[m].mean()) ** 2
                res += nb / n * (o[m].mean() - obar) ** 2
    bs = brier(y, p)
    return dict(brier=bs, reliability=float(rel), resolution=float(res),
                uncertainty=float(unc), residual=float(bs - (rel - res + unc)))


def max_prob_mean(p) -> float:
    return float(np.asarray(p).max(1).mean())


# --------------------------------------------------------------------------- #
# sharpness diagnostic and class-wise (Mondrian) LAC
# --------------------------------------------------------------------------- #
def sharpness_diagnostic(p_cal, y_cal, p_test, alpha=0.1) -> dict:
    """The mechanism paragraph as numbers: LAC admits label k iff
    ``p_k >= 1 - qhat``. Returns ``qhat``, the threshold ``tau = 1 - qhat``,
    the share of test rows with ``max p < tau`` (= the LAC empty rate, up to
    the 1e-8 tolerance), the mean margin ``max p - tau`` and the mean max p."""
    r = _sets.lac_sets(p_cal, y_cal, p_test, alpha)
    mp = np.asarray(p_test).max(1)
    tau = 1 - r.qhat
    return dict(qhat=r.qhat, threshold=float(tau), mass_below_threshold=float(np.mean(mp < tau - 1e-8)),
                mean_margin=float(np.mean(mp - tau)), mean_max_prob=float(mp.mean()),
                empty_rate=empty_rate(r.sets))


def mondrian_lac_by_class(p_cal, y_cal, p_test, alpha=0.1, quantile_method="mapie") -> _sets.SetResult:
    """Class-conditional (Mondrian) LAC: one quantile per class from the
    calibration points of that class; label k enters the set iff
    ``1 - p_k <= qhat_k + 1e-8``. Guarantees coverage >= 1-alpha *within each
    true class* (Vovk 2012) at the price of a larger set. Classes with too few
    calibration points for the level get ``qhat_k = 1`` (always included) and
    are listed in ``extra["classes_underfilled"]``."""
    p_cal, y_cal, p_test = _sets._check_inputs(p_cal, y_cal, p_test)
    K = p_cal.shape[1]
    q = np.ones(K)
    under = []
    for k in range(K):
        m = y_cal == k
        if m.sum() >= max(1 / alpha, 1 / (1 - alpha)):
            q[k] = _sets.conformal_quantile(1 - p_cal[m, k], alpha, quantile_method)
        else:
            under.append(k)
    sets = np.less_equal((1 - p_test) - q[None, :], _sets.EPSILON)
    return _sets.SetResult("lac_mondrian", sets, float(np.nan), alpha, len(y_cal), "tabsets",
                           extra=dict(qhat_by_class=q.tolist(), classes_underfilled=under))


def classwise_metrics(y, sets) -> pd.DataFrame:
    """Per true class: n, coverage, empty rate, mean width (class-wise eps)."""
    s = _as_sets(sets)
    y = np.asarray(y)
    sz = s.sum(1)
    cov = covered(y, s)
    rows = []
    for k in range(s.shape[1]):
        m = y == k
        n = int(m.sum())
        rows.append(dict(class_id=k, n=n, coverage=float(cov[m].mean()) if n else np.nan,
                         empty_rate=float((sz[m] == 0).mean()) if n else np.nan,
                         width=float(sz[m].mean()) if n else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# one flat record per (cell, score)
# --------------------------------------------------------------------------- #
def evaluate_sets(y, sets, p, alpha=None, min_stratum_count_sensitivity=None) -> dict:
    """Every set metric for one run, flat. ``p`` is needed for the top-1
    accounting only."""
    s = _as_sets(sets)
    y = np.asarray(y)
    out = dict(
        coverage_rate=coverage(y, s),
        mean_width=mean_width(s),
        ssc_score=sscs(y, s),
        sscs_plus=sscs_plus(y, s),
        sscs_pooled=sscs(y, s, "pooled"),
        sscs_plus_pooled=sscs_plus(y, s, "pooled"),
        empty_set_rate=empty_rate(s, y),
        eps_run=eps_run(s),
        n_empty=int((s.sum(1) == 0).sum()),
        min_stratum_count=min_stratum_count(s),
        n_strata_occupied=n_strata_occupied(s),
        singleton_rate=singleton_rate(s),
        confident_wrong_singleton_rate=confident_wrong_singleton_rate(y, s),
        selective_error=selective_error(y, s),
    )
    if min_stratum_count_sensitivity:
        out[f"ssc_score_min{min_stratum_count_sensitivity}"] = sscs(y, s, min_stratum_count=min_stratum_count_sensitivity)
        out[f"sscs_plus_min{min_stratum_count_sensitivity}"] = sscs_plus(y, s, min_stratum_count=min_stratum_count_sensitivity)
    if alpha is not None:
        out["eps_gt_alpha_flag"] = empty_bound_flag(out["empty_set_rate"], alpha, len(y))
    acc = three_accountings(y, s, p).set_index("accounting")
    for a in ("abstain", "top1"):
        out[f"coverage_{a}"] = float(acc.loc[a, "coverage"])
        out[f"width_{a}"] = float(acc.loc[a, "width"])
        out[f"sscs_{a}"] = float(acc.loc[a, "sscs"])
    out["fallback_top1_accuracy"] = float(acc.loc["top1"].get("fallback_accuracy", np.nan))
    return out


def evaluate_probs(y, p, legacy_mapie=True) -> dict:
    """Every probability-only metric for one cell (score-independent)."""
    y = np.asarray(y)
    p = np.asarray(p)
    dec = brier_decomposition(y, p)
    rc = aurc(y, p)
    out = dict(
        accuracy=accuracy(y, p), f1_score=f1_weighted(y, p), auc=auc(y, p),
        brier=dec["brier"], nll=log_loss(y, p),
        brier_reliability=dec["reliability"], brier_resolution=dec["resolution"],
        brier_uncertainty=dec["uncertainty"], brier_residual=dec["residual"],
        ece_top_label_15_eqmass=top_label_ece(y, p, 15, "equal_mass"),
        ece_top_label_15_uniform=top_label_ece(y, p, 15, "uniform"),
        max_prob_mean=max_prob_mean(p), aurc=rc["aurc"], e_aurc=rc["e_aurc"],
    )
    if p.shape[1] == 2:
        out["ece_binary_15_eqmass"] = ece_binary(y, p[:, 1], 15, "equal_mass")
    if legacy_mapie:
        try:
            out["ece"] = ece_legacy_mapie(y, p)
            out["ece_top_label_15"] = ece_top_label_15_mapie(y, p)
        except ImportError:
            out["ece"] = np.nan
            out["ece_top_label_15"] = np.nan
    return out


def commitment(sets) -> float:
    """``1 - eps``: the share of points on which the set is not empty.

    The article's first factor. A model that answers less often is not thereby worse,
    which is why this is reported beside the confidence rather than folded into it.
    """
    return 1.0 - empty_rate(sets)


def decomposition(y, sets, convention="mapie", min_stratum_count=None) -> dict:
    """The identity the article rests on, for one run.

    The composite size-stratified score is the confidence on the cases the model answers,
    times an indicator that it answered all of them: a single empty set anywhere in a run
    sends the composite to zero however well the rest of the run did. Reporting the two
    factors separately is what lets the families be ordered, because they order in
    opposite directions.
    """
    kw = dict(convention=convention, min_stratum_count=min_stratum_count)
    eps = empty_rate(sets)
    committed = eps <= 0
    plus = sscs_plus(y, sets, **kw)
    composite = sscs(y, sets, **kw)
    return dict(sscs=composite, sscs_plus=plus, commitment=1.0 - eps, empty_rate=eps,
                committed=bool(committed),
                residual=float(composite - (plus if committed else 0.0)))
