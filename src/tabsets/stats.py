"""Paired and rank statistics for the model x dataset tables.

Averaging order everywhere: cell (seed mean) -> dataset -> model. The family
contrast is the per-dataset difference between the mean of the TFM models and the
mean of the GBDT models; Wilcoxon signed-rank with the zeros dropped (``wilcox``)
and kept (``pratt``), a 10 000-resample percentile bootstrap CI of the mean, and
W/L/T counts at a 1e-9 tie tolerance.

Model families are read from ``data/models.csv``, the same roster the article
prints as its model table, and never written down a second time here. A literal
map in this module is what once let the package disagree with the released
manifest: it carried seven foundation models where the roster has thirteen, so
every grouping by family silently dropped six of them.
"""

from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

ROSTER_PATH = os.path.join(os.path.dirname(__file__), "data", "models.csv")
_ROSTER = None


def roster() -> pd.DataFrame:
    """The model roster of the release: one row per model, with its family and checkpoint."""
    global _ROSTER
    if _ROSTER is None:
        _ROSTER = pd.read_csv(ROSTER_PATH)
    return _ROSTER.copy()


#: model -> family, derived from the roster so it cannot drift from it.
FAMILY_OF_MODEL = dict(zip(roster().model, roster().family))

#: the roster order, which is the order of the article's model table. No computed
#: quantity depends on it: every use below groups or averages.
ORDER = list(roster().model)

FAMILIES = ("TFM", "GBDT", "classic", "deep")
ID_COLS = ["dataset", "task_type", "model", "tag", "seed"]

# Nemenyi q_0.05 for k models (Demsar 2006, Table 5)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
       11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391}


def family_of(model: str) -> str:
    return FAMILY_OF_MODEL.get(model, "other")


def load_families(path=None) -> dict:
    """dataset -> family id. Unlisted datasets are their own family
    (``"single:<name>"``), as ``tier0.famid`` does."""
    path = path or os.environ.get("TABSETS_DATASET_FAMILIES") or os.path.join(os.path.dirname(__file__), "data", "dataset_families.json")
    fam_sets = json.load(open(path))["families"]
    return {d: f for f, ds in fam_sets.items() for d in ds}


def family_id(dataset: str, families: dict) -> str:
    return families.get(dataset, "single:" + dataset)


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def cell_means(runs: pd.DataFrame, metrics, keys=("dataset", "model"), score=None) -> pd.DataFrame:
    """Seed means per cell. A cell is one (dataset, model) pair over its ten seeds.

    The averaging order of the whole article is seed -> cell -> dataset -> family,
    and it starts here. ``score`` filters ``conformity_score`` when the frame has one.
    """
    df = runs
    if score is not None and "conformity_score" in df:
        df = df[df["conformity_score"] == score]
    metrics = [m for m in metrics if m in df]
    g = df.groupby(list(keys), dropna=False)
    out = g[metrics].mean().reset_index()
    out["n_seeds"] = g.size().values
    out["family"] = out["model"].map(family_of)
    return out


def dataset_family_means(cells: pd.DataFrame, metric: str, models=None) -> pd.DataFrame:
    """dataset x family matrix: the mean over the family's models of their cell means."""
    if models is not None:
        cells = cells[cells.model.isin(models)]
    return cells.pivot_table(index="dataset", columns="family", values=metric, aggfunc="mean")


def paired(delta, label="", n_boot=10000, seed=0) -> dict:
    """A paired difference vector summarised: n, mean, median, a 10 000-resample
    percentile bootstrap CI of the mean, W/L/T at a 1e-9 tie tolerance, Wilcoxon
    signed-rank with the zeros dropped, and a sign test.

    The Wilcoxon p is left missing below six non-zero differences, where the test
    cannot reach conventional significance whatever the data say.
    """
    d = pd.Series(np.asarray(delta, dtype=float)).dropna()
    n = len(d)
    w, l_, t = int((d > 1e-9).sum()), int((d < -1e-9).sum()), int((d.abs() <= 1e-9).sum())
    out = dict(label=label, n=n, mean=float(d.mean()) if n else np.nan,
               median=float(d.median()) if n else np.nan, W=w, L=l_, T=t)
    nz = d[d.abs() > 1e-9]
    out["wilcoxon_p"] = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 6 else np.nan
    out["sign_p"] = float(stats.binomtest(w, w + l_, 0.5).pvalue) if w + l_ else np.nan
    if n > 1:
        rng = np.random.default_rng(seed)
        bs = rng.choice(d.values, size=(n_boot, n), replace=True).mean(axis=1)
        out["ci_lo"], out["ci_hi"] = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    else:
        out["ci_lo"] = out["ci_hi"] = np.nan
    return out


def family_contrast(runs: pd.DataFrame, metric: str, fam_a="TFM", fam_b="GBDT", models_a=None,
                    models_b=None, collapse=None, score=None, n_boot=10000, seed=0) -> dict:
    """Per-dataset ``mean(fam_a models) - mean(fam_b models)`` of the cell means, then
    :func:`paired`.

    ``models_a`` and ``models_b`` override family membership, which is how a contrast is
    restricted to a subset such as the eleven-model core or to a single model.
    ``collapse`` averages the per-dataset differences within a unit first, and takes
    either a name of a unit map (see :func:`tabsets.blocks.units`) or the map itself.
    Near-duplicate datasets otherwise enter the test as independent evidence.
    """
    cells = cell_means(runs, [metric], score=score)
    ma = models_a or [m for m in ORDER if FAMILY_OF_MODEL.get(m) == fam_a]
    mb = models_b or [m for m in ORDER if FAMILY_OF_MODEL.get(m) == fam_b]
    a = cells[cells.model.isin(ma)].groupby("dataset")[metric].mean()
    b = cells[cells.model.isin(mb)].groupby("dataset")[metric].mean()
    delta = (a - b).dropna()
    label = (f"{fam_a if models_a is None else '+'.join(ma)} - "
             f"{fam_b if models_b is None else '+'.join(mb)}: {metric}")
    if collapse is not None:
        if isinstance(collapse, str):
            from .blocks import units as _units
            u = _units(collapse)
            label += f" [{collapse} units]"
        else:
            u = collapse
            label += " [collapsed]"
        delta = delta.groupby(delta.index.map(lambda d: u.get(d, d))).mean()
    out = paired(delta, label, n_boot, seed)
    out["deltas"] = delta
    return out


def model_level_rho(runs: pd.DataFrame, stat: str, against=("accuracy", "eps"), models=None) -> dict:
    """Spearman correlation, over model means, of ``stat`` with each column of ``against``.

    A statistic meant to say something the accuracy and the abstention rate do not
    already say has to be checked against both of them at the level where the claim
    is made, which is the model.
    """
    cols = [stat] + [a for a in against if a in runs]
    cells = cell_means(runs, cols)
    if models is not None:
        cells = cells[cells.model.isin(models)]
    mm = cells.groupby("model")[cols].mean()
    out = {}
    for a in against:
        if a in mm:
            r = stats.spearmanr(mm[stat], mm[a])
            out[a] = (float(r.statistic), float(r.pvalue), len(mm))
    return out


def adjusted_family_effect(runs: pd.DataFrame, stat: str, covars=("accuracy",), fam_a="TFM",
                           fam_b="GBDT", n_boot=2000, seed=0) -> dict:
    """The family effect on ``stat`` holding ``covars`` fixed, with dataset fixed effects
    and a cluster bootstrap over datasets.

    Preferred over matching on an accuracy window, because the matched subset is itself
    selected on the variable being controlled for.
    """
    cols = [stat] + list(covars)
    c = cell_means(runs, cols)
    c = c[c.family.isin([fam_a, fam_b])].dropna(subset=cols)
    c["is_a"] = (c.family == fam_a).astype(float)

    def fit(cc):
        X = cc[["is_a"] + list(covars)].copy()
        y = cc[stat].copy()
        X = X - X.groupby(cc["dataset"]).transform("mean")
        y = y - y.groupby(cc["dataset"]).transform("mean")
        beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
        return beta[0]

    est = fit(c)
    rng = np.random.default_rng(seed)
    ds = c.dataset.unique()
    groups = {d: g for d, g in c.groupby("dataset")}
    bs = []
    for _ in range(n_boot):
        pick = rng.choice(ds, size=len(ds), replace=True)
        cc = pd.concat([groups[d].assign(dataset=f"{d}#{i}") for i, d in enumerate(pick)],
                       ignore_index=True)
        bs.append(fit(cc))
    return dict(stat=stat, covars=list(covars), effect=float(est),
                ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                n_datasets=len(ds), n_cells=len(c))


def empty_free_datasets(runs: pd.DataFrame, eps_col="eps") -> list:
    """The datasets on which no model of the frame emits an empty set in any seed.

    The control for every statistic that conditions on a non-empty set: on these
    datasets the conditioning is vacuous, so a contrast that survives here is not an
    artefact of who abstained. Block-specific, and worth naming with its block.
    """
    g = runs.groupby("dataset")[eps_col].max()
    return sorted(g[g <= 0].index)


def fmt(res: dict, digits=4) -> str:
    """One line for a :func:`paired` result."""
    def f(x):
        return f"{x:+.{digits}f}" if isinstance(x, float) and np.isfinite(x) else str(x)
    return (f"{res.get('label', '')}: mean {f(res['mean'])} [{f(res['ci_lo'])}, {f(res['ci_hi'])}] "
            f"W/L/T {res['W']}/{res['L']}/{res['T']} n={res['n']} p={res['wilcoxon_p']:.2g}")


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    pv = np.asarray(pvals, dtype=float)
    m = len(pv)
    order = np.argsort(pv)
    adj = np.empty(m)
    prev = 0.0
    for rank_i, idx in enumerate(order):
        a = min(1.0, (m - rank_i) * pv[idx])
        prev = max(prev, a)
        adj[idx] = prev
    return adj


def holm_within_metric(contrasts: dict) -> dict:
    """``contrasts``: name -> dict with ``wilcoxon_p`` (one per model pair or
    per family pair within one metric). Adds ``holm_p`` to each."""
    names = list(contrasts)
    pv = [contrasts[k].get("wilcoxon_p", np.nan) for k in names]
    pv_f = [1.0 if (p is None or np.isnan(p)) else p for p in pv]
    adj = holm(pv_f)
    for k, a in zip(names, adj):
        contrasts[k]["holm_p"] = float(a)
    return contrasts


# --------------------------------------------------------------------------- #
# Friedman + Nemenyi (tier0.friedman_block)
# --------------------------------------------------------------------------- #
def critical_difference(k: int, N: int, q=Q05) -> float:
    return q[k] * math.sqrt(k * (k + 1) / (6 * N))


def friedman_nemenyi(cells: pd.DataFrame, metric: str, higher_better=True, models=None,
                     score=None, alpha_sig=0.05) -> dict:
    """Friedman test over the complete dataset x model matrix of cell means,
    average ranks, Nemenyi CD (k = number of models, and k = 13 for the
    journal roster), the pairs beyond CD, scikit-posthocs' Nemenyi p-values
    as a cross-check, Holm-corrected pairwise Wilcoxon, the family mean
    ranks and the count of datasets where GBDT beats TFM."""
    if "conformity_score" in cells and score is not None:
        cells = cells[cells["conformity_score"] == score]
    if "family" not in cells:
        cells = cells.assign(family=cells["model"].map(family_of))
    models = list(models or sorted(cells["model"].unique()))
    Mx = cells.pivot_table(index="dataset", columns="model", values=metric, aggfunc="mean")
    Mx = Mx.reindex(columns=models).dropna()
    N, k = Mx.shape
    if N < 2 or k < 2:
        return dict(metric=metric, N=int(N), k=int(k), error="need >= 2 datasets and >= 2 models")
    chi, p = stats.friedmanchisquare(*[Mx[m].values for m in models]) if k >= 3 else (np.nan, np.nan)
    ranks = Mx.apply(lambda row: stats.rankdata(-row if higher_better else row), axis=1, result_type="expand")
    ranks.columns = models
    avg = ranks.mean()
    CD = critical_difference(k, N) if k in Q05 else np.nan
    CD13 = critical_difference(13, N)
    combos = list(itertools.combinations(models, 2))
    pairs = [(a, b, float(abs(avg[a] - avg[b]))) for a, b in combos]
    sig = [(a, b, round(d, 4)) for a, b, d in pairs if d > CD]
    fam = {m: family_of(m) for m in models}
    tg = [(a, b, d) for a, b, d in sig if {fam[a], fam[b]} == {"TFM", "GBDT"}]
    try:
        import scikit_posthocs as sph

        pn = sph.posthoc_nemenyi_friedman(Mx.values if higher_better else -Mx.values)
        pn.index = pn.columns = models
        n_sig_sph = int(sum(1 for a, b in combos if pn.loc[a, b] < alpha_sig))
        nemenyi_p = pn
    except Exception as e:  # noqa: BLE001
        nemenyi_p, n_sig_sph = None, f"scikit_posthocs failed: {e}"
    pv = []
    for a, b in combos:
        d = Mx[a] - Mx[b]
        nz = d[d.abs() > 1e-12]
        pv.append(float(stats.wilcoxon(nz).pvalue) if len(nz) else 1.0)
    hp = holm(pv)
    holm_sig = [(a, b, float(h)) for (a, b), h in zip(combos, hp) if h < alpha_sig]
    fam_rank = {f: float(np.mean([avg[m] for m in models if fam[m] == f])) for f in FAMILIES
                if any(fam[m] == f for m in models)}
    tf = [m for m in models if fam[m] == "TFM"]
    gb = [m for m in models if fam[m] == "GBDT"]
    gbdt_better = None
    if tf and gb:
        a_, b_ = Mx[tf].mean(axis=1), Mx[gb].mean(axis=1)
        gbdt_better = int(((b_ > a_) if higher_better else (b_ < a_)).sum())
    return dict(metric=metric, higher_better=higher_better, N=int(N), k=int(k), chi2=float(chi), p=float(p),
                avg_rank=avg.round(4).to_dict(), CD=float(CD), CD_k13=float(CD13),
                n_pairs_gt_CD=len(sig), pairs_gt_CD=sig, n_TFMxGBDT_pairs_gt_CD=len(tg), TFMxGBDT_pairs_gt_CD=tg,
                n_pairs_sig_nemenyi_scikit_posthocs=n_sig_sph, nemenyi_p=nemenyi_p,
                n_pairs_holm_wilcoxon_sig=len(holm_sig), holm_wilcoxon_sig=holm_sig,
                holm_p=dict(zip([f"{a}|{b}" for a, b in combos], hp.round(6).tolist())),
                family_mean_rank=fam_rank, GBDT_better_than_TFM_datasets=gbdt_better)


def collapse_families(df: pd.DataFrame, families=None, dataset_col="dataset") -> pd.DataFrame:
    """Group a per-dataset frame by dataset family (mean of numeric
    columns); singletons stay as they are."""
    families = families or load_families()
    fam = df[dataset_col].map(lambda d: family_id(d, families))
    num = df.select_dtypes("number").columns
    out = df.assign(family_id=fam).groupby("family_id")[list(num)].mean().reset_index()
    out["n_members"] = df.assign(family_id=fam).groupby("family_id").size().values
    return out
