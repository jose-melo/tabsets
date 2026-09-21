"""tabsets.stats: the roster, paired contrasts, Holm, Friedman/Nemenyi, unit collapse."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testlib import run_module  # noqa: E402

from tabsets import stats as ST  # noqa: E402


def _synthetic_runs(n_datasets=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_datasets):
        base = rng.random() * 0.3
        for m in ("tabpfn", "tabicl", "xgboost", "catboost", "knn"):
            for s in range(3):
                eps = base + (0.08 if ST.family_of(m) == "TFM" else 0.0) + rng.normal(0, 0.01)
                rows.append(dict(dataset=f"d{i}", task_type="binclass", model=m, tag="", seed=s,
                                 conformity_score="lac", empty_set_rate=max(eps, 0),
                                 auc=rng.random()))
    return pd.DataFrame(rows)


def test_roster_is_the_only_family_source():
    """The map must carry every model of the article, thirteen of them foundation models.

    A literal map in this module once carried seven, so grouping by family silently
    dropped six foundation models from every table that used it.
    """
    r = ST.roster()
    assert len(r) == len(ST.FAMILY_OF_MODEL) == len(ST.ORDER)
    assert dict(zip(r.model, r.family)) == ST.FAMILY_OF_MODEL
    assert sum(v == "TFM" for v in ST.FAMILY_OF_MODEL.values()) == 13
    for m in ("tabfm", "tabpfn26", "tabpfn25b", "exaone", "tabldm", "mitra@v2"):
        assert ST.family_of(m) == "TFM", m
    assert set(ST.FAMILY_OF_MODEL.values()) == set(ST.FAMILIES)


def test_paired_and_family_contrast():
    runs = _synthetic_runs()
    out = ST.family_contrast(runs, "empty_set_rate", score="lac")
    assert out["n"] == 30 and out["W"] >= 28 and out["wilcoxon_p"] < 1e-4
    assert out["ci_lo"] > 0.05 and out["ci_hi"] < 0.11
    assert abs(out["mean"] - 0.08) < 0.01
    z = ST.paired(np.zeros(10))
    assert z["T"] == 10 and np.isnan(z["wilcoxon_p"])


def test_paired_withholds_wilcoxon_when_it_cannot_reach_significance():
    """Below six non-zero differences the two-sided test cannot go under 0.05."""
    assert np.isnan(ST.paired(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))["wilcoxon_p"])
    assert ST.paired(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))["wilcoxon_p"] < 0.05


def test_contrast_accepts_an_explicit_model_list():
    runs = _synthetic_runs()
    both = ST.family_contrast(runs, "empty_set_rate", score="lac")
    one = ST.family_contrast(runs, "empty_set_rate", models_a=["tabpfn"], models_b=["xgboost"])
    assert one["n"] == both["n"]
    assert "tabpfn - xgboost" in one["label"]


def test_collapse_averages_within_a_unit_first():
    runs = _synthetic_runs(n_datasets=4)
    u = {"d0": "u1", "d1": "u1", "d2": "u2", "d3": "u2"}
    flat = ST.family_contrast(runs, "empty_set_rate")
    coll = ST.family_contrast(runs, "empty_set_rate", collapse=u)
    assert flat["n"] == 4 and coll["n"] == 2
    assert abs(coll["mean"] - flat["deltas"].groupby(
        flat["deltas"].index.map(u)).mean().mean()) < 1e-12


def test_holm():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    adj = ST.holm(p)
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])
    assert (np.diff(adj[np.argsort(p)]) >= 0).all()


def test_friedman_nemenyi_synthetic():
    runs = _synthetic_runs()
    cells = ST.cell_means(runs, ["empty_set_rate", "auc"], score="lac")
    fr = ST.friedman_nemenyi(cells, "empty_set_rate", higher_better=False)
    assert fr["N"] == 30 and fr["k"] == 5 and fr["p"] < 1e-3
    assert fr["avg_rank"]["tabpfn"] > fr["avg_rank"]["xgboost"]
    assert abs(fr["CD"] - ST.critical_difference(5, 30)) < 1e-12


def test_dataset_family_collapse():
    fam = ST.load_families()
    assert fam["ada_agnostic"] == "ada" and fam["kc1"] == "nasa-mdp"
    assert ST.family_id("credit-g", fam) == "single:credit-g"
    df = pd.DataFrame(dict(dataset=["ada", "ada_prior", "kc1", "pc1", "credit-g"], x=[1, 3, 2, 4, 5]))
    col = ST.collapse_families(df, fam).set_index("family_id")
    assert len(col) == 3
    assert col.loc["ada", "x"] == 2 and col.loc["nasa-mdp", "n_members"] == 2


def test_model_level_rho_and_empty_free():
    runs = _synthetic_runs()
    runs["eps"] = runs["empty_set_rate"]
    rho = ST.model_level_rho(runs, "auc", against=("eps",))
    assert set(rho) == {"eps"} and rho["eps"][2] == 5
    assert ST.empty_free_datasets(runs.assign(eps=0.0)) == sorted(runs.dataset.unique())
    assert ST.empty_free_datasets(runs) == []


if __name__ == "__main__":
    run_module(globals())
