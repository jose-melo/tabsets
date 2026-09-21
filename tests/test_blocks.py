"""tabsets.blocks: the manifest, the blocks, and the five corrections baked into them."""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testlib import run_module  # noqa: E402

from tabsets import blocks as B  # noqa: E402
from tabsets import stats as ST  # noqa: E402

BLOCKS = ("F1", "allA", "C39", "D29", "D7", "DK10")


def test_manifest_loads_and_agrees_with_the_roster():
    m = B.manifest()
    assert len(m) > 0
    assert (m.family == m.model.map(ST.FAMILY_OF_MODEL)).all()
    assert set(m.model) <= set(ST.ORDER)


def test_manifest_refuses_a_family_that_disagrees_with_the_roster():
    """The check that would have caught the family map drifting from the model table."""
    import tempfile

    m = B.manifest().head(50).copy()
    m.loc[m.index[0], "family"] = "not-a-family"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "manifest.csv")
        m.to_csv(p, index=False)
        try:
            B.manifest(path=p)
            raise AssertionError("a disagreeing family was accepted")
        except ValueError as exc:
            assert "roster" in str(exc)


def test_every_block_is_complete_at_ten_seeds():
    for name in BLOCKS:
        b = B.block(name)
        counts = b.groupby(["dataset", "model"]).seed.nunique()
        assert (counts == 10).all(), f"{name} has an incomplete cell"
        assert len(b) == b.dataset.nunique() * b.model.nunique() * 10


def test_blocks_are_disjoint_where_the_article_says_they_are():
    f1, c39, d29 = (set(B.block(n).dataset) for n in ("F1", "C39", "D29"))
    assert not f1 & c39 and not f1 & d29 and not c39 & d29


def test_main_block_drops_the_three_unreproducible_datasets():
    assert not set(B.block("F1").dataset) & set(B.DROP_F1)


def test_the_large_block_excludes_the_model_that_is_incomplete_on_it():
    assert "exaone" not in set(B.block("D29").model)
    assert "exaone" in set(B.block("F1").model)


def test_number_of_classes_is_taken_from_the_covariates():
    """The manifest records one class on the datasets of one synthetic family."""
    m = B.manifest()
    assert (m.K >= 2).all(), "a cell claims fewer than two classes"
    cov = B.covariates().set_index("dataset").K
    common = m[m.dataset.isin(cov.index)]
    assert (common.K.values == cov.loc[common.dataset].values).all()


def test_the_index_agrees_with_itself_about_the_class_count():
    """Both columns carry the same upstream defect, so correcting one is not enough.

    Five datasets are recorded with one class where their cells hold two columns of
    probabilities. A reader filtering on ``n_classes`` would otherwise drop exactly the
    cells whose ``K`` had just been repaired.
    """
    m = B.manifest()
    assert (m.K == m.n_classes).all()
    assert (m.n_classes >= 2).all()


def test_a_cell_reads_its_class_count_off_its_own_probabilities():
    """Whatever the index says, a cell counts classes from the width of its matrix."""
    import numpy as np

    from tabsets.cache import Cell

    c = Cell(path="x", dataset="d", task_type="binclass", model="m", tag="", seed=0,
             machine="t", p_cal=np.full((4, 2), 0.5), y_cal=np.zeros(4, int),
             p_test=np.full((3, 2), 0.5), y_test=np.zeros(3, int),
             idx_cal=np.arange(4), idx_test=np.arange(3), classes=np.arange(2),
             x_cal_sha1="", x_test_sha1="", meta={})
    assert c.n_classes == 2 and c.is_binary


def test_families_partition_the_roster():
    assert set(B.TFM + B.GBDT + B.DEEP + B.CLASSIC) == set(ST.ORDER)
    assert len(B.TFM) == 13 and len(B.TFM_CORE) == 11
    assert not set(B.TFM_CORE) & {"tabpfn", "mitra"}


def test_units_collapse_near_duplicates():
    for kind in ("strict", "orig"):
        u = B.units(kind)
        ds = set(B.block("F1").dataset)
        assert ds <= set(u)
        assert len({u[d] for d in ds}) < len(ds), f"{kind} collapses nothing"


def test_split_jobs_group_a_dataset_and_seed_across_models():
    b = B.block("F1")
    n = 0
    for _dataset, _seed, paths in B.split_jobs(b):
        assert len(paths) == b.model.nunique()
        n += 1
        if n == 5:
            break
    assert n == 5


def test_paths_resolve_under_the_cache_root():
    m = B.manifest()
    assert m.path.str.startswith(B.cache_root()).all()
    assert m.path.map(os.path.basename).nunique() == len(m)


def test_roster_of_a_block_lists_its_models():
    r = B.roster_of("D29")
    assert set(r.model) == set(B.block("D29").model)
    assert isinstance(r, pd.DataFrame)


if __name__ == "__main__":
    run_module(globals())
