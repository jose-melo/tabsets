"""Packing cells into shards loses nothing.

The released cells are parquet shards rather than one file per cell: 45,000 files is the
right shape for a machine writing them one at a time and the wrong shape for handing them
to someone else. Packing must be exactly reversible, because every number in the article
is computed from these arrays, and a prediction set is a comparison against a threshold,
so a point sitting on that threshold moves if the value it is compared to is rounded.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testlib import make_cell, run_module  # noqa: E402

from tabsets import cache  # noqa: E402

FIELDS = cache.ROW_ARRAYS


def _write_npz(cell, directory):
    path = os.path.join(directory, f"{cell.name}.npz")
    np.savez(path, p_cal=cell.p_cal, y_cal=cell.y_cal.astype(np.int16),
             p_test=cell.p_test, y_test=cell.y_test.astype(np.int16),
             idx_cal=cell.idx_cal.astype(np.int32), idx_test=cell.idx_test.astype(np.int32),
             classes=cell.classes.astype(np.int16), x_cal_sha1="a" * 40,
             x_test_sha1="b" * 40, meta=cache.json.dumps(cell.meta), qhat_lac=np.array([0.25]))
    return path


def _cells(n=4):
    return [make_cell(dataset=f"d{i // 2}", model=f"m{i % 2}", seed=i, K=3 + i % 2,
                      n_cal=40 + i, n_test=50 + i) for i in range(n)]


def test_a_cell_survives_the_round_trip_exactly():
    with tempfile.TemporaryDirectory() as d:
        for c in _cells():
            want = cache.load_npz(_write_npz(c, d))
            got = cache.row_to_cell(cache.cell_to_row(want))
            for f in FIELDS:
                a, b = getattr(got, f), getattr(want, f)
                assert a.dtype == b.dtype, f"{f}: {a.dtype} != {b.dtype}"
                assert a.shape == b.shape, f"{f}: {a.shape} != {b.shape}"
                assert np.array_equal(a, b), f
            assert got.meta == want.meta
            assert got.x_cal_sha1 == want.x_cal_sha1 and got.x_test_sha1 == want.x_test_sha1
            assert set(got.qhat) == set(want.qhat)
            assert all(np.array_equal(got.qhat[k], want.qhat[k]) for k in got.qhat)


def test_probabilities_are_not_rounded():
    """float64 in, float64 out, to the last bit: a set is a comparison against a threshold."""
    with tempfile.TemporaryDirectory() as d:
        c = make_cell(K=4, n_cal=64, n_test=64)
        want = cache.load_npz(_write_npz(c, d))
        got = cache.row_to_cell(cache.cell_to_row(want))
        assert got.p_cal.dtype == np.float64
        assert got.p_cal.tobytes() == want.p_cal.tobytes()
        assert got.p_test.tobytes() == want.p_test.tobytes()


def test_a_shard_holds_many_cells_and_can_be_read_back_by_name():
    with tempfile.TemporaryDirectory() as d:
        cells = [cache.load_npz(_write_npz(c, d)) for c in _cells(6)]
        shard = os.path.join(d, "cells-000.parquet")
        cache.write_shard(cells, shard)
        assert os.path.getsize(shard) > 0

        back = {c.name: c for c in cache.read_shard(shard)}
        assert set(back) == {c.name for c in cells}
        for c in cells:
            assert np.array_equal(back[c.name].p_test, c.p_test)

        one = cells[2].name
        only = cache.read_shard(shard, names=[one])
        assert [c.name for c in only] == [one]


def test_sets_built_from_a_shard_equal_sets_built_from_the_file():
    """The property that matters downstream, rather than the bytes on their own."""
    from tabsets import metrics, sets

    with tempfile.TemporaryDirectory() as d:
        for c in _cells(4):
            want = cache.load_npz(_write_npz(c, d))
            got = cache.row_to_cell(cache.cell_to_row(want))
            for score in ("lac", "aps"):
                a = sets.sets_for_cell(got, score, alpha=0.10)
                b = sets.sets_for_cell(want, score, alpha=0.10)
                assert np.array_equal(a.sets, b.sets) and a.qhat == b.qhat
                assert metrics.decomposition(got.y_test, a.sets) == \
                       metrics.decomposition(want.y_test, b.sets)


if __name__ == "__main__":
    run_module(globals())


def _load_exporter():
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "export_cache", os.path.join(root, "scripts", "export_cache.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _two_files_one_name(directory):
    """The same cell name in two places with different bytes, as a repeated run leaves."""
    import pandas as pd

    a, b = (os.path.join(directory, d) for d in ("kept", "other"))
    os.makedirs(a), os.makedirs(b)
    cell = make_cell(dataset="d", model="m", seed=1, K=3)
    name = f"{cell.name}.npz"
    np.savez(os.path.join(a, name), p_cal=cell.p_cal, y_cal=cell.y_cal,
             p_test=cell.p_test, y_test=cell.y_test, idx_cal=cell.idx_cal,
             idx_test=cell.idx_test, classes=cell.classes, x_cal_sha1="keep",
             x_test_sha1="", meta=cache.json.dumps(cell.meta))
    np.savez(os.path.join(b, name), p_cal=cell.p_cal * 0 + 1 / 3, y_cal=cell.y_cal,
             p_test=cell.p_test * 0 + 1 / 3, y_test=cell.y_test, idx_cal=cell.idx_cal,
             idx_test=cell.idx_test, classes=cell.classes, x_cal_sha1="other",
             x_test_sha1="", meta=cache.json.dumps(cell.meta))
    man = pd.DataFrame([dict(path=name, source_path=os.path.join("kept", name),
                             dataset="d", model="m", seed="seed1")])
    return man, name


def test_the_export_reads_the_file_the_manifest_chose():
    """A name can belong to several files, and only the manifest says which one counts.

    Repeated runs leave the same cell name in more than one place with different bytes.
    Resolving by name picks an arbitrary one, which silently packs results nobody
    reported; on the real tree that was 11,164 cells of 45,871.
    """
    export_cache = _load_exporter()
    with tempfile.TemporaryDirectory() as d:
        man, name = _two_files_one_name(d)
        found, missing = export_cache.find_sources(man, d)
        assert not missing
        assert found[name[:-4]] == os.path.join(d, "kept", name)
        assert cache.load_npz(found[name[:-4]]).x_cal_sha1 == "keep"


def test_an_ambiguous_name_without_a_recorded_source_is_refused():
    """Without the column, a duplicated name has no right answer, so it is not guessed."""
    export_cache = _load_exporter()
    with tempfile.TemporaryDirectory() as d:
        man, _ = _two_files_one_name(d)
        try:
            export_cache.find_sources(man.drop(columns=["source_path"]), d)
            raise AssertionError("an ambiguous name was resolved anyway")
        except SystemExit as exc:
            assert "more than one file" in str(exc)


def test_the_released_manifest_records_a_source_for_every_cell():
    import pandas as pd

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    man = pd.read_parquet(os.path.join(root, "reproduce", "data", "manifest.parquet"))
    assert "source_path" in man, "the manifest cannot identify which file each cell is"
    assert man.source_path.notna().all()
    assert man.source_path.nunique() == len(man) == man.path.nunique()


def test_the_released_shards_reproduce_the_published_run_table():
    """End to end: a number in the article, rebuilt from the cells as they ship.

    Skipped unless TABSETS_CACHE points at an export, since the cells are deposited
    separately. This is the check that caught the export reading the wrong file for
    11,164 of 45,871 cells: the byte-level verification could not see it, because it
    resolved its own reference the same wrong way.
    """
    import pandas as pd

    from _testlib import require

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(os.path.join(
            os.path.expanduser(os.environ.get("TABSETS_CACHE", "~/.cache/tabsets/cells")),
            "index.parquet")):
        require("tabsets_cache_not_set")          # reported as a skip

    from tabsets import blocks, metrics, sets

    published = pd.read_parquet(os.path.join(root, "reproduce", "data", "runs_F1.parquet"))
    blk = blocks.block("F1")
    rng = np.random.default_rng(0)
    pick = blk.iloc[rng.choice(len(blk), size=25, replace=False)]

    rows = []
    for _, c in pick.iterrows():
        cell = blocks.load(c.path)
        r = sets.sets_for_cell(cell, "lac", alpha=0.10)
        d = metrics.decomposition(cell.y_test, r.sets)
        rows.append(dict(dataset=c.dataset, model=c.model, seed=str(c.seed),
                         eps=d["empty_rate"], sscsp=d["sscs_plus"],
                         cov=metrics.coverage(cell.y_test, r.sets),
                         width=metrics.mean_width(r.sets)))
    m = pd.DataFrame(rows).merge(published, on=["dataset", "model", "seed"],
                                 suffixes=("_s", "_r"))
    assert len(m) == len(rows)

    # A few dozen runs of the whole grid were measured twice, into two directories, and
    # the frozen table kept the other copy for 37 of 24,860. Both are valid runs of the
    # same configuration and the difference sits inside the training noise; the release
    # ships the copy the run index names. So a cell either matches to floating point or
    # is one of those, and the provenance of this file records which datasets they are.
    TWICE = {"Firm-Teacher_Clave-Direction_Classification", "pol", "jm1", "eucalyptus"}
    for col in ("eps", "sscsp", "cov", "width"):
        d = np.abs(m[f"{col}_s"] - m[f"{col}_r"])
        for dataset, gap in zip(m.dataset, d):
            if gap > 1e-12:
                assert dataset in TWICE, f"{col} differs by {gap:.3e} on {dataset}"
                assert gap < 0.05, f"{col} differs by {gap:.3e} on {dataset}, beyond the noise"
