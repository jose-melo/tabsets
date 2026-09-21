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
