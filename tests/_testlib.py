"""Shared helpers: synthetic cells, and running a module without pytest.

The suite builds its own cells rather than reading the released cache, so it runs on a
clean checkout with nothing downloaded. The parity tests against MAPIE are skipped when
MAPIE is absent, and are the reason for the ``test`` extra.
"""
from __future__ import annotations

import os
import sys
import traceback
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
warnings.filterwarnings("ignore")


def make_cell(dataset="synthetic", model="m", seed=0, n_cal=200, n_test=300, K=3,
              concentration=2.0, ties=False, flat_share=0.0, label_noise=None,
              machine="test", tag=""):
    """One cell with probabilities drawn from a Dirichlet, or tied to multiples of 1/k.

    ``ties=True`` imitates a nearest-neighbour classifier, whose probabilities are
    multiples of 1/k and tie at the maximum in most rows. Tie handling is where a set
    implementation most easily diverges from another, so it has its own tests.

    Labels are drawn from ``p`` by default, which makes the probabilities canonically
    calibrated. ``label_noise`` instead takes the most probable label and flips that
    share of it, which makes the model *accurate* rather than merely calibrated.

    ``flat_share`` replaces that share of rows with a uniform distribution. An accurate
    model with a few flat rows is what makes a least-ambiguous set empty, and it is the
    article's own mechanism: being right about nearly everything puts the calibrated
    threshold high, and the points the model is unsure about then reach no label at all.
    The flat share has to stay below the level, or those rows set the quantile
    themselves and nothing is ever abstained on. Without this the empty-set tests are
    vacuous.
    """
    from tabsets.cache import Cell

    rng = np.random.default_rng(abs(hash((dataset, model, seed))) % (2**32))
    n = n_cal + n_test
    if ties:
        k = 5
        counts = rng.multinomial(k, np.ones(K) / K, size=n).astype(float)
        p = counts / k
    else:
        p = rng.dirichlet(np.full(K, concentration), size=n)
    if flat_share:
        flat = rng.random(n) < flat_share
        p[flat] = 1.0 / K
    if label_noise is None:
        y = np.array([rng.choice(K, p=row) for row in p])
    else:
        y = p.argmax(1)
        flip = rng.random(n) < label_noise
        y = np.where(flip, rng.integers(0, K, n), y)
    return Cell(
        path=f"<memory>/{dataset}__{model}__seed{seed}.npz",
        dataset=dataset, task_type="binclass" if K == 2 else "multiclass",
        model=model, tag=tag, seed=seed, machine=machine,
        p_cal=p[:n_cal], y_cal=y[:n_cal], p_test=p[n_cal:], y_test=y[n_cal:],
        idx_cal=np.arange(n_cal), idx_test=np.arange(n_cal, n),
        classes=np.arange(K), x_cal_sha1="", x_test_sha1="",
        meta=dict(confidence_level=0.9, mapie_random_state=seed),
    )


def smoke_cells(n_models=3, n_datasets=2, n_seeds=2, K=3, flat_share=0.05,
                label_noise=0.02):
    """A small grid of cells, enough to exercise a pass end to end.

    Accurate, with a minority of flat rows, so that empty sets occur.
    """
    return [make_cell(dataset=f"d{d}", model=f"m{m}", seed=s, K=K, concentration=0.35,
                      flat_share=flat_share, label_noise=label_noise)
            for d in range(n_datasets) for m in range(n_models) for s in range(n_seeds)]


def require(module):
    """Import an optional test dependency, or skip the test that asked for it.

    ``unittest.SkipTest`` is what pytest reports as a skip and what :func:`run_module`
    below understands, so the suite behaves the same either way. MAPIE is optional
    because the package does not need it to build a set; it needs it to prove that the
    sets it builds are the same ones an independent implementation builds.
    """
    import importlib
    from unittest import SkipTest

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise SkipTest(f"{module} is not installed; pip install 'tabsets[test]'") from exc


def run_module(namespace, argv=None):
    """Run every ``test_*`` function of a module; exit 1 on any failure."""
    names = [n for n in namespace if n.startswith("test_") and callable(namespace[n])]
    sel = (argv or sys.argv)[1:]
    if sel:
        names = [n for n in names if any(s in n for s in sel)]
    from unittest import SkipTest

    failed = skipped = 0
    for n in names:
        try:
            namespace[n]()
            print(f"PASS {n}")
        except SkipTest as exc:
            skipped += 1
            print(f"SKIP {n}: {exc}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {n}")
            traceback.print_exc()
    ran = len(names) - skipped
    print(f"{ran - failed}/{ran} passed" + (f", {skipped} skipped" if skipped else ""))
    sys.exit(1 if failed else 0)
