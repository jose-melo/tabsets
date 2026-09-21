"""--hpo-objective {accuracy,logloss} (E-lltune, 2026-08-30).

The trial value is a pure function (``model.utils.hpo_trial_value``), tested
here without training anything; the end-to-end check (credit-g x lightgbm,
``--n-trials 2 --hpo-objective logloss --tag lltune``) is a benchmark run whose
cell meta must carry ``hpo_objective == "logloss"`` - see the commit message.

Run with ``pytest tests/`` or plainly ``python tests/test_hpo_objective.py``.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _hplr_testlib import ROOT, run_module  # noqa: E402

from model.utils import (  # noqa: E402
    ACCURACY_TIEBREAK,
    HPO_OBJECTIVES,
    HPO_WORST,
    LOGLOSS_TIEBREAK,
    hpo_objective_formula,
    hpo_trial_value,
)


def test_objectives_declared():
    assert HPO_OBJECTIVES == ("accuracy", "logloss")
    assert set(HPO_WORST) == set(HPO_OBJECTIVES)
    for o in HPO_OBJECTIVES:
        assert o in hpo_objective_formula(o) or "val_" in hpo_objective_formula(o)


def test_accuracy_objective_unchanged():
    # TALENT's value, plus the d410b93 tie-break, exactly as before this flag
    assert hpo_trial_value("accuracy", 0.9, 0.4) == 0.9 - LOGLOSS_TIEBREAK * 0.4
    assert hpo_trial_value("accuracy", 0.9, None) == 0.9
    # accuracy dominates log-loss: a sharper but less accurate model loses
    assert hpo_trial_value("accuracy", 0.91, 1.0) > hpo_trial_value("accuracy", 0.90, 0.1)
    # ... and log-loss breaks an exact accuracy tie
    assert hpo_trial_value("accuracy", 0.9, 0.2) > hpo_trial_value("accuracy", 0.9, 0.3)


def test_logloss_objective_is_negative_logloss():
    assert hpo_trial_value("logloss", 0.9, 0.4) == -0.4 + ACCURACY_TIEBREAK * 0.9
    # log-loss dominates accuracy: a better-calibrated but less accurate model wins
    assert hpo_trial_value("logloss", 0.80, 0.30) > hpo_trial_value("logloss", 0.95, 0.31)
    # ... and accuracy breaks an exact log-loss tie
    assert hpo_trial_value("logloss", 0.95, 0.30) > hpo_trial_value("logloss", 0.80, 0.30)


def test_worst_value_below_every_attainable_value():
    # sklearn caps log-loss at -log(1e-15) ~ 34.54 per row
    worst_attainable_ll = 34.6
    assert HPO_WORST["logloss"] < hpo_trial_value("logloss", 0.0, worst_attainable_ll)
    # accuracy (d410b93): 0.0 sits below every trial whose accuracy exceeds the
    # tie-break weight times the capped log-loss (3.5e-5) - i.e. every real one
    assert HPO_WORST["accuracy"] < hpo_trial_value("accuracy", 1e-3, worst_attainable_ll)
    # a logloss trial that cannot be scored returns the worst value, never 0.0
    # (0.0 would beat every real trial, whose value is -logloss < 0)
    assert hpo_trial_value("logloss", 0.99, None) == HPO_WORST["logloss"]
    assert HPO_WORST["logloss"] < -worst_attainable_ll


def test_unknown_objective_rejected():
    for bad in ("auc", "", None):
        try:
            hpo_trial_value(bad, 0.5, 0.5)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} accepted")


def test_cli_requires_tag_for_non_accuracy_objective():
    """`--hpo-objective logloss` without `--tag` must be refused at argparse
    time (exit 2), before any dataset or model is touched."""
    cmd = [sys.executable, os.path.join(ROOT, "talent_benchmark.py"),
           "--datasets", "credit-g", "--models", "lightgbm", "--hpo-objective", "logloss",
           "--disable-wandb", "--device", "cpu"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 2, (r.returncode, r.stderr[-500:])
    assert "needs --tag" in r.stderr, r.stderr[-500:]
    r = subprocess.run(cmd + ["--hpo-objective", "auc"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 2 and "invalid choice" in r.stderr, r.stderr[-500:]


if __name__ == "__main__":
    run_module(globals())
