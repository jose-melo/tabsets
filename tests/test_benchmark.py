"""The harness snapshot: what it documents is checkable, and what it excludes stays out.

``benchmark/`` is the code that wrote the probability cells, kept so that the protocol can
be read. Its README makes one claim a referee would want to verify rather than believe,
that the calibration rows are held out of fitting, tuning, early stopping and in-context
data, and it cites the lines that do it. These tests keep those citations honest and keep
the snapshot from growing back into the working tree it was cut from.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testlib import run_module  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "benchmark")

#: The lines the README points at, and what each has to still be doing.
CITED = {
    1098: ("train_test_split", "X_calib"),
    1174: ("_stratified_split", "X_hpo"),
    1251: ("_stratified_split", "X_fit"),
    1261: ("X_train",),
    1265: ("model.fit",),
    1279: ("predict_proba", "X_calib"),
    1282: ("predict_proba", "X_test"),
}


def test_the_readme_cites_lines_that_still_say_what_it_claims():
    text = open(os.path.join(BENCH, "README.md")).read()
    cited = {int(n) for n in re.findall(r"utils\.py:(\d+)", text)}
    assert cited == set(CITED), f"README cites {sorted(cited)}, expected {sorted(CITED)}"
    lines = open(os.path.join(BENCH, "utils.py")).read().splitlines()
    for n, tokens in CITED.items():
        line = lines[n - 1]
        for t in tokens:
            assert t in line, f"utils.py:{n} no longer contains {t!r}: {line.strip()!r}"


def test_the_calibration_split_is_taken_before_anything_else_uses_the_pool():
    """The order matters as much as the split: tuning and early stopping both draw from
    the training part, and both come after the calibration rows are already out of it."""
    src = open(os.path.join(BENCH, "utils.py")).read().splitlines()
    pool_split = next(i for i, l in enumerate(src) if "X_train, X_calib" in l)
    hpo_split = next(i for i, l in enumerate(src) if "X_hpo_train, X_hpo_val" in l)
    fit = next(i for i, l in enumerate(src) if "model.fit(X_fit, y_fit, X_val, y_val)" in l)
    assert pool_split < hpo_split < fit
    # everything after the pool split draws from X_train, never from X_calib
    for i, l in enumerate(src[pool_split + 1:fit], start=pool_split + 1):
        if "_stratified_split(" in l:
            assert "X_calib" not in src[i] + src[i + 1], f"line {i + 1} splits the calibration rows"


def test_nothing_in_the_package_imports_the_harness():
    for d, _, fs in os.walk(os.path.join(ROOT, "src")):
        for f in fs:
            if f.endswith(".py"):
                text = open(os.path.join(d, f)).read()
                for bad in ("import talent_benchmark", "from talent_benchmark",
                            "import benchmark", "from benchmark"):
                    assert bad not in text, f"{f} imports the harness"


def test_the_harness_is_not_a_package():
    assert not os.path.exists(os.path.join(BENCH, "__init__.py"))


def test_the_snapshot_stays_a_snapshot():
    """No results, no checkpoints, no duplicated variants: the traps of the tree it came from."""
    for d, _, fs in os.walk(BENCH):
        for f in fs:
            p = os.path.join(d, f)
            rel = os.path.relpath(p, BENCH)
            assert os.path.getsize(p) <= 1_000_000, f"{rel} is over 1 MB"
            assert "copy" not in f.lower(), rel
            assert not re.search(r"_v[23]\.py$|_final\.py$|_old\.py$", f), rel
            assert os.path.splitext(f)[1] not in (".pkl", ".npz", ".csv", ".pth", ".cpkt",
                                                  ".ipynb", ".png", ".pdf"), rel


def test_the_configs_kept_are_the_models_the_release_used():
    import csv

    roster = list(csv.DictReader(open(os.path.join(ROOT, "src", "tabsets", "data", "models.csv"))))
    keys = {r["model"].split("@")[0] for r in roster}
    tuned = {r["model"].split("@")[0] for r in roster if r["tuning"].strip().lower() != "none"}

    def stems(sub, strip=""):
        d = os.path.join(BENCH, sub)
        return {f[:-5].replace(strip, "") for f in os.listdir(d) if f.endswith(".json")}

    assert stems("configs/default") <= keys
    assert stems("configs/opt_space") == tuned
    assert stems("tuned", "-tuned") == tuned


if __name__ == "__main__":
    run_module(globals())
