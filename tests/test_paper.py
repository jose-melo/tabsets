"""The article's floats regenerate from the released tables, exactly.

This is the claim the release exists to support: every number, every cell of every table
and every figure in the article comes from a file in ``reproduce/data/``, through a
generator in this repository, with nothing typed by hand.

The macro file and the LaTeX tables are byte-compared against a committed copy. The
figures are not: a PDF is a poor thing to keep under version control and a worse thing to
read a diff of, so what is checked is that they are drawn from the tables without error
and that two builds agree byte for byte. That is what makes a real change visible when
the tables are refreshed.

Nothing here touches the probability cells. The tables in ``data/`` are a frozen state of
a benchmark that is still running, so the numbers they carry move as runs land; what may
not move is the relation between those tables and the article's floats. When the tables
are refreshed, the committed expectation is regenerated with them, and its diff is the
record of what changed.
"""
from __future__ import annotations

import filecmp
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _testlib import run_module  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED = os.path.join(ROOT, "reproduce", "expected")
OUT = os.path.join(ROOT, "reproduce", "out")
sys.path.insert(0, ROOT)


def _regenerate_text():
    from reproduce import macros, tables

    shutil.rmtree(os.path.join(OUT, "tables"), ignore_errors=True)
    tables.main()
    macros.main([])


def _relpaths(root, skip="figures"):
    return sorted(p for p in (os.path.relpath(os.path.join(d, f), root)
                              for d, _, fs in os.walk(root) for f in fs)
                  if not p.startswith(skip + os.sep))


def test_macros_and_tables_regenerate_byte_for_byte():
    _regenerate_text()
    want, got = _relpaths(EXPECTED), _relpaths(OUT)
    assert want == got, f"missing {set(want) - set(got)}, extra {set(got) - set(want)}"
    differing = [p for p in want
                 if not filecmp.cmp(os.path.join(EXPECTED, p), os.path.join(OUT, p), shallow=False)]
    assert not differing, f"regenerated output differs from the committed copy: {differing}"


def test_no_macro_is_missing_its_source():
    """A macro whose table is absent is emitted as a visible '??' rather than dropped."""
    text = open(os.path.join(EXPECTED, "numbers.tex")).read()
    assert "??" not in text, "a macro has no source table"
    assert text.count("newcommand") > 300


def test_every_macro_names_where_it_came_from():
    """A macro states the file and column it came from, or that it renames another one."""
    for line in open(os.path.join(EXPECTED, "numbers.tex")):
        if line.startswith("\\newcommand"):
            assert "% source:" in line or "% alias" in line, line.strip()


def test_text_generation_is_deterministic():
    _regenerate_text()
    first = {p: open(os.path.join(OUT, p), "rb").read() for p in _relpaths(OUT)}
    _regenerate_text()
    assert {p: open(os.path.join(OUT, p), "rb").read() for p in _relpaths(OUT)} == first


def test_every_figure_is_drawn_and_two_builds_agree():
    from reproduce import figures

    shutil.rmtree(figures.OUT, ignore_errors=True)
    figures.main()
    drawn = sorted(f for f in os.listdir(figures.OUT) if f.endswith(".pdf"))
    assert len(drawn) == len(figures.FIGURES), f"drew {drawn}"
    for name in figures.FIGURES:
        stem = name.split("_", 1)[1]
        assert any(f.startswith(stem) for f in drawn), f"{stem} missing from {drawn}"

    first = {f: open(os.path.join(figures.OUT, f), "rb").read() for f in drawn}
    figures.main()
    again = {f: open(os.path.join(figures.OUT, f), "rb").read() for f in drawn}
    assert again == first, [f for f in first if first[f] != again[f]]


def test_the_analysis_plan_classifies_its_outcomes_by_a_stated_rule():
    """How many analyses did not hold depends on how the question is read, so the rule is
    executable rather than a convention.

    The outcome strings are prose and admit more than one count: five read exactly "held",
    nine begin with "held", six read exactly "did not hold" and nine contain that phrase.
    The ``outcome_class`` column fixes one reading, and this re-derives it from the prose
    so the two cannot drift.
    """
    import csv

    def classify(outcome):
        o = outcome.strip()
        if "did not hold" in o:
            return "rejected"
        if o == "held":
            return "held"
        if o.startswith("held"):
            return "qualified"
        return "undecided"

    rows = list(csv.DictReader(open(os.path.join(ROOT, "reproduce", "data", "prereg.csv"))))
    assert len(rows) == 20
    for r in rows:
        assert r["outcome_class"] == classify(r["outcome"]), r
    counts = {k: sum(r["outcome_class"] == k for r in rows)
              for k in ("held", "qualified", "rejected", "undecided")}
    assert sum(counts.values()) == 20
    assert counts["rejected"] == 9


def test_the_analysis_plan_table_ignores_the_classification():
    """The column is for counting, not for printing: the table is the article's."""
    text = open(os.path.join(EXPECTED, "tables", "H_prereg.tex")).read()
    for word in ("rejected", "qualified", "undecided", "outcome_class"):
        assert word not in text, word


if __name__ == "__main__":
    run_module(globals())
