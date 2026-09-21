"""Regenerate the article's floats from the released tables.

Two layers, and only the first is deterministic. ``macros`` and ``tables`` turn the
tables in ``data/`` into the macro file and the LaTeX tables, with no cache and no
randomness, so their output is committed under ``expected/`` and checked by the suite.
``passes`` recomputes those tables from the probability cells; its output moves as runs
are added, which is why it is not compared against anything frozen.
"""
