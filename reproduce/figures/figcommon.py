"""Shared plumbing for the TabSets result figures (fig_F6.py, fig_F7.py).

The palette, markers, model display names and rcParams are the manuscript's, imported from the
commitment-figure folder's style.py (validated with the dataviz validator on 2026-09-19: aqua is
below 3:1 on white, so family identity is always doubled by marker shape and by direct labels).
Importing style.py loads its own data/models_F1.csv relative to its folder; that is harmless.

Every figure is 5.4 in wide, vector PDF, TrueType fonts embedded (pdf.fonttype 42, set by style.py).
Nothing plotted is typed in: every number is read from a CSV in ../data/ at plot time, except the
transfer statistic of Figure 7b, which is computed from the campaign's pairs file and written to
../data/P02_transfer_by_type.csv by the same script.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")
FIGDIR = os.path.join(REPO, "out", "figures")
CAMPAIGN = os.environ.get("TABSETS_CAMPAIGN", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from style import (FAM_HUE, FAM_NAME, FAMILIES, GREY, GRID, INK, LABEL, MARKER,  # noqa: E402,F401
                   MUTED, mk)

W_FULL = 5.4          # inches, the float register's width
ALPHA = 0.10
MINUS = "−"
BAND = "#ececea"      # the reference band: one step lighter than the grid
LINK = "#d3d3cf"      # dot-to-tick connector in the dot plots
FAM_LEGEND_NAME = {"TFM": "TFMs", "GBDT": "GBDTs", "deep": "deep", "classic": "classical"}


def fnum(v, d=2, sign=False):
    """A number with a typographic minus, for text printed inside a panel."""
    s = f"{v:+.{d}f}" if sign else f"{v:.{d}f}"
    return s.replace("-", MINUS)


def family_handles(size=4.4):
    """Legend handles: one marker per family, in the register's order."""
    return [Line2D([], [], label=FAM_LEGEND_NAME[f], **mk(f, size=size)) for f in FAMILIES]


def title(ax, letter, text, pad=4):
    """Panel title as an assertion, left-aligned, the letter in front."""
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=7.5, fontweight="bold",
                 color=INK, pad=pad, linespacing=1.15)


def dot_rows(tab, value, ascending):
    """Row layout of a per-model dot plot: families in register order, a gap between them,
    models sorted by `value` inside each family. Returns (rows, y_of_family_centre)."""
    rows, centres, y = [], {}, 0.0
    for fam in FAMILIES:
        sub = tab[tab.family == fam].sort_values(value, ascending=ascending)
        ys = []
        for m, r in sub.iterrows():
            rows.append((m, r, y)); ys.append(y); y += 1.0
        centres[fam] = float(np.mean(ys))
        y += 0.9  # the gap
    return rows, centres


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    pdf = os.path.join(FIGDIR, f"{name}.pdf")
    fig.savefig(pdf)
    prev = os.environ.get("PREVIEW_DIR")
    if prev:
        os.makedirs(prev, exist_ok=True)
        fig.savefig(os.path.join(prev, f"{name}.png"), dpi=220)
    plt.close(fig)
    return pdf
