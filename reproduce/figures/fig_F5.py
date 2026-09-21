#!/usr/bin/env python3
"""Figure 5 — abstention begins where calibration accuracy crosses the coverage target.

Drawn from the N2b CSVs only (data/N2_ramp_bins.csv, data/N2_forecast_points.csv, data/N2_forecast.csv,
data/N2_binary_exclusivity.csv), with the manuscript's palette, markers and labels from
output/2026-09-19_hplr-commitment-figure/style.py. Three panels, 5.4 in wide:
    (a) eps against calibration accuracy, one line per family (bins of the (dataset, model) pairs), 1 - alpha
        as a vertical line; bins holding fewer than MIN_CELLS pairs are drawn hollow;
    (b) the forecast: every (dataset, model) pair, with the two regime means as horizontal segments;
    (c) K = 2: the count of binary runs by {two-label only, empty only, both, neither}.
Writes reproduce/out/figures/F5_ramp.pdf (and reproduce/out/figures/F5_ramp.png for inspection).

Usage:  python3 -m reproduce.figures.fig_F5.py [--block F1]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402  (sets rcParams, palette, markers, labels; matplotlib Agg)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _preview():
    """Where the PNG previews go: beside the PDFs, under the build directory."""
    d = os.path.join(os.path.dirname(HERE), "out", "figures")
    os.makedirs(d, exist_ok=True)
    return d
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "out", "figures")
W = 5.4
MIN_CELLS = 5
X_LO = 0.30
EPS_LABEL = r"Abstention $\varepsilon$ (empty-set rate)"


def read(name):
    return pd.read_csv(os.path.join(DATA, name))


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=7.5, fontweight="bold", color=st.INK, pad=5)


def panel_a(ax, bins, alpha):
    y_top = float(bins.eps_mean.max()) * 1.25
    for fam in st.FAMILIES:
        b = bins[bins.family == fam].sort_values("acc_cal_mid")
        ax.plot(b.acc_cal_mid, b.eps_mean, color=st.FAM_HUE[fam], lw=1.1, zorder=3)
        full, thin = b[b.n_cells >= MIN_CELLS], b[b.n_cells < MIN_CELLS]
        ax.plot(full.acc_cal_mid, full.eps_mean, **st.mk(fam, size=3.4), zorder=4)
        ax.plot(thin.acc_cal_mid, thin.eps_mean, **st.mk(fam, size=3.4, filled=False), zorder=4)
    ax.axvline(1 - alpha, color=st.GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(1 - alpha - 0.012, y_top * 0.97, r"$1-\alpha$", ha="right", va="top", fontsize=6.5, color=st.GREY)
    ax.set_xlabel("Calibration accuracy")
    ax.set_ylabel(EPS_LABEL)
    ax.set_xlim(X_LO, 1.005)
    ax.set_ylim(-0.003, y_top)
    ax.grid(True, axis="y", zorder=0)
    title(ax, "(a) Null region, then the ramp")
    # direct labels, stacked in the empty upper-left region in the order of the families' last full bin
    # (the four lines ride one ramp, so end-of-line labels would sit on top of each other)
    ends = {fam: bins[(bins.family == fam) & (bins.n_cells >= MIN_CELLS)].sort_values("acc_cal_mid").iloc[-1]
            for fam in st.FAMILIES}
    order = sorted(ends, key=lambda f: ends[f].eps_mean, reverse=True)
    for i, f in enumerate(order):
        ax.text(0.05, 0.95 - 0.1 * i, st.FAM_NAME[f], transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
                color=st.FAM_HUE[f])


def panel_b(ax, pts, fc, alpha):
    for fam in st.FAMILIES:
        p = pts[pts.family == fam]
        kw = st.mk(fam, size=2.4)
        kw.update(alpha=0.5, markeredgewidth=0.2)
        ax.plot(p.acc_cal, p.eps, zorder=3, **kw)
    line = 1 - alpha
    y_top = float(pts.eps.max()) * 1.12
    ax.axvline(line, color=st.GREY, lw=0.7, ls=(0, (3, 2)), zorder=1)
    x0 = float(pts.acc_cal.min())
    for x_lo, x_hi, val in ((x0, line, fc.eps_below), (line, 1.0, fc.eps_above)):
        ax.plot([x_lo, x_hi], [val, val], color=st.INK, lw=1.5, zorder=5, solid_capstyle="butt")
    ax.annotate(rf"$\bar\varepsilon$ = {fc.eps_below:.4f}" + f"\n{int(fc.n_below):,} pairs",
                xy=((x0 + line) / 2, fc.eps_below), xytext=((x0 + line) / 2, y_top * 0.22), ha="center",
                va="bottom", fontsize=6.2, color=st.INK,
                arrowprops=dict(arrowstyle="-", color=st.GREY, lw=0.5, shrinkB=2))
    ax.annotate(rf"$\bar\varepsilon$ = {fc.eps_above:.4f}" + f"\n{int(fc.n_above):,} pairs",
                xy=(0.955, fc.eps_above), xytext=(0.74, y_top * 0.46), ha="center", va="bottom", fontsize=6.2,
                color=st.INK, arrowprops=dict(arrowstyle="-", color=st.GREY, lw=0.5, shrinkB=2))
    ax.text(0.04, 0.97, rf"Spearman $\rho$ = {fc.rho_spearman:+.3f}" + f"\n{int(fc.n_pairs):,} pairs",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color=st.INK)
    ax.set_xlabel("Calibration accuracy")
    ax.set_ylabel(r"$\varepsilon$ per (dataset, model)")
    ax.set_xlim(X_LO, 1.01)
    ax.set_ylim(-0.01, y_top)
    ax.grid(True, axis="y", zorder=0)
    title(ax, "(b) The forecast, pair by pair")


def panel_c(ax, ex):
    cats = [("two-label\nonly", int(ex.n_full_only)), ("empty\nonly", int(ex.n_empty_only)),
            ("both", int(ex.n_both_empty_and_full)), ("neither", int(ex.n_neither))]
    y = np.arange(len(cats))[::-1]
    vals = [v for _, v in cats]
    colors = [st.GREY, st.GREY, "#b3261e", st.GREY]
    ax.barh(y, vals, height=0.62, color=colors, zorder=3)
    top = max(vals)
    for yi, v in zip(y, vals):
        ax.text(v + top * 0.03, yi, f"{v:,}", ha="left", va="center", fontsize=6.6, color=st.INK)
    ax.set_yticks(y)
    ax.set_yticklabels([c for c, _ in cats], fontsize=6.4)
    ax.set_xlabel(f"Runs (of {int(ex.n_binary_runs):,})")
    ax.set_xlim(0, top * 1.5)
    ax.set_xticks([0, 5000, 10000])
    ax.set_xticklabels(["0", "5k", "10k"])
    ax.grid(True, axis="x", zorder=0)
    ax.tick_params(axis="y", length=0)
    title(ax, r"(c) Binary runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="F1")
    a = ap.parse_args()
    bins = read("N2_ramp_bins.csv")
    pts = read("N2_forecast_points.csv")
    fc = read("N2_forecast.csv").iloc[0]
    ex = read("N2_binary_exclusivity.csv").iloc[0]
    for d in (bins, pts):
        assert (d.block == a.block).all() and (d.score == "lac").all(), "CSV block or score differs from the request"
    alpha = float(fc.alpha)

    fig, axes = plt.subplots(1, 3, figsize=(W, 2.3), gridspec_kw=dict(width_ratios=[1.0, 1.0, 0.72]))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.19, top=0.80, wspace=0.6)
    panel_a(axes[0], bins, alpha)
    panel_b(axes[1], pts, fc, alpha)
    panel_c(axes[2], ex)
    handles = [Line2D([], [], **st.mk(f, size=4.0), label=st.FAM_NAME[f]) for f in st.FAMILIES]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.36, 1.0),
               handletextpad=0.3, columnspacing=1.2)
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "F5_ramp.pdf")
    fig.savefig(out)
    fig.savefig(os.path.join(_preview(), "F5_ramp.png"), dpi=220)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
