#!/usr/bin/env python3
"""Figure 4: the abstention difference opens with the level and closes when the level matches calibration error.

Paired TFM - GBDT eps difference against the requested level alpha (0.01 ... 0.20), two lines (LAC, fully
randomised APS) with 95 % paired-bootstrap bands, from data/N2_eps_contrast_by_level.csv; the matched-level
point (c = 1, LAC) from data/N2_matched_level.csv, drawn at its delivered miscoverage 1 - coverage (mean of the
two families' delivered coverage); the effective n (datasets with a non-zero paired difference) per level under
the axis. Style: the manuscript's (commitment-figure style.py). Width 5.4 in, DejaVu Sans.

    python3 -m reproduce.figures.fig_F4.py [--block F1]   -> reproduce/out/figures/F4_levels.pdf
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def _preview():
    """Where the PNG previews go: beside the PDFs, under the build directory."""
    d = os.path.join(os.path.dirname(HERE), "out", "figures")
    os.makedirs(d, exist_ok=True)
    return d
ROOT = os.path.dirname(HERE)
os.environ.setdefault("BLOCK", "F1")           # style.py loads its own tables at import; F1 always exists there
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
W = 5.4
# Hue in this paper means family, so the two scores take ink and the one palette hue whose family
# is absent from the TFM - GBDT contrast; line style and marker carry the distinction as well.
SCORE = {"lac": dict(label="LAC", color=st.INK, marker="o", ls="-"),
         "apsr": dict(label="randomised APS", color=st.FAM_HUE["deep"], marker="s", ls="--")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", default="F1")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "figures", "F4_levels.pdf"))
    args = ap.parse_args()
    con = pd.read_csv(os.path.join(ROOT, "data", "N2_eps_contrast_by_level.csv"))
    con = con[(con.block == args.block) & (con.contrast == "TFM-GBDT")]
    mat = pd.read_csv(os.path.join(ROOT, "data", "N2_matched_level.csv"))
    mat = mat[(mat.block == args.block) & (mat.contrast == "TFM-GBDT") & (mat.score == "lac") & (np.isclose(mat.c, 1.0))].iloc[0]
    n_ds = int(con.n.iloc[0])

    fig, ax = plt.subplots(figsize=(W, 3.25))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.965, bottom=0.335)
    ax.axhline(0, color=st.GRID, lw=0.8, zorder=0)
    ax.grid(axis="y", lw=0.4, color=st.GRID, zorder=0)
    levels = sorted(con.alpha.unique())
    for sc, kw in SCORE.items():
        d = con[con.score == sc].sort_values("alpha")
        ax.fill_between(d.alpha, d.lo, d.hi, color=kw["color"], alpha=0.13, lw=0, zorder=1)
        ax.plot(d.alpha, d["mean"], color=kw["color"], ls=kw["ls"], lw=1.2, marker=kw["marker"], ms=3.6,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        last = d.iloc[-1]
        ax.annotate(kw["label"], (last.alpha, last["mean"]), xytext=(-4, 7 if sc == "apsr" else -11),
                    textcoords="offset points", ha="right", va="center", fontsize=7, color=kw["color"])
    # the matched-level point, at its delivered miscoverage
    x_m = 1.0 - 0.5 * (mat.delivered_cov_TFM + mat.delivered_cov_GBDT)
    ax.errorbar([x_m], [mat["mean"]], yerr=[[mat["mean"] - mat.lo], [mat.hi - mat["mean"]]], fmt="o", ms=5,
                markerfacecolor="white", markeredgecolor=st.INK, markeredgewidth=0.9, ecolor=st.INK, elinewidth=0.8,
                capsize=2, zorder=4)
    ax.annotate(f"level matched to the calibration error (c = 1),\n"
                f"drawn at its delivered 1 $-$ coverage: {mat['mean']:+.4f}, {st.fmt_p(mat.wilcoxon_p)}".replace("-0.", "$-$0."),
                (x_m, mat["mean"]), xytext=(-14, -4), textcoords="offset points", fontsize=6.3, color=st.INK,
                ha="right", va="top", arrowprops=dict(arrowstyle="-", color=st.GREY, lw=0.5, shrinkB=4))
    ax.set_xlabel(r"requested level $\alpha$")
    ax.set_ylabel(r"TFM $-$ GBDT, $\varepsilon$")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{a:.2f}" for a in levels])
    ax.set_xlim(0.0, 0.215)
    lo = min(con.lo.min(), mat.lo) - 0.0035
    hi = max(con.hi.max(), mat.hi) + 0.001
    ax.set_ylim(lo, hi)
    # effective n per level, two rows under the axis
    tr = ax.get_xaxis_transform()
    for i, (sc, kw) in enumerate(SCORE.items()):
        d = con[con.score == sc].set_index("alpha")
        y = -0.25 - 0.09 * i
        ax.text(-0.003, y, "$n_{\\mathrm{eff}}$ " + ("LAC" if sc == "lac" else "APS"), transform=tr, fontsize=6,
                color=kw["color"], ha="right", va="center", clip_on=False)
        for a in levels:
            ax.text(a, y, f"{int(d.loc[a, 'n_effective'])}", transform=tr, fontsize=6, color=kw["color"],
                    ha="center", va="center", clip_on=False)
    ax.text(1.0, -0.445, f"of {n_ds} paired datasets; bands: 95 % paired bootstrap", transform=ax.transAxes,
            fontsize=6, color=st.MUTED, ha="right", va="center")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(os.path.join(_preview(), os.path.basename(args.out).replace(".pdf", ".png")), dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
