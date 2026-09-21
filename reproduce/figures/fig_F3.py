#!/usr/bin/env python3
"""Figure 3 -- three critical-difference diagrams, from data/N1_ranks.csv and data/N1_cd.csv (no recomputation).

    python3 -m reproduce.figures.fig_F3.py --block F1      # reproduce/out/figures/F3_cd.pdf (and reproduce/out/figures/F3_cd.png)

Layout after Demsar (2006): one rank axis per metric (1 = best, on the left), a dot per model at its
mean rank coloured by family, the critical difference drawn as a bar of length CD above the axis,
groups of models not separated at CD joined by a thick bar, and the model labels fanned out on the
left (better half) and right (worse half). Palette, labels and rcParams are the manuscript's
(output/2026-09-19_hplr-commitment-figure/style.py). Full text width, 5.4 in.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _preview():
    """Where the PNG previews go: beside the PDFs, under the build directory."""
    d = os.path.join(os.path.dirname(HERE), "out", "figures")
    os.makedirs(d, exist_ok=True)
    return d
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402  (sets rcParams; read-only)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans"})

W_FULL = 5.4
METRICS = [("auc", "Performance (AUROC)"),
           ("commit_pt", r"Commitment ($1-\varepsilon$)"),
           ("sscsp", r"Confidence on committed (SSCS$^{+}$)")]
# vertical geometry in inches from the top of each panel
IN_TITLE, IN_CD, IN_TICKS, IN_AXIS = 0.07, 0.20, 0.31, 0.40
IN_CLIQUE_GAP, IN_CLIQUE_STEP = 0.10, 0.06
IN_LABEL_GAP, IN_LABEL_STEP, IN_BOTTOM = 0.10, 0.097, 0.05
FS_LABEL = 5.8


def cliques(ranks: pd.Series, cd: float):
    """Maximal groups of rank-consecutive models whose spread is not beyond CD (Demsar's bars)."""
    s = ranks.sort_values()
    names, vals = list(s.index), list(s.values)
    out = []
    for i in range(len(names)):
        j = i
        while j + 1 < len(names) and vals[j + 1] - vals[i] <= cd:
            j += 1
        if j > i and not any(a <= i and b >= j for a, b in out):
            out.append((i, j))
    return [(vals[a], vals[b]) for a, b in out]


def stack_rows(bars):
    """Greedy row assignment so that overlapping clique bars do not sit on the same row."""
    rows = []
    for lo, hi in sorted(bars):
        for r, row in enumerate(rows):
            if all(hi < a or lo > b for a, b in row):
                row.append((lo, hi)); yield (lo, hi, r); break
        else:
            rows.append([(lo, hi)]); yield (lo, hi, len(rows) - 1)


def panel_height(ranks: pd.Series, cd: float, k: int) -> float:
    n_rows = max((r for *_, r in stack_rows(cliques(ranks, cd))), default=-1) + 1
    n_slots = (k + 1) // 2
    return (IN_AXIS + IN_CLIQUE_GAP + n_rows * IN_CLIQUE_STEP + IN_LABEL_GAP + (n_slots - 1) * IN_LABEL_STEP
            + IN_BOTTOM)


def cd_panel(ax, ranks: pd.Series, cd: float, k: int, title: str, n_sep: int, n_tot: int, h_in: float):
    ax.set_xlim(1 - 0.35, k + 0.35)
    ax.set_ylim(0, 1)
    ax.axis("off")
    tr = ax.get_xaxis_transform()   # x in data (rank), y in axes fraction
    Y = lambda inches_from_top: 1 - inches_from_top / h_in
    y_axis, y_ticks, y_title, y_cd = Y(IN_AXIS), Y(IN_TICKS), Y(IN_TITLE), Y(IN_CD)
    # rank axis with integer ticks
    ax.plot([1, k], [y_axis, y_axis], color=style.INK, lw=0.8, transform=tr, zorder=2)
    for t in range(1, k + 1):
        ax.plot([t, t], [y_axis, y_axis + 0.015 / h_in * 1.0], color=style.INK, lw=0.5, transform=tr, zorder=2)
        if t % 3 == 1 or t == k:
            ax.text(t, y_ticks, str(t), ha="center", va="center", fontsize=5.5, color=style.GREY, transform=tr)
    ax.text(1 - 0.5, y_ticks, "mean rank", ha="right", va="center", fontsize=5.5, color=style.GREY, transform=tr)
    # the title on its own line, the critical difference bar on the line below it
    ax.text(k + 0.35, y_title, f"{title}:  {n_sep} of {n_tot} pairs separated", ha="right", va="center",
            fontsize=6.5, color=style.INK, transform=tr)
    ax.plot([1, 1 + cd], [y_cd, y_cd], color=style.INK, lw=1.0, transform=tr, solid_capstyle="butt")
    for x in (1, 1 + cd):
        ax.plot([x, x], [y_cd - 0.03 / h_in, y_cd + 0.03 / h_in], color=style.INK, lw=0.8, transform=tr)
    ax.text(1 + cd + 0.3, y_cd, f"CD = {cd:.2f}", ha="left", va="center", fontsize=6, color=style.INK, transform=tr)
    # clique bars
    n_rows = 0
    for lo, hi, r in stack_rows(cliques(ranks, cd)):
        y = Y(IN_AXIS + IN_CLIQUE_GAP + r * IN_CLIQUE_STEP)
        ax.plot([lo, hi], [y, y], color=style.INK, lw=2.2, solid_capstyle="butt", transform=tr, zorder=3)
        n_rows = max(n_rows, r + 1)
    # models: a dot on the axis, a label fanned out to the side
    s = ranks.sort_values()
    left = list(s.index[: (k + 1) // 2])
    right = list(s.index[(k + 1) // 2:])[::-1]          # worst at the top slot on the right
    top_in = IN_AXIS + IN_CLIQUE_GAP + n_rows * IN_CLIQUE_STEP + IN_LABEL_GAP
    for side, names in (("left", left), ("right", right)):
        for i, m in enumerate(names):
            f = FAMILY[m]
            col = style.FAM_HUE[f]
            x = s[m]
            y = Y(top_in + i * IN_LABEL_STEP)
            ax.plot([x], [y_axis], marker=style.MARKER[f], ms=3.2, color=col, mec="white", mew=0.4, ls="", transform=tr,
                    zorder=4, clip_on=False)
            x_lab = 1 - 0.35 if side == "left" else k + 0.35
            ax.plot([x, x], [y_axis, y], color=col, lw=0.5, alpha=0.9, transform=tr, zorder=1)
            ax.plot([x_lab, x], [y, y], color=col, lw=0.5, alpha=0.9, transform=tr, zorder=1)
            ax.text(x_lab + (-0.12 if side == "left" else 0.12), y, style.LABEL.get(m, m),
                    ha="right" if side == "left" else "left", va="center", fontsize=FS_LABEL, color=col, transform=tr,
                    clip_on=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", default="F1")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "figures", "F3_cd.pdf"))
    a = ap.parse_args()
    ranks = pd.read_csv(os.path.join(ROOT, "data", "N1_ranks.csv"))
    cd = pd.read_csv(os.path.join(ROOT, "data", "N1_cd.csv"))
    wanted = [m for m, _ in METRICS]
    ranks = ranks[ranks.block == a.block]
    cd = cd[(cd.block == a.block) & (cd.metric.isin(wanted))]
    assert len(cd) == len(METRICS), cd
    global FAMILY
    FAMILY = dict(zip(ranks.model, ranks.family))

    panels = []
    for metric, title in METRICS:
        r = ranks[ranks.metric == metric].set_index("model")["mean_rank"]
        row = cd[cd.metric == metric].iloc[0]
        panels.append((r, float(row.critical_difference), int(row.k), title, int(row.n_pairs_separated),
                       int(row.n_pairs_total), panel_height(r, float(row.critical_difference), int(row.k))))
    heights = [p[-1] for p in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(W_FULL, sum(heights) + 0.05 * (len(panels) - 1)),
                             gridspec_kw=dict(height_ratios=heights, hspace=0.05 * (len(panels) - 1) / sum(heights)))
    fig.subplots_adjust(left=0.19, right=0.81, top=1.0, bottom=0.0)
    for ax, p in zip(axes, panels):
        cd_panel(ax, *p)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out)
    png = os.path.join(_preview(), os.path.basename(a.out).replace(".pdf", ".png"))
    fig.savefig(png, dpi=200)
    print(f"wrote {a.out} and {png}")


if __name__ == "__main__":
    main()
