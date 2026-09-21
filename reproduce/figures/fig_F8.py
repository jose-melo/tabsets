#!/usr/bin/env python3
"""Figure 8: every never-empty option delivers more coverage than was requested, and width is what it costs.

Delivered coverage (y) against mean set width (x), family means of model means from data/N2_arms.csv at
alpha = 0.10: one point per family for LAC, one arrow per arm from the LAC point (randomised APS, deterministic
APS, LAC + top-1 default), the requested level 0.90 as a horizontal line, and the re-tuned LAC + top-1 point
(computed on the reachable runs) marked hollow. Family identity is doubled by marker and colour (style.py).
Width 5.4 in, DejaVu Sans.

    python3 -m reproduce.figures.fig_F8.py [--block F1]   -> reproduce/out/figures/F8_arms.pdf
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
os.environ.setdefault("BLOCK", "F1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as st  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
W = 5.4
ARM_LABEL = {"apsr": "randomised APS", "apsd": "deterministic APS", "lac_top1": "LAC + top-1 default",
             "lac_top1_retuned": "LAC + top-1, re-tuned\n(reachable runs)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--block", default="F1")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "figures", "F8_arms.pdf"))
    args = ap.parse_args()
    arms = pd.read_csv(os.path.join(ROOT, "data", "N2_arms.csv"))
    arms = arms[(arms.block == args.block) & (arms.family != "all")]
    A = {(r.arm, r.family): r for _, r in arms.iterrows()}
    level = 1.0 - float(arms.alpha.iloc[0])

    # broken x-axis: the deterministic-APS arm sits at width 2.1-2.4, everything else below 1.7
    fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, figsize=(W, 3.3),
                                  gridspec_kw=dict(width_ratios=[3.4, 1.0], wspace=0.05))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.965, bottom=0.14)
    xs_left = [A[(a, f)].width for a in ("lac", "apsr", "lac_top1") for f in st.FAMILIES] + \
              [A[("lac_top1_retuned", f)].width_fb for f in st.FAMILIES]
    xs_right = [A[("apsd", f)].width for f in st.FAMILIES]
    ax.set_xlim(min(xs_left) - 0.05, max(xs_left) + 0.16)
    ax2.set_xlim(min(xs_right) - 0.06, max(xs_right) + 0.06)
    ys = [A[k].delivered_cov for k in A]
    ax.set_ylim(min(ys) - 0.011, max(ys) + 0.006)
    for a in (ax, ax2):
        a.grid(lw=0.4, color=st.GRID, zorder=0)
        a.axhline(level, color=st.GREY, lw=0.8, ls="--", zorder=1)
    ax.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(axis="y", left=False)
    d = 0.012
    kw = dict(transform=ax.transAxes, color=st.GREY, clip_on=False, lw=0.6)
    ax.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw["transform"] = ax2.transAxes
    ax2.plot((-d * 3.4, +d * 3.4), (-d, +d), **kw)
    ax.text(0.99, level, f"requested level {level:.2f}", transform=ax.get_yaxis_transform(), fontsize=6.5,
            color=st.GREY, ha="right", va="bottom")
    for fam in st.FAMILIES:
        base = A[("lac", fam)]
        for arm in ("apsr", "lac_top1"):
            t = A[(arm, fam)]
            ax.annotate("", xy=(t.width, t.delivered_cov), xytext=(base.width, base.delivered_cov),
                        arrowprops=dict(arrowstyle="-|>", color=st.FAM_HUE[fam], lw=0.7, alpha=0.75,
                                        shrinkA=2.5, shrinkB=2.5, mutation_scale=7), zorder=2)
            ax.plot([t.width], [t.delivered_cov], **st.mk(fam, size=4.0), zorder=4)
        t = A[("apsd", fam)]
        cp = ConnectionPatch(xyA=(base.width, base.delivered_cov), coordsA=ax.transData,
                             xyB=(t.width, t.delivered_cov), coordsB=ax2.transData, arrowstyle="-|>",
                             color=st.FAM_HUE[fam], lw=0.7, alpha=0.75, shrinkA=2.5, shrinkB=2.5, mutation_scale=7,
                             zorder=2)
        fig.add_artist(cp)
        ax2.plot([t.width], [t.delivered_cov], **st.mk(fam, size=4.0), zorder=4)
        ax.plot([base.width], [base.delivered_cov], **st.mk(fam, size=5.2), zorder=5)
        rt = A[("lac_top1_retuned", fam)]
        ax.plot([rt.width_fb], [rt.delivered_cov], **st.mk(fam, size=5.0, filled=False), zorder=5)
    # arm labels, once each
    cen = lambda arm, col="width": np.array([(getattr(A[(arm, f)], col), A[(arm, f)].delivered_cov) for f in st.FAMILIES])
    p = cen("apsr"); ax.annotate("randomised APS", (p[:, 0].max(), p[p[:, 0].argmax(), 1]), xytext=(7, 0),
                                 textcoords="offset points", fontsize=6.5, color=st.GREY, ha="left", va="center")
    p = cen("lac_top1"); ax.annotate("LAC + top-1 default", (p[:, 0].max(), p[:, 1].mean()), xytext=(7, 2),
                                     textcoords="offset points", fontsize=6.5, color=st.GREY, ha="left", va="center")
    p = cen("lac_top1_retuned", "width_fb"); ax.annotate("LAC + top-1, re-tuned to the level\n(reachable runs)",
                                                          (p[:, 0].max(), p[:, 1].mean()), xytext=(8, -2), textcoords="offset points",
                                                          fontsize=6.5, color=st.GREY, ha="left", va="center")
    p = cen("apsd"); ax2.annotate("deterministic APS", (p[:, 0].mean(), p[:, 1].min()), xytext=(0, -20),
                                  textcoords="offset points", fontsize=6.5, color=st.GREY, ha="center", va="top")
    p = cen("lac"); ax.annotate("LAC, as requested", (p[:, 0].min(), p[:, 1].min()), xytext=(min(xs_left) - 0.04, level - 0.0055),
                                textcoords="data", fontsize=6.5, color=st.INK, ha="left", va="center",
                                arrowprops=dict(arrowstyle="-", color=st.GREY, lw=0.5, shrinkB=5),
                                annotation_clip=False)
    ax.set_xlabel("mean set width (labels per prediction)", x=0.62)
    ax.set_ylabel("delivered coverage")
    handles = [Line2D([], [], **st.mk(f, size=4.5), label=st.FAM_NAME[f]) for f in st.FAMILIES]
    handles.append(Line2D([], [], marker="o", ms=4.5, ls="", markerfacecolor="white", markeredgecolor=st.GREY,
                          markeredgewidth=0.7, label="re-tuned point"))
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=1, handletextpad=0.4, borderaxespad=0.6)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(os.path.join(_preview(), os.path.basename(args.out).replace(".pdf", ".png")), dpi=200)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
