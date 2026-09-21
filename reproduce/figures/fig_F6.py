#!/usr/bin/env python3
"""Figure 6 -- A threshold read from the model's own probabilities holds the level for the
models whose shipped probabilities are calibrated, and inside a family accuracy does not resolve
which those are.

Block F1: 22 models x 113 real datasets x 10 seeds, split conformal under LAC at alpha = 0.10 as
the reference. Panel (a): transductive label-free coverage error (`lftr_coverr`, the threshold
solved on the unlabelled test probabilities) against accuracy (`acc`), one marker per model; the
grey band spans split conformal's own coverage error over the 22 models (`lac_coverr`, min to max);
the within-family Spearman rho is read from P01_rho_F1_a10.csv, row `lftr_coverr`. Panel (b): share
of runs in which the label-free set holds the level (`pass_lf_plain`) per model, with split
conformal's share on the same runs (`pass_lac_plain`) as a grey tick, sorted by family then value.

Reads:  data/P01_model_table_F1_a10.csv, data/P01_rho_F1_a10.csv
Writes: reproduce/out/figures/F6_labelfree.pdf  (PNG preview too if PREVIEW_DIR is set)
"""
import os

from matplotlib.legend_handler import HandlerTuple

from figcommon import (ALPHA, BAND, DATA, FAM_HUE, GREY, INK, LABEL, LINK, Line2D, MUTED, Patch,
                       W_FULL, dot_rows, family_handles, fnum, mk, pd, plt, save, title)

tab = pd.read_csv(os.path.join(DATA, "P01_model_table_F1_a10.csv")).set_index("model")
rho = pd.read_csv(os.path.join(DATA, "P01_rho_F1_a10.csv")).set_index("stat").loc["lftr_coverr"]
assert len(tab) == 22 and set(tab.family) == {"TFM", "GBDT", "deep", "classic"}

# direct labels in panel (a): every non-TFM, the two weakest TFMs, and the TFM cluster as a group.
# offsets in points, checked on the render
LAB_A = {"LogReg": (5, 0, "left"), "knn": (5, 0, "left"),
         "mlp": (-5, 0, "right"), "resnet": (-5, 0, "right"), "ftt": (-5, 0, "right"),
         "tabm": (-5, 0, "right"), "lightgbm": (5, 0, "left"), "xgboost": (5, 0, "left"),
         "catboost": (-5, 0, "right"), "tabpfn": (5, 1, "left"), "mitra": (-5, 0, "right")}
NAME_A = dict(LABEL, tabpfn="TabPFN (v1)", mitra="Mitra (v1)")
SIDE = {"TFM": "TFMs", "GBDT": "GBDTs", "deep": "deep", "classic": "classic"}


def panel_a(ax):
    x, y = tab.acc, tab.lftr_coverr
    lo, hi = tab.lac_coverr.min(), tab.lac_coverr.max()
    ax.axhspan(lo, hi, color=BAND, lw=0, zorder=0)
    for m, r in tab.iterrows():
        ax.plot([r.acc], [r.lftr_coverr], zorder=3, **mk(r.family))
        if m in LAB_A:
            dx, dy, ha = LAB_A[m]
            ax.annotate(NAME_A[m], (r.acc, r.lftr_coverr), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=6.0, color=INK)
    core = tab[(tab.family == "TFM") & ~tab.index.isin(["tabpfn", "mitra"])]
    ax.annotate(f"{len(core)} other TFMs", (core.acc.mean(), core.lftr_coverr.min()),
                xytext=(0, -7), textcoords="offset points", ha="center", va="top", fontsize=6.0,
                color=INK)
    ax.text(0.99, 0.99,
            "Spearman ρ with accuracy\n"
            f"within the 13 TFMs  {fnum(rho.rho_TFM13, 2, sign=True)}, p = {rho.p_TFM13:.2f}\n"
            f"within the 9 others  {fnum(rho.rho_nonTFM9, 2, sign=True)}, p = {rho.p_nonTFM9:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.0, color=GREY, linespacing=1.25)
    ax.grid(True, axis="y"); ax.set_axisbelow(True)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Label-free coverage error\n(threshold solved on the test probabilities)")
    xr = x.max() - x.min()
    ax.set_xlim(x.min() - 0.10 * xr, x.max() + 0.10 * xr)
    ax.set_ylim(0, y.max() * 1.42)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    title(ax, "a", "The error follows the probability\nscale, not the accuracy")


def panel_b(ax):
    rows, centres = dot_rows(tab, "pass_lf_plain", ascending=False)
    ax.axvline(0.85, color=MUTED, lw=0.6, zorder=0)
    for m, r, y in rows:
        ax.plot([r.pass_lf_plain, r.pass_lac_plain], [y, y], color=LINK, lw=0.8, zorder=1,
                solid_capstyle="butt")
        ax.plot([r.pass_lac_plain], [y], marker="|", ms=5, mew=0.9, color=GREY, ls="", zorder=2)
        ax.plot([r.pass_lf_plain], [y], zorder=3, **mk(r.family, size=4.4))
    ax.set_yticks([y for _, _, y in rows])
    ax.set_yticklabels([LABEL[m] for m, _, _ in rows], fontsize=6.0)
    ax.tick_params(axis="y", length=0, pad=2)
    for fam, yc in centres.items():
        ax.text(1.012, yc, SIDE[fam], transform=ax.get_yaxis_transform(), rotation=90, ha="left",
                va="center", fontsize=5.8, color=FAM_HUE[fam], clip_on=False)
    ax.text(0.85, -0.6, "0.85", ha="center", va="bottom", fontsize=5.8, color=MUTED)
    ax.invert_yaxis()
    ax.set_ylim(rows[-1][2] + 0.7, -0.7)
    ax.set_xlim(0.48, 1.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.grid(True, axis="x"); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"Share of runs holding the {1 - ALPHA:.2f} level")
    title(ax, "b", "The level holds in most runs\nwhere the shipped scale is calibrated")


def main():
    H = 3.25
    fig = plt.figure(figsize=(W_FULL, H))
    ax_a = fig.add_axes([0.56 / W_FULL, 0.42 / H, 2.18 / W_FULL, 2.13 / H])
    ax_b = fig.add_axes([3.46 / W_FULL, 0.42 / H, 1.62 / W_FULL, 2.13 / H])
    panel_a(ax_a); panel_b(ax_b)
    handles = family_handles() + [
        (Patch(facecolor=BAND, edgecolor="none"),
         Line2D([], [], marker="|", ms=5, mew=0.9, color=GREY, ls=""))]
    labels = [h.get_label() for h in handles[:-1]] + ["split conformal on the same runs (band, tick)"]
    fig.legend(handles=handles, labels=labels, loc="upper center", ncol=5, frameon=False,
               fontsize=6.3, handletextpad=0.35, columnspacing=1.2, handlelength=1.5,
               bbox_to_anchor=(0.5, 1.0), handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)})
    print(save(fig, "F6_labelfree"))


if __name__ == "__main__":
    main()
