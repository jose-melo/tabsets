#!/usr/bin/env python3
"""Figure 7 -- Under adaptive sets the threshold is one number for every calibrated model; under
the least ambiguous score it transfers only within the foundation models.

Block F1: 22 models x 113 real datasets x 10 seeds, alpha = 0.10.
Panel (a): coverage error of the constant threshold 1 - alpha = 0.9 under fully randomised APS,
per model (`const_coverr_apsr`, mean over datasets of |coverage - 0.9|), with 0 as the line and a
grey band spanning the family means of the model's OWN split-conformal APS threshold error
(`own_coverr_apsr`, P02_family_table). The error is plotted rather than the delivered coverage
because the mean coverage hides the per-dataset miss: XGBoost delivers 0.893 on average and misses
by 0.039 per dataset.
Panel (b): threshold-transfer error TE(A -> B) = |cov_B(q_A) - 0.9|, median over (dataset, ordered
pair) units of each pair type at matched calibration accuracy, |dacc_cal| <= WINDOW (the
better-powered window of FINDINGS-P02 section 6.5; the 0.0025 window's GBDT figure rests on 60
pairs over 15 datasets and is not drawn), LAC (filled) and randomised APS (open) side by side, with
a 95 % cluster-bootstrap interval of the median (datasets resampled) and n printed under each type.

Reads:  data/P02_model_table_F1_a10.csv, data/P02_family_table_F1_a10.csv,
        data/P01_family_table_F1_a10.csv (the LAC own-threshold error, for the band in b),
        the per-pair transfer table (27 MB), an intermediate not shipped with the release
Writes: data/P02_transfer_by_type.csv  (score,type,window,n_pairs,n_datasets,median,lo,hi,alpha)
        reproduce/out/figures/F7_transfer.pdf   (PNG preview too if PREVIEW_DIR is set)
"""
import os

from figcommon import (ALPHA, BAND, CAMPAIGN, DATA, FAM_HUE, GREY, INK, LABEL, LINK, MUTED, Patch,
                       W_FULL, dot_rows, family_handles, mk, np, pd, plt, save, title)

WINDOW = 0.01          # |dacc_cal| <= WINDOW: the better-powered window (FINDINGS-P02 section 6.5)
B_BOOT, SEED = 2000, 20260920
TYPES = [("TFMxTFM", "TFM-TFM", "within\nTFMs", FAM_HUE["TFM"]),
         ("GBDTxGBDT", "GBDT-GBDT", "within\nGBDTs", FAM_HUE["GBDT"]),
         ("deepxdeep", "deep-deep", "within\ndeep", FAM_HUE["deep"]),
         ("cross-family", "cross-family", "cross-\nfamily", GREY)]
SCORES = [("lac", "te_lac", "LAC"), ("apsr", "te_apsr", "randomised APS")]
SIDE = {"TFM": "TFMs", "GBDT": "GBDTs", "deep": "deep", "classic": "classic"}

mod = pd.read_csv(os.path.join(DATA, "P02_model_table_F1_a10.csv")).set_index("model")
fam2 = pd.read_csv(os.path.join(DATA, "P02_family_table_F1_a10.csv")).set_index("family")
fam1 = pd.read_csv(os.path.join(DATA, "P01_family_table_F1_a10.csv")).set_index("family")
assert len(mod) == 22 and len(fam2) == 4


def boot_median_ci(vals, groups, B=B_BOOT, seed=SEED):
    """95 % percentile interval of the median under a cluster bootstrap: datasets are resampled
    with replacement and every unit of a drawn dataset comes with it (a weighted median)."""
    rng = np.random.default_rng(seed)
    _, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(vals, kind="stable")
    v, g = np.asarray(vals)[order], inv[order]
    G = inv.max() + 1
    out = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, G, G), minlength=G)[g]
        cw = np.cumsum(w)
        out[b] = v[np.searchsorted(cw, cw[-1] / 2.0)]
    return np.percentile(out, [2.5, 97.5])


def transfer_table():
    """The transfer statistic, from the released table, or recomputed from the pairs file.

    The pairs file is an intermediate of the campaign and is not part of the release; the
    table it produces is. Set TABSETS_CAMPAIGN to recompute rather than read.
    """
    released = os.path.join(DATA, "P02_transfer_by_type.csv")
    if not CAMPAIGN and os.path.exists(released):
        return pd.read_csv(released)
    pairs = pd.read_csv(os.path.join(CAMPAIGN, "results", "P02_pairs_typed.csv"),
                        usecols=["dataset", "alpha", "type", "te_lac", "te_apsr", "dacc_cal"])
    p = pairs[(pairs.alpha == ALPHA) & (pairs.dacc_cal <= WINDOW)]
    rows = []
    for key, tag, _, _ in TYPES:
        s = p[p.type == key]
        for score, col, _ in SCORES:
            lo, hi = boot_median_ci(s[col].to_numpy(), s.dataset.to_numpy())
            rows.append(dict(score=score, type=tag, window=WINDOW, n_pairs=len(s),
                             n_datasets=s.dataset.nunique(), median=float(s[col].median()),
                             lo=float(lo), hi=float(hi), alpha=ALPHA))
    out = pd.DataFrame(rows)
    path = os.path.join(DATA, "P02_transfer_by_type.csv")
    out.to_csv(path, index=False, float_format="%.6g")
    print(path); print(out.to_string(index=False))
    return out


def panel_a(ax):
    rows, centres = dot_rows(mod, "const_coverr_apsr", ascending=True)
    lo, hi = fam2.own_coverr_apsr.min(), fam2.own_coverr_apsr.max()
    ax.axvspan(lo, hi, color=BAND, lw=0, zorder=0)
    ax.axvline(0, color=GREY, lw=0.6, zorder=1)
    for m, r, y in rows:
        ax.plot([0, r.const_coverr_apsr], [y, y], color=LINK, lw=0.8, zorder=1, solid_capstyle="butt")
        ax.plot([r.const_coverr_apsr], [y], zorder=3, **mk(r.family, size=4.4))
    ax.set_yticks([y for _, _, y in rows])
    ax.set_yticklabels([LABEL[m] for m, _, _ in rows], fontsize=6.0)
    ax.tick_params(axis="y", length=0, pad=2)
    for fam, yc in centres.items():
        ax.text(1.012, yc, SIDE[fam], transform=ax.get_yaxis_transform(), rotation=90, ha="left",
                va="center", fontsize=5.8, color=FAM_HUE[fam], clip_on=False)
    ax.annotate("own calibrated\nthreshold,\nfamily means", (hi, 1.5), xytext=(0.024, 0.9),
                textcoords="data", ha="left", va="top", fontsize=5.8, color=GREY, linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=1))
    ax.invert_yaxis()
    ax.set_ylim(rows[-1][2] + 0.7, -0.7)
    ax.set_xlim(0, mod.const_coverr_apsr.max() * 1.12)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(True, axis="x"); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Coverage error of the constant\nthreshold 0.9, randomised APS")
    title(ax, "a", "On the TFMs the constant 0.9\nmatches a calibrated threshold")


def panel_b(ax, tt):
    lo = min(fam2.own_coverr_apsr.min(), fam1.lac_coverr.min())
    hi = max(fam2.own_coverr_apsr.max(), fam1.lac_coverr.max())
    ax.axhspan(lo, hi, color=BAND, lw=0, zorder=0)
    bw, off = 0.34, 0.19
    for i, (_, tag, _, hue) in enumerate(TYPES):
        for score, _, _ in SCORES:
            r = tt[(tt.type == tag) & (tt.score == score)].iloc[0]
            x = i - off if score == "lac" else i + off
            if score == "lac":
                ax.bar(x, r["median"], bw, color=hue, lw=0, zorder=2)
            else:
                ax.bar(x, r["median"], bw, facecolor="white", edgecolor=hue, lw=0.9, zorder=2)
            ax.plot([x, x], [r.lo, r.hi], color=INK, lw=0.7, zorder=4, solid_capstyle="butt")
        r = tt[(tt.type == tag) & (tt.score == "lac")].iloc[0]
        ax.text(i, -0.19, f"n = {r.n_pairs:,}\n{r.n_datasets} datasets",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=5.7, color=GREY,
                linespacing=1.15)
    ax.annotate("own calibrated threshold,\nfamily means, both scores", (3.55, hi),
                xytext=(0.98, 0.97), textcoords="axes fraction", ha="right", va="top", fontsize=5.8,
                color=GREY, linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=1))
    ax.set_xticks(range(len(TYPES)))
    ax.set_xticklabels([lab for _, _, lab, _ in TYPES], fontsize=6.4, linespacing=1.1)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_xlim(-0.6, len(TYPES) - 0.4)
    ax.set_ylim(0, tt.hi.max() * 1.18)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.grid(True, axis="y"); ax.set_axisbelow(True)
    ax.set_ylabel("Threshold-transfer error, median\n|coverage of B at A's threshold − 0.9|")
    ax.set_xlabel(f"ordered pairs at matched calibration accuracy\n|Δacc| ≤ {WINDOW:g}", fontsize=7,
                  linespacing=1.15)
    ax.xaxis.set_label_coords(0.5, -0.40)
    title(ax, "b", "Under LAC the threshold transfers\nonly within the TFMs")


def main():
    tt = transfer_table()
    H = 3.7
    fig = plt.figure(figsize=(W_FULL, H))
    ax_a = fig.add_axes([0.74 / W_FULL, 1.03 / H, 1.40 / W_FULL, 1.98 / H])
    ax_b = fig.add_axes([2.95 / W_FULL, 1.03 / H, 2.30 / W_FULL, 1.98 / H])
    panel_a(ax_a); panel_b(ax_b, tt)
    handles = family_handles() + [
        Patch(facecolor=GREY, edgecolor="none", label="LAC (filled)"),
        Patch(facecolor="white", edgecolor=GREY, lw=0.9, label="randomised APS (open)")]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=6.3,
               handletextpad=0.35, columnspacing=1.1, handlelength=1.4, bbox_to_anchor=(0.5, 1.0))
    print(save(fig, "F7_transfer"))


if __name__ == "__main__":
    main()
