"""Shared style and data for the Performance / Commitment / Confidence figures.
Palette and marker convention are the manuscript's (figs_journal123.py), validated with the dataviz
validator on 2026-09-19: all-pairs CVD dE >= 9.2, normal-vision >= 16.3; aqua < 3:1 on white, so
family identity is always doubled by marker shape and by direct labels."""
import os
import numpy as np, pandas as pd
# A PDF carries its creation time unless told otherwise, which would make two builds of
# the same figure differ in bytes and hide a real change among false ones.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
FIG = os.path.join(os.path.dirname(HERE), "out", "figures")
W_FULL = 6.9
ALPHA = 0.10
FAMILIES = ["TFM", "GBDT", "deep", "classic"]
FAM_HUE = {"TFM": "#2a78d6", "GBDT": "#eb6834", "classic": "#1baf7a", "deep": "#4a3aa7"}
FAM_NAME = {"TFM": "TFMs", "GBDT": "GBDTs", "deep": "deep", "classic": "classical"}
MARKER = {"TFM": "o", "GBDT": "s", "classic": "^", "deep": "D"}
INK, GREY, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e6e3"
LABEL = {"knn": "kNN", "LogReg": "LogReg", "tabm": "TabM", "mlp": "MLP", "PFN-v2": "TabPFN-v2",
         "tabicl": "TabICL", "mitra": "Mitra", "tabpfn": "TabPFN", "lightgbm": "LightGBM",
         "catboost": "CatBoost", "xgboost": "XGBoost", "tabiclv2": "TabICL-v2", "tabpfn3": "TabPFN-3",
         "ftt": "FT-T", "resnet": "ResNet", "tabpfn25": "TabPFN-2.5", "tabpfn25b": "TabPFN-2.5 (syn.)",
         "tabpfn26": "TabPFN-2.6", "mitra@v2": "Mitra-2", "tabldm": "TabLDM", "tabfm": "TabFM",
         "exaone": "EXAONE"}

# the wording is Zé's (sketch of 2026-09-19): keep it
Y_LAB = "Performance (AUROC)"
X_LAB = {"sscs": "Confidence (SSCS)",
         "commit_pt": r"Commitment ($1-\varepsilon$)",
         "sscsp": r"Confidence on committed (SSCS$^{+}$)"}

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "axes.edgecolor": GREY, "xtick.color": GREY, "ytick.color": GREY,
                     "axes.labelcolor": INK, "grid.color": GRID, "grid.linewidth": 0.5,
                     "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "sans-serif", "mathtext.fontset": "dejavusans"})

# BLOCK=F1: expansion, 22 models x 113 real datasets (DATA-REPORT-0919 F1, the default)
# BLOCK=J : frozen journal grid, 15 models x 112 published datasets (the V1 manuscript's evidence base)
BLOCK = os.environ.get("BLOCK", "F1")
def load_cells(block):
    """Per (dataset, model) means, derived from the run index rather than stored again.

    The seed means are a function of the runs, and keeping a second copy on disk is how
    the two drift apart. Checked against the stored table when this was ported: the
    largest disagreement over 2,486 cells was 4e-16.
    """
    runs = pd.read_parquet(os.path.join(DATA, f"runs_{block}.parquet"))
    met = [c for c in ("auc", "eps", "commit_pt", "commit_run", "sscs", "sscsp", "cov", "width")
           if c in runs]
    out = runs.groupby(["dataset", "model"])[met].mean().reset_index()
    roster = pd.read_csv(os.path.join(DATA, "models.csv")).set_index("model")
    out["family"] = out.model.map(roster.family)
    return out


per = pd.read_csv(os.path.join(DATA, f"models_{BLOCK}.csv")).set_index("model")
cells = load_cells(BLOCK)
con = pd.read_csv(os.path.join(DATA, f"family_contrast_{BLOCK}.csv"))
N_DS = int(per.n_datasets.iloc[0]); N_MOD = len(per); N_RUNS = N_DS * N_MOD * 10


def contrast(metric, pair="TFM-GBDT"):
    return con[(con.contrast == pair) & (con.metric == metric)].iloc[0]


def pareto(x, y):
    """Indices of points not dominated when maximising both x and y."""
    x, y = np.asarray(x), np.asarray(y)
    keep = [i for i in range(len(x))
            if not np.any((x >= x[i]) & (y >= y[i]) & ((x > x[i]) | (y > y[i])))]
    return sorted(keep, key=lambda i: x[i])


def fmt_p(p):
    if p >= 0.01:
        return f"p = {p:.2f}"
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    return rf"p = {m:.0f}$\times$10$^{{{e}}}$" if round(m) < 10 else rf"p = 10$^{{{e+1}}}$"


def mk(fam, size=4.2, filled=True):
    kw = dict(marker=MARKER[fam], markersize=size, linestyle="", color=FAM_HUE[fam])
    if filled:
        kw.update(markeredgecolor="white", markeredgewidth=0.45)
    else:
        kw.update(markerfacecolor="white", markeredgecolor=FAM_HUE[fam], markeredgewidth=0.7)
    return kw


def save(fig, name):
    name = name if BLOCK == "F1" else f"{name}_{BLOCK}"
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=220)
    plt.close(fig)
