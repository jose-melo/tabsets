#!/usr/bin/env python3
"""make_numbers.py -- regenerate paper/numbers.tex from the tables in data/.

Rule: a number reaches the paper only through a macro written here, and every macro names the
file and column it came from (or the document that states a constant no table carries). If a table is
missing, the macro is emitted as a red '??' and the file is listed on stderr; the build then
compiles, and `--check` fails, so a missing table can never reach a PDF unnoticed.

    python3 -m reproduce.macros            # writes paper/numbers.tex
    python3 -m reproduce.macros --check    # exit 1 if stale, or if any macro is '??'

Blocks: the main block is whatever the tables in data/ were produced on (today 22 x 113 real
x 10 seeds). The macros never encode 113: the count is read from the tables, so rerunning the
passes on the full collection and this script swaps every number in the paper.
"""
import argparse, csv, math, pathlib, sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "out" / "numbers.tex"
MISSING = []
LINES = []

# ---- constants no table carries: each names the document that states it -------------
CONSTANTS = [
    ("nNDatasets", "218", "asset total; output/2026-09-07_hplr-expansion-run/DATA-REPORT-0919.md S1"),
    ("nNDatasetsReal", "178", "112 + 4 + 62 real; DATA-REPORT-0919.md S1"),
    ("nNDatasetsPub", "112", "conference datasets; DATA-REPORT-0919.md S1"),
    ("nNDatasetsTierA", "4", "tier A real; DATA-REPORT-0919.md S1"),
    ("nNDatasetsNew", "62", "new real (tier B+C); DATA-REPORT-0919.md S1"),
    ("nNDatasetsSynth", "40", "20 baseline + 20 built to favour trees; DATA-REPORT-0919.md S1"),
    ("nNDatasetsBin", "154", "DATA-REPORT-0919.md S1 (75+2+40+37)"),
    ("nNDatasetsMulti", "64", "DATA-REPORT-0919.md S1 (37+2+25)"),
    ("nRowsMin", "736", "DATA-REPORT-0919.md S1"),
    ("nRowsMax", "149{,}332", "DATA-REPORT-0919.md S1"),
    ("nFeatMin", "3", "DATA-REPORT-0919.md S1"),
    ("nFeatMax", "141", "DATA-REPORT-0919.md S1"),
    ("nKMax", "100", "DATA-REPORT-0919.md S1"),
    ("nNewOutsideRows", "55", "DATA-REPORT-0919.md S1: new datasets outside the published envelope by rows"),
    ("nNewOutsideK", "13", "idem, by classes"),
    ("nNewOutsideFeat", "2", "idem, by features"),
    ("nNewKgtTen", "11", "idem, more than ten classes"),
    ("nNTFM", "13", "foundation models in the collection; DATA-REPORT-0919.md S2"),
    ("nNSeeds", "10", "seeds per (dataset, model); DATA-REPORT-0919.md S1"),
    ("nAlpha", "0.10", "default level of every contrast unless a sentence says otherwise"),
    ("nCoverageTarget", "0.90", "1 - alpha"),
    ("nSplitTrain", "56", "percent; DATA-REPORT-0919.md S1"),
    ("nSplitCal", "24", "percent; DATA-REPORT-0919.md S1"),
    ("nSplitTest", "20", "percent, TALENT's fixed test fifth; DATA-REPORT-0919.md S1"),
    ("nNSplits", "2{,}164", "(dataset, seed) splits whose fingerprints agree; DATA-REPORT-0919.md S1"),
    ("nCells", "45{,}263", "usable cells in the cache; 2026-09-19_tabsets-uq-analysis/CONTEXT.md"),
    ("nOptunaTrials", "25", "DATA-REPORT-0919.md S2"),
    ("nDropFOne", "3", "Fitness_Club_c, VulNoneVul, ringnorm; DATA-REPORT-0919.md S4"),
    ("nCeilingModels", "11", "in-context models run only at K <= 10; CLASS-CEILING-0912.md"),
    ("nDKDatasets", "5", "K > 10 datasets in the many-class block; UNDERSTANDING.md S1.3"),
    ("nDKModels", "7", "models in the many-class block; UNDERSTANDING.md S1.3"),
    ("nQuantileDiffCells", "1{,}174", "runs where the textbook and implemented LAC sets differ; UNDERSTANDING.md S1.5 G10"),
    ("nStrictUnits", "76", "strict source units on the main block; UNDERSTANDING.md S1.5 item 13, P17"),
    ("nGModels", "15", "models of the 31 August grid; ~/hplr-neurocomputing/analysis/journal/tier1_verdicts.txt"),
    ("nGDatasets", "112", "datasets of the 31 August grid; idem"),
]


def add(name, value, src):
    LINES.append(f"\\newcommand{{\\{name}}}{{{value}}}  % source: {src}")


def missing(name, src):
    MISSING.append(f"{name} <- {src}")
    add(name, "\\textcolor{red}{??}", f"MISSING {src}")


def rows(name):
    p = DATA / name
    if not p.exists():
        return None
    with p.open() as fh:
        return list(csv.DictReader(fh))


def F(x):
    return float(x)


def signed(x, nd=4):
    v = F(x); s = f"{v:+.{nd}f}".replace("+", "+").replace("-", "-")
    return "\\ensuremath{" + s.replace("-", "-") + "}"


def plain(x, nd=4):
    return f"{F(x):.{nd}f}"


def ci(lo, hi, nd=4):
    return "\\ensuremath{[" + f"{F(lo):+.{nd}f}, {F(hi):+.{nd}f}" + "]}"


def pval(p):
    p = F(p)
    if math.isnan(p):
        return "\\ensuremath{\\text{n/a}}"
    if p >= 0.01:
        return "\\ensuremath{" + f"{p:.2f}" + "}"
    e = int(math.floor(math.log10(p))); m = p / 10 ** e
    if round(m) >= 10:
        m, e = 1, e + 1
    return "\\ensuremath{" + f"{m:.0f} \\times 10^{{{e}}}" + "}"


def wlt(r, w="pos", l="neg", t="zero"):
    return f"{int(F(r[w]))}\\,/\\,{int(F(r[l]))}\\,/\\,{int(F(r[t]))}"


def pct(x, nd=1):
    return f"{100 * F(x):.{nd}f}"


def thousands(n):
    return f"{int(n):,}".replace(",", "{,}")


def build():
    # LINES and MISSING are module state, so a second call in the same process would
    # append to the first one's output. Harmless in a script run once; not in a library.
    LINES.clear()
    MISSING.clear()
    add_hdr = LINES.append
    add_hdr("% GENERATED by reproduce/macros.py -- DO NOT EDIT BY HAND.")
    add_hdr("% Every macro names the file and column it came from. Regenerate: python3 -m reproduce.macros")
    add_hdr("% Naming: \\n<Thing> = main block (22 models x the datasets in data/, LAC 0.10 unless the name says otherwise);")
    add_hdr("%         \\nD<Thing> = large-dataset block; \\nC<Thing> = synthetic block; \\nG<Thing> = the 31 August 15-model grid.")
    add_hdr("\\providecommand{\\textcolor}[2]{#2}")
    add_hdr("")
    add_hdr("% ---- collection and protocol constants ----------------------------------------")
    for name, value, src in CONSTANTS:
        add(name, value, src)

    # ---- main block: models, families, contrasts -------------------------------------------
    models = rows("models_F1.csv") or []
    add("nNModels", str(len(models)), "data/models_F1.csv, row count")
    add("nCdPairsTotal", str(len(models) * (len(models) - 1) // 2), "k(k-1)/2 for k = rows of data/models_F1.csv")
    by_fam = defaultdict(list)
    for r in models:
        by_fam[r["family"]].append(F(r["auc"]))
    for fam, tag in (("TFM", "TFM"), ("GBDT", "GBDT"), ("deep", "Deep"), ("classic", "Classic")):
        a = by_fam.get(fam, [])
        if a:
            add(f"nCount{tag}", str(len(a)), f"data/models_F1.csv rows with family = {fam}")
            add(f"nAuc{tag}Min", f"{min(a):.3f}", f"data/models_F1.csv min auc over {fam}")
            add(f"nAuc{tag}Max", f"{max(a):.3f}", f"data/models_F1.csv max auc over {fam}")
    fams = {r["family"]: r for r in rows("families_F1.csv") or []}
    for fam, tag in (("TFM", "TFM"), ("GBDT", "GBDT"), ("deep", "Deep"), ("classic", "Classic")):
        r = fams.get(fam)
        if not r:
            continue
        add(f"nAuc{tag}", f"{F(r['auc']):.3f}", f"data/families_F1.csv {fam}.auc")
        add(f"nCommit{tag}", plain(r["commit_pt"]), f"data/families_F1.csv {fam}.commit_pt")
        add(f"nSscs{tag}", plain(r["sscs"]), f"data/families_F1.csv {fam}.sscs")
        add(f"nConf{tag}", plain(r["sscsp"]), f"data/families_F1.csv {fam}.sscsp")
        add(f"nCov{tag}", plain(r["cov"]), f"data/families_F1.csv {fam}.cov")
        add(f"nWidth{tag}", f"{F(r['width']):.3f}", f"data/families_F1.csv {fam}.width")
    con = rows("family_contrast_F1.csv") or []
    if con:
        add("nNDatasetsFOne", con[0]["n"], "data/family_contrast_F1.csv column n")
    else:
        missing("nNDatasetsFOne", "data/family_contrast_F1.csv")
    tag_of = {"auc": "Auc", "sscs": "Sscs", "commit_pt": "Commit", "sscsp": "Conf", "cov": "Cov"}
    for r in con:
        if r["contrast"] != "TFM-GBDT" or r["metric"] not in tag_of:
            continue
        t = tag_of[r["metric"]]; s = f"data/family_contrast_F1.csv TFM-GBDT.{r['metric']}"
        add(f"nGap{t}", signed(r["mean"]), s + ".mean"); add(f"nGap{t}CI", ci(r["lo"], r["hi"]), s + ".lo/hi")
        add(f"nGap{t}P", pval(r["wilcoxon_p"]), s + ".wilcoxon_p"); add(f"nGap{t}WLT", wlt(r), s + ".pos/neg/zero")
        if r["metric"] == "commit_pt":
            add("nEpsGapPerHundred", f"{abs(F(r['mean'])) * 100:.1f}", s + ".mean x 100")
    agg = rows("aggregation_check_F1.csv") or []
    for r in agg:
        if r["aggregation"] == "raw":
            add({"sscs": "nRhoAucSscs", "commit_pt": "nRhoAucCommit", "sscsp": "nRhoAucConf"}[r["metric"]],
                signed(r["rho_all"], 2), f"data/aggregation_check_F1.csv raw.{r['metric']}.rho_all")
    rob = rows("robustness_F1.csv") or []
    if rob:
        add("nRhoCommitWorst", signed(max(F(r["rho_commit"]) for r in rob), 2), "data/robustness_F1.csv max over settings of rho_commit")
        add("nRhoConfWorst", signed(min(F(r["rho_sscsp"]) for r in rob), 2), "data/robustness_F1.csv min over settings of rho_sscsp")
        add("nIdentityViolations", str(sum(int(r["identity_violations"]) for r in rob)), "data/robustness_F1.csv sum of identity_violations")
    dec = rows("decomposition_F1.csv") or []
    summ = next((r for r in dec if r["model"] == "SUMMARY"), None)
    if summ:
        add("nFOneRuns", thousands(summ["n_runs"]), "data/decomposition_F1.csv SUMMARY.n_runs (every run of the main block)")
        add("nSscsZeroShare", pct(summ["share_sscs_zero"]), "data/decomposition_F1.csv SUMMARY.share_sscs_zero")
        add("nZeroEmpty", thousands(summ["zero_from_empty"]), "data/decomposition_F1.csv SUMMARY.zero_from_empty")
        add("nZeroConverse", summ["zero_converse"], "data/decomposition_F1.csv SUMMARY.zero_converse")
        add("nLogVarCommit", f"{F(summ['logvar_share_commit']):.0f}", "data/decomposition_F1.csv SUMMARY.logvar_share_commit")
        add("nLogVarConf", "\\ensuremath{" + f"{F(summ['logvar_share_conf']):.0f}" + "}", "data/decomposition_F1.csv SUMMARY.logvar_share_conf")
    else:
        for n in ["nFOneRuns", "nSscsZeroShare", "nZeroEmpty", "nZeroConverse", "nLogVarCommit", "nLogVarConf"]:
            missing(n, "data/decomposition_F1.csv")
    sz = rows("sizes_F1.csv") or []
    for r in sz:
        t = {"n_cal": "Cal", "n_test": "Test"}[r["quantity"]]
        for q in ["min", "median", "max"]:
            add(f"nN{t}{q.capitalize()}", thousands(r[q]), f"data/sizes_F1.csv {r['quantity']}.{q}")

    # ---- P01 label-free ---------------------------------------------------------------------
    p01 = {r["family"]: r for r in rows("P01_family_table_F1_a10.csv") or []}
    for fam, tag in (("TFM", "TFM"), ("GBDT", "GBDT"), ("deep", "Deep"), ("classic", "Classic")):
        r = p01.get(fam)
        if not r:
            continue
        s = f"data/P01_family_table_F1_a10.csv {fam}"
        add(f"nPassShare{tag}", pct(r["pass_lf_plain"], 0), s + ".pass_lf_plain (percent)")
        add(f"nPassShare{tag}Exact", pct(r["pass_lf_plain"]), s + ".pass_lf_plain")
        add(f"nPassShareConf{tag}", pct(r["pass_lac_plain"]), s + ".pass_lac_plain")
        add(f"nLabelFreeCovErr{tag}", plain(r["lf_coverr"]), s + ".lf_coverr")
        add(f"nTransdCovErr{tag}", plain(r["lftr_coverr"]), s + ".lftr_coverr")
        add(f"nLabelFreeWidth{tag}", f"{F(r['lf_width_fb']):.2f}", s + ".lf_width_fb")
        add(f"nConfWidth{tag}", f"{F(r['lac_width_fb']):.2f}", s + ".lac_width_fb")
        add(f"nLabelFreeEps{tag}", f"{F(r['lf_eps']):.3f}", s + ".lf_eps")
        add(f"nConfEps{tag}", f"{F(r['lac_eps']):.3f}", s + ".lac_eps")
    if "TFM" in p01:
        add("nGuaranteePrice", f"{100 * (F(p01['TFM']['pass_lac_plain']) - F(p01['TFM']['pass_lf_plain'])):.1f}",
            "data/P01_family_table_F1_a10.csv TFM.pass_lac_plain - pass_lf_plain (points)")
    p01m = {r["model"]: r for r in rows("P01_model_table_F1_a10.csv") or []}
    for m, tag in (("LogReg", "LogReg"), ("tabpfn", "TabPFNvOne"), ("tabm", "TabM"), ("knn", "Knn")):
        if m in p01m:
            add(f"nPassShare{tag}", pct(p01m[m]["pass_lf_plain"], 0), f"data/P01_model_table_F1_a10.csv {m}.pass_lf_plain (percent)")
    p01c = rows("P01_contrasts_F1_a10.csv") or []
    r = next((x for x in p01c if x["stat"] == "lftr_coverr" and x["level"] == "dataset"), None)
    if r:
        add("nTransdGap", signed(r["mean"]), "data/P01_contrasts_F1_a10.csv lftr_coverr.dataset.mean")
        add("nTransdGapCI", ci(r["ci_lo"], r["ci_hi"]), "idem ci_lo/ci_hi"); add("nTransdGapWLT", wlt(r, "W", "L", "T"), "idem W/L/T")
    rho = {x["stat"]: x for x in rows("P01_rho_F1_a10.csv") or []}
    if "lftr_coverr" in rho:
        r = rho["lftr_coverr"]
        add("nRhoLfTFM", signed(r["rho_TFM13"], 2), "data/P01_rho_F1_a10.csv lftr_coverr.rho_TFM13")
        add("nRhoLfTFMP", pval(r["p_TFM13"]), "idem p_TFM13"); add("nRhoLfNonTFM", signed(r["rho_nonTFM9"], 2), "idem rho_nonTFM9")
    na = rows("P01_neverabstaining_F1_a10.csv") or []
    r = next((x for x in na if x["stat"] == "lftr_err"), None)
    if r:
        add("nNATransdGap", signed(r["mean"]), "data/P01_neverabstaining_F1_a10.csv lftr_err.mean")
        add("nNATransdGapWLT", wlt(r, "W", "L", "T"), "idem W/L/T")
    else:
        missing("nNATransdGap", "data/P01_neverabstaining_F1_a10.csv"); missing("nNATransdGapWLT", "idem")
    tmp = {x["family"]: x for x in rows("P01_temperature_F1_a10.csv") or []}
    if "knn" in tmp:
        add("nKnnLfErrRaw", plain(tmp["knn"]["lf_coverr_raw"]), "data/P01_temperature_F1_a10.csv knn.lf_coverr_raw")
        add("nKnnLfErrAfterT", plain(tmp["knn"]["lf_coverr_afterT"]), "idem knn.lf_coverr_afterT")
    tmc = {x["stat"]: x for x in rows("P01_temperature_contrasts_F1_a10.csv") or []}
    if "lf_coverr_raw" in tmc:
        add("nLfGapRaw", signed(tmc["lf_coverr_raw"]["mean"]), "data/P01_temperature_contrasts_F1_a10.csv lf_coverr_raw.mean")
        add("nLfGapAfterT", signed(tmc["lf_coverr_afterT"]["mean"]), "idem lf_coverr_afterT.mean")
        add("nLfGapAfterTCI", ci(tmc["lf_coverr_afterT"]["ci_lo"], tmc["lf_coverr_afterT"]["ci_hi"]), "idem ci_lo/ci_hi")
    pit = {x["family"]: x for x in rows("P02_pit_family_F1_a10.csv") or []}
    if "TFM" in pit and "GBDT" in pit:
        add("nPitGapRaw", signed(F(pit["TFM"]["pit_pass_raw"]) - F(pit["GBDT"]["pit_pass_raw"]), 3), "data/P02_pit_family_F1_a10.csv TFM - GBDT pit_pass_raw")
        add("nPitGapAfterT", signed(F(pit["TFM"]["pit_pass_T"]) - F(pit["GBDT"]["pit_pass_T"]), 3), "idem pit_pass_T")

    # ---- P02 constant threshold and transfer -----------------------------------------------
    p02 = {x["family"]: x for x in rows("P02_family_table_F1_a10.csv") or []}
    for fam, tag in (("TFM", "TFM"), ("GBDT", "GBDT")):
        if fam in p02:
            r = p02[fam]; s = f"data/P02_family_table_F1_a10.csv {fam}"
            add(f"nConstCov{tag}", plain(r["const_cov_apsr"]), s + ".const_cov_apsr")
            add(f"nConstCovErr{tag}", plain(r["const_coverr_apsr"]), s + ".const_coverr_apsr")
            add(f"nOwnCovErr{tag}", plain(r["own_coverr_apsr"]), s + ".own_coverr_apsr")
    tr = rows("P02_transfer_by_type.csv") or []
    for r in tr:
        t = {"TFM-TFM": "TFM", "GBDT-GBDT": "GBDT", "deep-deep": "Deep", "cross-family": "Cross"}.get(r["type"])
        if t and r["score"] in ("lac", "apsr"):
            add(f"nTransfer{t}{'Lac' if r['score'] == 'lac' else 'Apsr'}", plain(r["median"]), f"data/P02_transfer_by_type.csv {r['score']}.{r['type']}.median")
            if r["score"] == "lac":
                add(f"nTransfer{t}N", thousands(r["n_pairs"]), f"idem n_pairs")
    if not tr:
        for n in ["nTransferTFMLac", "nTransferGBDTLac", "nTransferTFMApsr", "nTransferGBDTApsr"]:
            missing(n, "data/P02_transfer_by_type.csv")

    # ---- P03 contained mass -----------------------------------------------------------------
    p03 = {x["family"]: x for x in rows("P03_family_table_F1_a10.csv") or []}
    for fam, tag in (("TFM", "TFM"), ("GBDT", "GBDT")):
        if fam in p03:
            add(f"nAbsR{tag}", plain(p03[fam]["apsmapie_absR"]), f"data/P03_family_table_F1_a10.csv {fam}.apsmapie_absR")
    p03m = {x["model"]: x for x in rows("P03_model_table_F1_a10.csv") or []}
    for m, tag in (("xgboost", "Xgb"), ("catboost", "Cat")):
        if m in p03m:
            add(f"n{tag}Promised", f"{F(p03m[m]['apsmapie_Phat']):.3f}", f"data/P03_model_table_F1_a10.csv {m}.apsmapie_Phat")
            add(f"n{tag}Delivered", f"{F(p03m[m]['apsmapie_cov']):.3f}", f"data/P03_model_table_F1_a10.csv {m}.apsmapie_cov")
    p03r = {x["stat"]: x for x in rows("P03_rho_F1_a10.csv") or []}
    if "apsdet_absR" in p03r:
        add("nRhoAbsRDet", signed(p03r["apsdet_absR"]["rho_TFM13"], 2), "data/P03_rho_F1_a10.csv apsdet_absR.rho_TFM13")

    # ---- P08 soup, P17 inference, P16 certificate ------------------------------------------
    soup = rows("P08_soup_contrast.csv") or []
    if soup:
        add("nSoupTFM", signed(soup[0]["mean"]), "data/P08_soup_contrast.csv row 0 (TFM mean - GBDT mean)")
        r = next((x for x in soup if x["contrast"].startswith("soup X7 - GBDT") and "auc_dis" in x["statistic"]), None)
        if r:
            add("nSoupTuned", signed(r["mean"]), "data/P08_soup_contrast.csv soup X7 - GBDT mean (auc_dis)")
    se = rows("P17_se_ratios.csv") or []
    for r in se:
        if r["scope"].startswith("all pairs (113") or r["scope"].startswith("all pairs ("):
            if r["metric"] == "accuracy":
                add("nSeRatioAcc", f"{F(r['med_se_ratio']):.3f}", "data/P17_se_ratios.csv all pairs.accuracy.med_se_ratio")
                add("nSepSeed", pct(r["sep_seed_t2"]), "idem sep_seed_t2"); add("nSepPoint", pct(r["sep_point_t2"]), "idem sep_point_t2")
            if r["metric"] == "brier":
                add("nSeRatioBrier", f"{F(r['med_se_ratio']):.3f}", "data/P17_se_ratios.csv all pairs.brier.med_se_ratio")
    mde = rows("P17_mde_card.csv") or []
    r = next((x for x in mde if x["metric"] == "eps" and x["alpha_s"] in ("0.1", "0.10")), None)
    if r:
        add("nMdeUnitsEps", f"{F(r['D_at_power80']):.0f}", "data/P17_mde_card.csv eps.0.1.D_at_power80")
    cert = rows("P16_certificate.csv") or []
    lac10 = [x for x in cert if x["score"] == "LAC" and F(x["alpha"]) == 0.1 and x["block"] == "all"]
    lac20 = [x for x in cert if x["score"] == "LAC" and F(x["alpha"]) == 0.2 and x["block"] == "all"]
    aps10 = [x for x in cert if x["score"] == "APS" and F(x["alpha"]) == 0.1 and x["block"] == "all"]
    if lac10:
        add("nCertModels", str(sum(x["verdict"] == "CERTIFIED" for x in lac10)), "data/P16_certificate.csv LAC 0.1 all, CERTIFIED count")
        add("nCertAlphaTwenty", str(sum(x["verdict"] == "CERTIFIED" for x in lac20)), "data/P16_certificate.csv LAC 0.2 all, CERTIFIED count")
        add("nCertDevTwenty", str(sum(x["verdict"] != "CERTIFIED" for x in lac20)), "data/P16_certificate.csv LAC 0.2 all, not certified")
        add("nApsOverMin", plain(min(F(x["mean_residual"]) for x in aps10)), "data/P16_certificate.csv APS 0.1 all, min mean_residual")
        add("nApsOverMax", plain(max(F(x["mean_residual"]) for x in aps10)), "data/P16_certificate.csv APS 0.1 all, max mean_residual")
    apsr = rows("P16_apsrand_models_all.csv") or []
    if apsr:
        vals = sorted(abs(F(x["R_apsrand"])) for x in apsr)
        add("nApsrResidualMax", plain(vals[-2]), "data/P16_apsrand_models_all.csv second-largest |R_apsrand| (all but one model)")
        add("nApsrCertModels", str(len(vals) - 1), "data/P16_apsrand_models_all.csv model count minus the one outlier")

    # ---- Vovk efficiency criteria within the TFM family, TabPFN 2.5 pair, alpha sweep --------
    vk = rows("P20_vovk_rho.csv") or []
    if vk:
        from scipy import stats as _st
        tfm_rows = [r for r in vk if r["model"] in ("tabfm", "tabpfn3", "tabiclv2", "tabpfn26", "tabpfn25", "tabpfn25b", "PFN-v2", "tabicl", "exaone", "tabldm", "mitra@v2", "mitra", "tabpfn")]
        n_follow = 0
        for c in ["S", "N", "U", "F", "M", "E", "OU", "OF", "OM", "OE"]:
            rr = _st.spearmanr([F(r[c]) for r in tfm_rows], [F(r["accuracy"]) for r in tfm_rows])
            if rr.pvalue < 0.05:
                n_follow += 1
        add("nVovkFollowAcc", {9: "nine", 10: "ten", 8: "eight", 7: "seven"}.get(n_follow, str(n_follow)), "data/P20_vovk_rho.csv: criteria with Spearman p < 0.05 against accuracy within the 13 TFMs")
    else:
        missing("nVovkFollowAcc", "data/P20_vovk_rho.csv")
    mm = {r["model"]: r for r in models}
    if "tabpfn25" in mm and "tabpfn25b" in mm:
        add("nPairAucDelta", plain(abs(F(mm["tabpfn25"]["auc"]) - F(mm["tabpfn25b"]["auc"])), 3), "data/models_F1.csv |auc(tabpfn25) - auc(tabpfn25b)|")
        add("nPairCommitDelta", plain(abs(F(mm["tabpfn25"]["commit_pt"]) - F(mm["tabpfn25b"]["commit_pt"])), 4), "data/models_F1.csv |commit_pt(tabpfn25) - commit_pt(tabpfn25b)|")
    sw = rows("P01_alpha_sweep_family.csv") or []
    for r in sw:
        if r.get("block") == "F1" and r["family"] in ("TFM", "GBDT"):
            tag = {"0.05": "Five", "0.2": "Twenty", "0.20": "Twenty"}.get(r["alpha"])
            if tag:
                add(f"nPassShare{r['family']}{tag}", pct(r["pass_lf"]), f"data/P01_alpha_sweep_family.csv F1.{r['alpha']}.{r['family']}.pass_lf")
    if not sw:
        for n in ["nPassShareTFMFive", "nPassShareTFMTwenty", "nPassShareGBDTFive", "nPassShareGBDTTwenty"]:
            missing(n, "data/P01_alpha_sweep_family.csv")

    # ---- N1 rank picture --------------------------------------------------------------------
    cd = rows("N1_cd.csv")
    if cd:
        for r in cd:
            t = {"auc": "Auc", "commit_pt": "Commit", "sscsp": "Conf", "sscs": "Sscs"}.get(r["metric"])
            if t:
                add(f"nCdPairs{t}", r["n_pairs_separated"], f"data/N1_cd.csv {r['metric']}.n_pairs_separated")
        add("nCdValue", f"{F(cd[0]['critical_difference']):.2f}", "data/N1_cd.csv critical_difference")
    else:
        for n in ["nCdPairsAuc", "nCdPairsCommit", "nCdPairsConf", "nCdPairsSscs", "nCdValue"]:
            missing(n, "data/N1_cd.csv")

    # ---- N2a levels and matched level -------------------------------------------------------
    lv = rows("N2_eps_contrast_by_level.csv")
    if lv:
        def pick(score, alpha):
            return next((x for x in lv if x["score"] == score and abs(F(x["alpha"]) - alpha) < 1e-9 and x["contrast"] == "TFM-GBDT"), None)
        for score, tag in (("lac", "Lac"), ("apsr", "Apsr")):
            lo_, hi_ = pick(score, 0.01), pick(score, 0.20)
            if lo_: add(f"nEpsGap{tag}Low", signed(lo_["mean"]), f"data/N2_eps_contrast_by_level.csv {score}.0.01.TFM-GBDT.mean")
            if hi_: add(f"nEpsGap{tag}High", signed(hi_["mean"]), f"data/N2_eps_contrast_by_level.csv {score}.0.20.TFM-GBDT.mean")
        ten = pick("lac", 0.10)
        if ten:
            add("nEffNLacTen", ten["n_effective"], "data/N2_eps_contrast_by_level.csv lac.0.10.n_effective")
            add("nTiesLacTen", str(int(F(ten["n"])) - int(F(ten["n_effective"]))), "idem n - n_effective")
    else:
        for n in ["nEpsGapLacLow", "nEpsGapLacHigh", "nEpsGapApsrLow", "nEpsGapApsrHigh", "nEffNLacTen", "nTiesLacTen"]:
            missing(n, "data/N2_eps_contrast_by_level.csv")
    ml = rows("N2_matched_level.csv")
    r = next((x for x in (ml or []) if x["score"] == "lac" and abs(F(x["c"]) - 1.0) < 1e-9 and x["contrast"] == "TFM-GBDT"), None)
    if r:
        add("nMatchedGap", signed(r["mean"]), "data/N2_matched_level.csv lac.c=1.TFM-GBDT.mean"); add("nMatchedGapP", pval(r["wilcoxon_p"]), "idem wilcoxon_p")
        add("nMatchedCovTFM", f"{F(r['delivered_cov_TFM']):.3f}", "idem delivered_cov_TFM"); add("nMatchedCovGBDT", f"{F(r['delivered_cov_GBDT']):.3f}", "idem delivered_cov_GBDT")
    else:
        for n in ["nMatchedGap", "nMatchedGapP", "nMatchedCovTFM", "nMatchedCovGBDT"]:
            missing(n, "data/N2_matched_level.csv")

    # ---- N2b forecast, two-class, controls --------------------------------------------------
    fc = rows("N2_forecast.csv")
    if fc:
        r = fc[0]
        add("nForecastRho", signed(r["rho_spearman"], 3), "data/N2_forecast.csv rho_spearman"); add("nForecastN", thousands(r["n_pairs"]), "idem n_pairs")
        add("nEpsBelow", plain(r["eps_below"]), "idem eps_below"); add("nNBelow", thousands(r["n_below"]), "idem n_below")
        add("nEpsAbove", plain(r["eps_above"]), "idem eps_above"); add("nNAbove", thousands(r["n_above"]), "idem n_above")
    else:
        for n in ["nForecastRho", "nForecastN", "nEpsBelow", "nNBelow", "nEpsAbove", "nNAbove"]:
            missing(n, "data/N2_forecast.csv")
    be = rows("N2_binary_exclusivity.csv")
    if be:
        r = be[0]
        add("nBinRuns", thousands(r["n_binary_runs"]), "data/N2_binary_exclusivity.csv n_binary_runs"); add("nBinBoth", r["n_both_empty_and_full"], "idem n_both_empty_and_full")
        add("nOnsetShare", pct(r["share_onset_within_2_over_ncal"]), "idem share_onset_within_2_over_ncal"); add("nOnsetMedianGap", signed(r["median_onset_gap"]), "idem median_onset_gap")
    else:
        for n in ["nBinRuns", "nBinBoth", "nOnsetShare", "nOnsetMedianGap"]:
            missing(n, "data/N2_binary_exclusivity.csv")
    cc = rows("N2_confidence_controls.csv") or []
    def ctl(name):
        return next((x for x in cc if x["control"] == name and x["contrast"] == "TFM-GBDT"), None)
    for name, tag in (("sscsp_never_abstaining", "NAConf"), ("selective_error_budget_0.02", "SelErrTwo"), ("selective_error_budget_0.05", "SelErrFive"),
                      ("selective_error_budget_0.10", "SelErrTen"), ("aurc", "Aurc"), ("brier_reliability", "BrierRel")):
        r = ctl(name)
        if r:
            add(f"n{tag}Gap", signed(r["mean"]), f"data/N2_confidence_controls.csv {name}.TFM-GBDT.mean")
            add(f"n{tag}GapWLT", wlt(r), "idem pos/neg/zero"); add(f"n{tag}GapP", pval(r["wilcoxon_p"]), "idem wilcoxon_p")
        else:
            missing(f"n{tag}Gap", f"data/N2_confidence_controls.csv {name}")
    # the prose names the three budgets without the word Gap
    for tag in ("SelErrTwo", "SelErrFive", "SelErrTen"):
        LINES.append(f"\\newcommand{{\\n{tag}}}{{\\n{tag}Gap}}  % alias")
    nab = rows("N2_never_abstaining.csv")
    if nab:
        add("nNeverAbstainN", nab[0]["n_datasets_never_abstaining"], "data/N2_never_abstaining.csv n_datasets_never_abstaining")
        add("nNeverAbstainUnits", nab[0]["n_strict_units"], "idem n_strict_units")
    else:
        missing("nNeverAbstainN", "data/N2_never_abstaining.csv"); missing("nNeverAbstainUnits", "idem")
    mb = rows("N2_multiclass_binary.csv") or []
    for task, tt in (("binary", "Bin"), ("multiclass", "Mc")):
        for metric, mt in (("commit_pt", "Commit"), ("sscsp", "Conf")):
            r = next((x for x in mb if x["task"] == task and x["metric"] == metric and x["contrast"] == "TFM-GBDT"), None)
            if r:
                add(f"n{tt}{mt}Gap", signed(r["mean"]), f"data/N2_multiclass_binary.csv {task}.{metric}.TFM-GBDT.mean")
                add(f"n{tt}{mt}GapP", pval(r["wilcoxon_p"]), "idem wilcoxon_p")
            else:
                missing(f"n{tt}{mt}Gap", f"data/N2_multiclass_binary.csv {task}.{metric}")

    # ---- N2c arms and price -----------------------------------------------------------------
    arms = rows("N2_arms.csv") or []
    armtag = {"lac": "Lac", "apsr": "Apsr", "apsd": "Apsd", "lac_top1": "Top", "lac_top1_retuned": "Retuned"}
    for r in arms:
        if r["family"] == "all" and r["arm"] in armtag:
            t = armtag[r["arm"]]; s = f"data/N2_arms.csv {r['arm']}.all"
            add(f"nArmCov{t}", f"{F(r['delivered_cov']):.3f}", s + ".delivered_cov"); add(f"nArmWidth{t}", f"{F(r['width_fb']):.2f}", s + ".width_fb")
            add(f"nArmEps{t}", f"{F(r['eps']):.3f}", s + ".eps")
            if r["arm"] == "apsd":
                add("nArmApsdExcess", f"{100 * (F(r['delivered_cov']) - 0.9):.1f}", s + ".delivered_cov - 0.90, points")
    if not arms:
        for t in armtag.values():
            missing(f"nArmCov{t}", "data/N2_arms.csv"); missing(f"nArmWidth{t}", "data/N2_arms.csv")
        missing("nArmApsdExcess", "data/N2_arms.csv")
    rt = rows("N2_arms_retune.csv") or []
    r = next((x for x in rt if x.get("contrast") == "TFM-GBDT"), rt[0] if rt else None)
    if r:
        add("nRetuneRuns", thousands(r["n_runs"]), "data/N2_arms_retune.csv n_runs"); add("nRetuneReachable", thousands(r["n_reachable"]), "idem n_reachable")
        add("nRetuneUnreachable", thousands(r["n_unreachable"]), "idem n_unreachable"); add("nRetuneAlphaMean", f"{F(r['alpha_star_mean']):.3f}", "idem alpha_star_mean")
        add("nRetuneAlphaMedian", f"{F(r['alpha_star_median']):.3f}", "idem alpha_star_median")
        add("nRetuneWidthGap", signed(r["retune_width_mean"]), "idem retune_width_mean (TFM-GBDT)"); add("nRetuneAlphaGap", signed(r["retune_alpha_mean"]), "idem retune_alpha_mean (TFM-GBDT)")
    else:
        for n in ["nRetuneRuns", "nRetuneReachable", "nRetuneUnreachable", "nRetuneAlphaMean", "nRetuneAlphaMedian"]:
            missing(n, "data/N2_arms_retune.csv")
    pr = rows("N2_price.csv")
    if pr:
        r = pr[0]
        add("nPriceFired", thousands(r["n_runs_fired"]), "data/N2_price.csv n_runs_fired"); add("nPriceFillAcc", f"{F(r['fill_accuracy']):.3f}", "idem fill_accuracy")
        add("nPriceMajority", f"{F(r['majority_share']):.3f}", "idem majority_share"); add("nBreakEven", f"{F(r['breakeven_price']):.2f}", "idem breakeven_price")
        add("nBreakEvenPct", f"{100 * F(r['breakeven_price']):.0f}", "idem breakeven_price x 100"); add("nTopOneCov", f"{F(r['delivered_cov_top1']):.3f}", "idem delivered_cov_top1")
    else:
        for n in ["nPriceFired", "nPriceFillAcc", "nPriceMajority", "nBreakEven", "nBreakEvenPct", "nTopOneCov"]:
            missing(n, "data/N2_price.csv")

    # ---- N3 large-dataset and synthetic blocks ----------------------------------------------
    for pref, fname, tagn in (("D", "N3_D29_contrasts.csv", "D29"), ("C", "N3_C39_contrasts.csv", "C39")):
        c3 = rows(fname) or []
        for r in c3:
            if r.get("contrast") != "TFM-GBDT" or r.get("metric") not in tag_of:
                continue
            t = tag_of[r["metric"]]; s = f"data/{fname} TFM-GBDT.{r['metric']}"
            add(f"n{pref}Gap{t}", signed(r["mean"]), s + ".mean"); add(f"n{pref}Gap{t}CI", ci(r["lo"], r["hi"]), s + ".lo/hi")
            add(f"n{pref}Gap{t}P", pval(r["wilcoxon_p"]), s + ".wilcoxon_p"); add(f"n{pref}Gap{t}WLT", wlt(r), s + ".pos/neg/zero")
        if c3:
            add(f"n{pref}NDatasets", next(x["n"] for x in c3 if x["contrast"] == "TFM-GBDT"), f"data/{fname} column n")
        else:
            for t in ("Auc", "Sscs", "Commit", "Conf"):
                missing(f"n{pref}Gap{t}", f"data/{fname}")
            missing(f"n{pref}NDatasets", f"data/{fname}")
    dr = rows("N3_D29_rho.csv") or []
    for r in dr:
        t = {"commit_pt": "Commit", "sscsp": "Conf", "sscs": "Sscs"}.get(r.get("metric"))
        if t and r.get("block", "D29") == "D29":
            add(f"nDRho{t}", signed(r["rho_spearman"], 2), f"data/N3_D29_rho.csv {r['metric']}.rho_spearman")
    if not dr:
        missing("nDRhoCommit", "data/N3_D29_rho.csv"); missing("nDRhoConf", "data/N3_D29_rho.csv")
    dm = rows("N3_D29_models.csv") or []
    add("nDNModels", str(len(dm)) if dm else "\\textcolor{red}{??}", "data/N3_D29_models.csv row count")
    if not dm:
        MISSING.append("nDNModels <- data/N3_D29_models.csv")
    add("nCNDatasets", "\\nCNDatasets", "alias defined above from N3_C39_contrasts.csv") if False else None
    dl = {x["family"]: x for x in rows("N3_D29_labelfree.csv") or []}
    for fam, tag in (("TFM", "TFM"), ("GBDT", "GBDT")):
        if fam in dl:
            add(f"nDPassShare{tag}", pct(dl[fam]["pass_lf_plain"], 0), f"data/N3_D29_labelfree.csv {fam}.pass_lf_plain (percent)")
            add(f"nDTransdCovErr{tag}", plain(dl[fam]["lftr_coverr"]), f"data/N3_D29_labelfree.csv {fam}.lftr_coverr")
            add(f"nDPassShareConf{tag}", pct(dl[fam]["pass_lac_plain"], 0), f"data/N3_D29_labelfree.csv {fam}.pass_lac_plain (percent)")
        else:
            missing(f"nDPassShare{tag}", "data/N3_D29_labelfree.csv")

    # ---- G: the 31 August 15-model grid -----------------------------------------------------
    gi = rows("G_interventions.csv") or []
    ens = [x for x in gi if x["test"] == "ensemble 1 to 32"]
    if ens:
        for r in ens:
            t = {"PFN-v2": "PFN", "tabicl": "ICL"}[r["model"]]
            add(f"nGEns{t}", signed(r["value"]), f"data/G_interventions.csv ensemble.{r['model']}.value"); add(f"nGEns{t}P", pval(r["p"]), "idem p")
        add("nGEnsMaxDelta", plain(max(abs(F(r["value"])) for r in ens)), "data/G_interventions.csv ensemble, max |value|")
        add("nGEnsMinP", pval(min(F(r["p"]) for r in ens)), "data/G_interventions.csv ensemble, min p")
    tmpc = next((x for x in gi if x["statistic"].startswith("two-class cells with bitwise identical")), None)
    if tmpc:
        add("nGTempIdentical", thousands(F(tmpc["value"])), "data/G_interventions.csv temperature identical cells"); add("nGTempCells", thousands(F(tmpc["n"])), "idem n")
    llb = next((x for x in gi if "accuracy-tuned" in x["statistic"]), None); lla = next((x for x in gi if "log-loss-tuned" in x["statistic"]), None)
    if llb and lla:
        add("nGLLGapBefore", signed(llb["value"]), "data/G_interventions.csv log-loss tuning, accuracy-tuned gap"); add("nGLLGap", signed(lla["value"]), "idem log-loss-tuned gap")
        add("nGLLDatasets", llb["n"], "idem n")
    if not gi:
        for n in ["nGEnsPFN", "nGEnsPFNP", "nGEnsICL", "nGEnsICLP", "nGEnsMaxDelta", "nGEnsMinP", "nGTempIdentical", "nGTempCells", "nGLLGapBefore", "nGLLGap"]:
            missing(n, "data/G_interventions.csv")
    gt = {x["stat"]: x for x in rows("G_taus.csv") or []}
    for k, n in (("nCJTauAuc", "nGTauAuc"), ("nCJTauSscs", "nGTauSscs"), ("nCJMaxAbsDeltaAuc", "nGMaxDeltaAuc"), ("nCJMaxAbsDeltaSscs", "nGMaxDeltaSscs")):
        if k in gt:
            add(n, gt[k]["value"], f"data/G_taus.csv {k}")
        else:
            missing(n, "data/G_taus.csv")
    return "\n".join(LINES) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); a = ap.parse_args(argv)
    text = build()
    if MISSING:
        print("MISSING macros (printed as red ??):", file=sys.stderr)
        for m in MISSING:
            print("   ", m, file=sys.stderr)
    if a.check:
        cur = OUT.read_text() if OUT.exists() else ""
        if cur != text:
            print(f"STALE: {OUT}", file=sys.stderr); return 1
        if MISSING:
            return 1
        print(f"fresh: {OUT}"); return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text); print(f"wrote {OUT} ({text.count(chr(10))} lines, {len(MISSING)} missing)"); return 0


if __name__ == "__main__":
    sys.exit(main())
