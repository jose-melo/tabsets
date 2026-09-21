# data/ — what is here, and what is not

**Frozen tables, copied on 2026-09-20.** These are the small aggregates the paper prints
from. `scripts/make_numbers.py` reads them and writes `paper/numbers.tex`; nothing in the
prose types a number by hand.

| file | rows | what | produced by |
|---|---|---|---|
| `models_F1.csv` | 22 | per model on F1: AUC, ε, commitment, SSCS, SSCS⁺, coverage, width | `output/2026-09-19_hplr-commitment-figure/` |
| `families_F1.csv` | 4 | the same, as family means of model means | idem |
| `family_contrast_F1.csv` | 15 | paired family contrasts with interval, Wilcoxon p, W/L/T, n | idem |
| `aggregation_check_F1.csv` | — | the four aggregations: which contrasts move and which do not | idem |
| `robustness_F1.csv` | — | α = 0.05 / 0.10 / 0.20 and randomised APS | idem |
| `zscores_TFM_GBDT_F1.csv` | — | standardised contrasts | idem |
| `P01_model_table_F1_a10.csv` | 22 | label-free threshold per model: coverage error, pass share, width | `output/2026-09-19_tabsets-uq-analysis/` (run `wf_de4c6abb-090`) |
| `P01_family_table_F1_a10.csv` | 4 | idem, by family, plus split conformal's own pass share | idem |
| `P01_contrasts_F1_a10.csv` | — | the paired contrasts of P01 | idem |
| `P02_model_table_F1_a10.csv` | 22 | the constant threshold 1 − α and its delivered coverage | idem |
| `P01_rho_F1_a10.csv` | 6 | Spearman ρ of each label-free statistic with accuracy: 22 models, within the 13 TFMs, within the 9 others (Figure 6a) | idem, copied 2026-09-20 |
| `P02_family_table_F1_a10.csv` | 4 | the constant threshold by family: delivered coverage, its error, the own-threshold error, the deterministic companion (Figure 7a band) | idem, copied 2026-09-20 |
| `P02_transfer_by_type.csv` | 8 | threshold-transfer error by pair type and score at matched calibration accuracy: median, cluster-bootstrap interval, n pairs, n datasets (Figure 7b) | `analysis/fig_F7.py`, from `results/P02_pairs_typed.csv` in the campaign folder |
| `P03_*_F1_a10.csv` | — | the contained-mass residual R by model and family, and its ρ | idem |
| `P16_certificate.csv`, `P16_certificate_strict.csv` | — | the level certificate, 45,263 cells | idem |
| `P16_width_table.csv` | — | set width raw and under the fallback accounting | idem |
| `N2_forecast.csv` | 1 | N2b: Spearman ρ of ε against calibration accuracy over the (dataset, model) pairs of F1, and the mean ε below / above acc_cal ≤ 1 − α (LAC 0.10) | `analysis/n2b_forecast.py` over `analysis/cache/n2b_cells_F1.csv` (`analysis/n2b_cells.py`, 2026-09-20) |
| `N2_ramp_bins.csv` | 96 | N2b: ε binned on calibration accuracy (width 0.025), one row per (family, bin), for Figure 5a | idem |
| `N2_forecast_points.csv` | 2,486 | N2b: the (dataset, model) pairs behind the forecast (acc_cal, ε), for Figure 5b | idem |
| `N2_binary_exclusivity.csv` | 1 | N2b: the 16,280 binary runs by {both, empty only, two-label only, neither} under LAC 0.10, and the onset level (smallest α that empties the run, exact for MAPIE's quantile) against the calibration error rate | idem |
| `N2_confidence_controls.csv` | 18 | N2b: paired TFM−GBDT / −deep / −classic contrasts of SSCS⁺ on the never-abstaining datasets, selective error at abstention budgets 0.02 / 0.05 / 0.10 (lowest-max-probability points abstained; `hplr.metrics.risk_coverage_curve`), AURC (`hplr.metrics.aurc`) and the Brier reliability component (`hplr.metrics.brier_decomposition`); the last three are probability-only and carry the grid's score / level columns by schema, not by dependence | `analysis/n2b_controls.py` over the same cache |
| `N2_never_abstaining.csv` | 1 | N2b: the never-abstaining datasets of F1 at LAC 0.10 (`L.empty_free_datasets`) and their strict-unit count (`L.units("strict")`) | idem |
| `N2_multiclass_binary.csv` | 24 | N2b: commitment, SSCS⁺, SSCS and AUC contrasts on the 74 binary and the 39 multiclass datasets of F1 separately, never pooled | idem |

**N2b and the frozen tables.** `analysis/n2b_cells.py --verify` checks the per-run cache against `families_F1.csv` and the commitment figure's `runs_F1.csv`: the identity SSCS = SSCS⁺ × 𝟙[no empty set] holds on 24,860 of 24,860 runs, and the family means agree to 1e-6 on 22 of the 28 (family, column) cells. The six that do not (largest: deep-family SSCS, 1.4e-3) come from **42 runs measured twice** (two cache files for one (dataset, model, seed): `Firm-Teacher_Clave-Direction_Classification`, `pol`, `jm1`, `eucalyptus`), where the campaign manifest kept one copy and the 09-19 tables the other; substituting the frozen copies for those runs reproduces `families_F1.csv` to 4e-16. N2b uses the manifest's block (`L.block(L.manifest(), "F1")`) as the contract says; the report is `analysis/cache/n2b_selfcheck_F1.txt`.
| `N2_eps_by_level.csv` | 48 | family means of ε at six levels under LAC and fully randomised APS (block, score, alpha, family) | `analysis/n2a_levels.py` (shared pass `analysis/n2_common.py`, cell cache `analysis/cache/N2_cells_F1.csv`, 155 s on 3 workers) |
| `N2_eps_contrast_by_level.csv` | 36 | paired TFM−GBDT / TFM−deep / TFM−classic ε contrasts at each level and score, with n_effective (datasets with a non-zero paired difference) | `analysis/n2a_levels.py` |
| `N2_matched_level.csv` | 18 | the ε contrasts with the level matched to c × the calibration error rate (c = 0.5, 1, 2; clipped to [1/n_cal, 0.5]), both scores, with the delivered coverage of TFM and GBDT | `analysis/n2a_levels.py` |
| `N2_arms.csv` | 25 | the five arms at α = 0.10 by family and `all` (= mean of the four family means): delivered coverage, ε as delivered (0 for the never-empty arms), width, width_fb (empty set counted as size 1), SSCS, SSCS⁺, n_runs; `lac_top1_retuned` on the reachable runs only | `analysis/n2c_arms.py` |
| `N2_arms_retune.csv` | 3 | the re-tune of LAC + top-1: reachable/unreachable runs (reachable = test accuracy below 0.90 and a level in [1/n_cal, 1 − 1/n_cal] that brings the fallback arm to ≤ 0.90), α* mean/median over reachable runs, width_fb at α*, and the paired α* and width contrasts over the datasets where both families have reachable runs | `analysis/n2c_arms.py` |
| `N2_price.csv` | 1 | the top-1 default where it fires: runs with an empty set, pooled accuracy of the filled label, pooled share of those points whose true label is the calibration majority class, break-even price 1 − fill accuracy, delivered coverage of `lac_top1` (all) | `analysis/n2c_arms.py` |
| `decomposition_F1.csv` | 23 | per model: the exact product mean SSCS = P(no empty set) x mean SSCS+ over those runs; zero shares; SUMMARY row with the log-variance shares | analysis/derived.py from the commitment-figure cells table |
| `perdataset_F1.csv` | 113 | per dataset: family means (TFM, GBDT, deep, classic) of AUC, commitment, SSCS+ at LAC 0.10, with n, K, p | analysis/derived.py from the commitment-figure cells table and the campaign covariates |
| `sizes_F1.csv` | 2 | calibration and test sizes over the (dataset, seed) splits of the block | analysis/derived.py from the campaign manifest |
| `P01_temperature_F1_a10.csv` | 6 | label-free coverage error raw and after a held-out global temperature, by family and for LogReg / kNN | analysis/derived.py from results/P01_pass1_cells (campaign) |
| `P01_temperature_contrasts_F1_a10.csv` | 4 | paired TFM - GBDT contrasts of the label-free coverage error, raw and after temperature | analysis/derived.py |
| `P02_pit_family_F1_a10.csv` | 4 | family means of the share of runs whose randomised-APS score passes the uniformity test, raw and after temperature | analysis/derived.py from results/P02_pit_model_table |
| `P02_pit_model_table_F1_a10.csv` | 22 | per model: uniformity (PIT) test of the randomised-APS score, raw and after temperature | copied from the campaign results/ |
| `P17_se_ratios.csv` | 30 | seed vs point standard errors: the inference layer (P17) | copied from the campaign results/ (run wf_de4c6abb-090) |
| `P17_mde_card.csv` | 19 | minimum detectable effect of the family contrast on strict units (P17) | copied from the campaign results/ (run wf_de4c6abb-090) |
| `P01_alpha_sweep_family.csv` | 36 | label-free pass shares and errors by family at alpha 0.05 / 0.10 / 0.20 (P01) | copied from the campaign results/ (run wf_de4c6abb-090) |
| `P01_alpha_sweep_named_models.csv` | 24 | the same for LogReg, kNN, TabPFN v1, TabM (P01) | copied from the campaign results/ (run wf_de4c6abb-090) |
| `P02_rho_F1_a10.csv` | 4 | the same for the constant-threshold statistics (P02) | copied from the campaign results/ (run wf_de4c6abb-090) |
| `P08_soup_contrast.csv` | 7 | alignment of low confidence with cross-family disagreement: TFM family vs a seven-model tuned soup (P08 addendum) | transcribed from FINDINGS-P08.md section 6 addendum table (run wf_de4c6abb-090); primary from its section 3 clause 1 |
| `G_interventions.csv` | 9 | 15-model grid: ensemble sweep, temperature, log-loss tuning of the boosters | analysis/g_interventions.py from ~/hplr-neurocomputing/analysis/journal/numbers_tier1.json and the @lltune cells of the journal cache |
| `A_conference.csv` | 11 | the conference table as published, its reconstruction on the 15-model grid, and the TabSets main block | analysis/g_interventions.py from V1 appA-conference.tex, V1 numbers.tex and data/models_F1.csv |
| `G_taus.csv` | 6 | Kendall taus and max deltas, conference ordering vs the 15-model grid | analysis/g_interventions.py from V1 numbers.tex (numbers_tier1.json E0) |
| `P01_neverabstaining_F1_a10.csv` | 2 | label-free coverage error contrast on the datasets where no model abstains under LAC 0.10 (P01 control) | analysis/derived.py from results/P01_pass1_cells and the commitment-figure cells table |
| `P16_apsrand_models_all.csv` | 22 | level residual of the fully randomised APS score per model, all cells (P16) | copied from the campaign results/ (run wf_de4c6abb-090) |
| `N1_cd.csv` | 4 | Friedman + Nemenyi at k = 22 on the 113 real datasets of F1, LAC α = 0.10: critical difference and pairs separated of 231, for auc, commit_pt, sscsp, sscs (higher is better for all four). Q table extended past k = 15 (hplr's returns NaN there), verified against scipy | `analysis/n1_ranks.py --block F1` (cache pass of 2026-09-20; shared code `analysis/tabsets_cells.py`) |
| `N1_ranks.csv` | 88 | mean rank per model and metric, with family | idem |
| `N1_friedman.csv` | 4 | Friedman χ² and p, Holm-Wilcoxon and scikit-posthocs Nemenyi pair counts (the latter reproduces N1_cd exactly), family mean ranks, q_0.05(22) | idem |
| `N3_D29_contrasts.csv` | 18 | paired TFM−GBDT / TFM−deep / TFM−classic on the 29 large datasets (21 models, no exaone) for auc, sscs, commit_pt, sscsp, cov, width; `lib_tabsets.family_contrast` | `analysis/n3_blocks.py --block D29` |
| `N3_D29_models.csv` | 21 | per model on D29, the columns of `models_F1.csv`, same aggregation and bootstrap seed | idem |
| `N3_D29_families.csv` | 4 | family means of model means on D29, the columns of `families_F1.csv` | idem |
| `N3_D29_rho.csv` | 4 | Spearman over model means of AUC with commit_pt, sscsp, sscs, commit_run on D29 | idem |
| `N3_D29_labelfree.csv` | 4 | label-free threshold on D29 by family: coverage error, pass share, width | **copied** by `n3_blocks.py` from `output/2026-09-19_tabsets-uq-analysis/results/P01_family_table_D29_a10.csv` (run `wf_de4c6abb-090`), not recomputed |
| `N3_C39_contrasts.csv` | 18 | the synthetic block (22 models × 39 datasets), same schema as D29; Appendix G only | `analysis/n3_blocks.py --block C39` |
| `N3_C39_families.csv` | 4 | idem | idem |
| `N3_C39_models.csv` | 22 | idem (not in the contract's list; written for Appendix G) | idem |
| `N3_C39_rho.csv` | 4 | idem | idem |
| `strict_contrasts_F1.csv` | 8 | TFM - GBDT contrasts of AUC, SSCS, commitment, SSCS+ on datasets and on strict source units | analysis/derived.py via lib_tabsets.family_contrast(collapse='strict') |
| `P01_DK10_by_cell.csv` | 35 | many-class block: transductive label-free coverage error per dataset and model (P01) | copied from the campaign results/ (run wf_de4c6abb-090) |
| `models.csv` | 22 | the model set: package version, checkpoint, tuning, context cap, softmax temperature, class ceiling (hand-maintained from DATA-REPORT-0919.md S2, V1 Table 3, NEW-MODELS.md, CLASS-CEILING-0912.md) | hand-maintained; n.r. = not recorded in the run logs |
| `P20_vovk_rho.csv` | 22 | the ten efficiency criteria of Vovk et al. 2016 per model on the main block, with accuracy (P20) | copied from the campaign results/ (run wf_de4c6abb-090) |
| `prereg.csv` | 20 | the analysis plan: each analysis specified before the cache was read, its outcome, where it enters the article, and `outcome_class` | derived from the article's appendix table; the class is `rejected` if the outcome contains "did not hold", `held` if it reads exactly "held", `qualified` if it begins with "held", `undecided` otherwise |

### Which copy of a twice-computed cell

Some cells exist twice in the cache (a re-run into another directory; DATA-REPORT-0919 §5 puts the
difference inside the training noise floor). The N1 and N3 passes read the copy named by the
2026-09-19 campaign manifest, like every other pass in `analysis/`; the frozen `*_F1.csv` tables
above were built from the machines' 0918 exports, which broke the same tie by file order and chose
the other copy on **42 of 24,860 F1 cells** (4 datasets; ftt, mlp, resnet, tabiclv2, tabpfn3).
`python3 analysis/n3_blocks.py --block F1 --check --inventory exports` reproduces
`families_F1.csv` and `models_F1.csv` to 1e-15 from the cache; with the manifest copy the family
means move by at most 1.4e-3 (the deep family only) and the N1 counts move only for SSCS⁺
(134 → 136 of 231). On D29 only deep models are duplicated, so its TFM−GBDT rows are identical under
either copy; TFM−deep moves by at most 6e-4. The comparison tables are in
`analysis/cache/sensitivity/` (`--inventory exports|alt`); the choice of the canonical copy is Zé's.

## What is deliberately not here

The cache, the library and the 568 result files of the campaign live in
`~/mugango/output/2026-09-19_tabsets-uq-analysis/` (**1.1 GB**) and
`~/mugango/output/2026-09-19_hplr-commitment-figure/data/` (cells and runs tables, 11 MB).
Two files the paper needs but that are too large to copy:

- `results/P02_pairs_typed.csv` — 27 MB, threshold transfer, 156,618 units (Figure 7b)
- `results/P16_level_cells.csv` — 44 MB, the per-cell level residuals (Appendix E)

A manuscript repository is not a data repository. **The release itself** is contribution 1
and needs a persistent identifier (NEEDS N9); a GitHub URL alone is not one.

## The rule

A number reaches the paper only if it can be recomputed from a file named in this table or
from a pass recorded in `plan/NEEDS.md`. Add a file here when you add it to `data/`, with
the folder that produced it. A CSV with no provenance row is not usable.
