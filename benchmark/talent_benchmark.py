import argparse
import json
import logging
import os
import re
import socket
import subprocess
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model.utils import HPO_OBJECTIVES
from utils import (
    N_ESTIMATORS_MODELS,
    NEW_TFM_MODELS,
    SOFTMAX_TEMPERATURE_MODELS,
    evaluate_talent_dataset,
    set_seed,
)


def _configure_logging(log_path: Path) -> logging.Logger:
    """Logger that streams to stdout and writes ONE file per job."""
    logger = logging.getLogger("talent_benchmark")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def _append_to_csv(record: dict, csv_path: Path) -> None:
    """Append one record to a CSV owned by this job, creating headers if needed."""
    import pandas as pd

    df = pd.DataFrame([record])
    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)


def _git_sha(repo: Path) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _default_job_tag() -> str:
    aid, tid = os.environ.get("SLURM_ARRAY_JOB_ID"), os.environ.get("SLURM_ARRAY_TASK_ID")
    if aid and tid:
        return f"{aid}_{tid}"
    jid = os.environ.get("SLURM_JOB_ID")
    if jid:
        return jid
    return f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"


def _default_env_tag() -> str:
    return os.environ.get("CONDA_DEFAULT_ENV") or Path(
        os.environ.get("VIRTUAL_ENV", sys.prefix)
    ).name


def seeds_for_datasets(datasets, seeds, master_seed, n_seeds):
    """Per-dataset seed lists.

    ``--seeds`` given: every dataset gets exactly those values. Otherwise the
    legacy derivation is reproduced *in the canonical ``default_datasets``
    order* (``--seed 42 --n-seeds 10`` yields the published seeds for every
    dataset regardless of the ``--datasets`` order or ``--shard``); a dataset
    outside ``default_datasets`` is seeded from ``(master_seed, crc32(name))``.
    """
    if seeds:
        return {name: [int(s) for s in seeds] for name in datasets}
    rng = np.random.default_rng(master_seed)
    by_name = {}
    for name in default_datasets:
        by_name[name] = [
            int(s)
            for s in rng.integers(
                low=0, high=np.iinfo(np.int32).max, size=n_seeds, dtype=np.int64
            )
        ]
    out = {}
    for name in datasets:
        if name not in by_name:
            r = np.random.default_rng([master_seed, zlib.crc32(name.encode())])
            by_name[name] = [
                int(s)
                for s in r.integers(
                    low=0, high=np.iinfo(np.int32).max, size=n_seeds, dtype=np.int64
                )
            ]
        out[name] = by_name[name]
    return out


default_datasets = [
    "BLE_RSSI_dataset_for_Indoor_localization",
    "Bank_Customer_Churn_Dataset",
    "Basketball_c",
    "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.0GHz(Urbinati)",
    "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-10.5GHz(Urbinati)",
    "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-11.0GHz(Urbinati)",
    "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.0GHz(Urbinati)",
    "Contaminant-detection-in-packaged-cocoa-hazelnut-spread-jars-using-Microwaves-Sensing-and-Machine-Learning-9.5GHz(Urbinati)",
    "Customer_Personality_Analysis",
    "Diabetic_Retinopathy_Debrecen",
    "Employee",
    "FICO-HELOC-cleaned",
    "FOREX_audcad-day-High",
    "FOREX_audchf-day-High",
    "FOREX_audjpy-day-High",
    "FOREX_cadjpy-day-High",
    "Fitness_Club_c",
    "GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1",
    "GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001",
    "Gender_Gap_in_Spanish_WP",
    "GesturePhaseSegmentationProcessed",
    "Heart-Disease-Dataset-(Comprehensive)",
    "Is-this-a-good-customer",
    "JapaneseVowels",
    "KDD",
    "KDDCup09_upselling",
    "Long",
    "Marketing_Campaign",
    "Mobile_Price_Classification",
    "National_Health_and_Nutrition_Health_Survey",
    "Performance-Prediction",
    "PieChart3",
    "Pima_Indians_Diabetes_Database",
    "PizzaCutter3",
    "Pumpkin_Seeds",
    "QSAR_biodegradation",
    "Telecom_Churn_Dataset",
    "VulNoneVul",
    "Water_Quality_and_Potability",
    "Waterstress",
    "Wilt",
    "abalone",
    "ada",
    "ada_agnostic",
    "ada_prior",
    "airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True",
    "allbp",
    "allrep",
    "analcatdata_authorship",
    "autoUniv-au7-1100",
    "banknote_authentication",
    "baseball",
    "car-evaluation",
    "churn",
    "cmc",
    "company_bankruptcy_prediction",
    "contraceptive_method_choice",
    "credit-g",
    "delta_ailerons",
    "dis",
    "drug_consumption",
    "estimation_of_obesity_levels",
    "eye_movements",
    "first-order-theorem-proving",
    "golf_play_dataset_extended",
    "heloc",
    "ibm-employee-performance",
    "kc1",
    "kdd_ipums_la_97-small",
    "kr-vs-kp",
    "maternal_health_risk",
    "mice_protein_expression",
    "national-longitudinal-survey-binary",
    "ozone-level-8hr",
    "ozone_level",
    "page-blocks",
    "pc1",
    "pc3",
    "pc4",
    "phoneme",
    "predict_students_dropout_and_academic_success",
    "rice_cammeo_and_osmancik",
    "ringnorm",
    "rl",
    "satimage",
    "segment",
    "seismic+bumps",
    "shill-bidding",
    "shrutime",
    "spambase",
    "splice",
    "sports_articles_for_objectivity_analysis",
    "statlog",
    "steel_plates_faults",
    "svmguide3",
    "sylvine",
    "taiwanese_bankruptcy_prediction",
    "telco-customer-churn",
    "thyroid",
    "thyroid-ann",
    "thyroid-dis",
    "turiye_student_evaluation",
    "twonorm",
    "vehicle",
    "wall-robot-navigation",
    "water_quality",
    "waveform-5000",
    "waveform_database_generator_version_1",
    "website_phishing",
    "wine",
    "wine-quality-red",
    "wine-quality-white",
]

default_models = [
    "tabpfn",
    "PFN-v2",
    "tabicl",
    "mitra",
    "xgboost",
    "lightgbm",
    "catboost",
    "tabm",
    "ftt",
    "tabr",
    "knn",
    "LogReg",
]


def main():
    parser = argparse.ArgumentParser(description="Run TALENT Benchmark for UQ")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=default_datasets,
        help="List of dataset names from TALENT benchmark.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        help="List of models to evaluate. TALENT names (default list) plus the pip "
        f"foundation models {sorted(NEW_TFM_MODELS)} (tabicl>=2.1 / tabpfn>=8 env).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Ensemble-size override for the in-context models "
        f"{sorted(N_ESTIMATORS_MODELS)} (tabpfn v1: N_ensemble_configurations). "
        "Unset = each constructor's own default (3 / 4 / 32 / 8). Use with --tag.",
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=None,
        help="Softmax-temperature override for "
        f"{sorted(SOFTMAX_TEMPERATURE_MODELS)} (vendor default 0.9). Use with --tag.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Cache/CSV suffix for sweep runs (e.g. ne8, t1.0): npz cells become "
        "<model>@<tag>, per-job files get __<tag>, rows carry tag=<tag>. Without "
        "it a sweep would collide with (and be skipped as) the base-grid cell.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data",
        help="Path to TALENT datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed for the legacy per-dataset seed derivation (ignored when --seeds is given).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seed values, used for every dataset as given (no derivation).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of derived seeds per dataset when --seeds is not given.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.9,
        help="Target confidence level for prediction sets.",
    )
    parser.add_argument(
        "--conformity-score",
        "--conformity-scores",
        dest="conformity_scores",
        nargs="+",
        default=["lac", "top_k", "aps", "raps"],
        help="Conformity scores computed from the cached probabilities of ONE fit.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="talent-uq-benchmark",
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb-entity", type=str, default=None, help="W&B entity name."
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Never import or initialise W&B (air-gapped compute nodes).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=25, help="Number of Optuna trials for HPO."
    )
    parser.add_argument(
        "--hpo-objective",
        choices=list(HPO_OBJECTIVES),
        default="accuracy",
        help="What the Optuna study maximises for tuned classifiers: 'accuracy' "
        "(TALENT's validation accuracy with a log-loss tie-break; the base grid) or "
        "'logloss' (negative validation log-loss with an accuracy tie-break; E-lltune). "
        "The constant-classifier guard applies to both. Regression always tunes on RMSE. "
        "Anything but 'accuracy' needs --tag (same collision rule as the sweep flags).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Held-out share of the training part used by epoch-trained deep models "
        "for early stopping / best-epoch selection (0 disables it).",
    )
    parser.add_argument(
        "--mock-run",
        action="store_true",
        help=(
            "Run a quick smoke-test over all datasets and models by sampling smaller "
            "subsets and skipping expensive steps like HPO. Outputs go to <out-dir>/mock."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results_v2",
        help="Root for per-job CSVs, logs and the probability cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Probability cache directory (default <out-dir>/cache, or <out-dir>/mock/cache).",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="TALENT scratch dir for this job (checkpoints, trlog, *-tuned.json). "
        "Default <out-dir>/work/<job-tag>. Never share it between concurrent jobs.",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default=None,
        help="Machine tag written into file names and rows (default: short hostname).",
    )
    parser.add_argument(
        "--env-tag",
        type=str,
        default=None,
        help="Environment tag written into rows (default: conda env or venv name).",
    )
    parser.add_argument(
        "--job-tag",
        type=str,
        default=None,
        help="Per-job file-name tag (default: SLURM array/job id, else timestamp_pid).",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default=None,
        help="k/N: run datasets[k::N] of the (ordered) --datasets list.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute cells whose npz already exists (default: skip them = resume).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda | cuda:N | mps",
    )
    args = parser.parse_args()

    if args.tag is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.\-]*", args.tag) or "__" in args.tag:
            parser.error("--tag must match [A-Za-z0-9.-]+ (no '__', '@', '/', spaces)")
    if args.n_estimators is not None:
        if args.n_estimators < 1:
            parser.error("--n-estimators must be >= 1")
        bad = [m for m in args.models if m not in N_ESTIMATORS_MODELS]
        if bad:
            parser.error(f"--n-estimators is not supported by {bad}; supported: {sorted(N_ESTIMATORS_MODELS)}")
    if args.softmax_temperature is not None:
        if args.softmax_temperature <= 0:
            parser.error("--softmax-temperature must be > 0")
        bad = [m for m in args.models if m not in SOFTMAX_TEMPERATURE_MODELS]
        if bad:
            parser.error(f"--softmax-temperature is not supported by {bad}; supported: {sorted(SOFTMAX_TEMPERATURE_MODELS)}")
    if (args.n_estimators is not None or args.softmax_temperature is not None) and not args.tag:
        parser.error("--n-estimators / --softmax-temperature need --tag, otherwise the sweep collides with the base-grid cache cells")
    if args.hpo_objective != "accuracy" and not args.tag:
        parser.error(f"--hpo-objective {args.hpo_objective} needs --tag, otherwise the run collides with the base-grid cache cells (tuned on accuracy)")

    if args.device != "auto":
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else (
                "mps"
                if getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
                else "cpu"
            )
        )
    print(f"Using device: {device}")

    machine = args.machine or socket.gethostname().split(".")[0]
    env_tag = args.env_tag or _default_env_tag()
    job_tag = args.job_tag or _default_job_tag()
    git_sha = _git_sha(Path(__file__).resolve().parent)

    output_root = Path(args.out_dir)
    if args.mock_run:
        output_root = output_root / "mock"
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_root / "cache"
    work_dir = Path(args.work_dir) if args.work_dir else output_root / "work" / job_tag
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    stem = f"talent_benchmark__{machine}__{job_tag}" + (f"__{args.tag}" if args.tag else "")
    csv_path = output_root / f"{stem}.csv"
    errors_csv_path = output_root / f"{stem}__errors.csv"
    log_path = output_root / "logs" / f"{stem}.log"

    logger = _configure_logging(log_path)
    logger.info("Benchmark results will be written to %s", csv_path)
    logger.info(
        "machine=%s env=%s job=%s git=%s cache=%s work=%s tag=%s n_estimators=%s softmax_temperature=%s hpo_objective=%s n_trials=%s",
        machine,
        env_tag,
        job_tag,
        git_sha,
        cache_dir,
        work_dir,
        args.tag,
        args.n_estimators,
        args.softmax_temperature,
        args.hpo_objective,
        args.n_trials,
    )
    if args.mock_run:
        logger.info(
            "Quick-test mode enabled: sampling smaller datasets and disabling HPO."
        )

    datasets = list(args.datasets)
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        assert 0 <= k < n, "--shard k/N needs 0 <= k < N"
        datasets = datasets[k::n]
        logger.info("Shard %d/%d -> %d datasets", k, n, len(datasets))

    seeds_by_name = seeds_for_datasets(datasets, args.seeds, args.seed, args.n_seeds)
    with open(output_root / f"{stem}__seeds.json", "w") as f:
        json.dump(
            {"master_seed": args.seed, "explicit": args.seeds, "seeds": seeds_by_name},
            f,
            indent=1,
        )

    run = None
    if args.disable_wandb or os.environ.get("WANDB_MODE") == "disabled":
        logger.info("W&B disabled")
    else:
        import wandb

        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, config=vars(args)
        )

    all_results = []
    all_errors = []

    def handle_result(result: dict) -> None:
        all_results.append(result)
        wandb_payload = result.get("wandb_log", {})
        csv_record = {k: v for k, v in result.items() if k != "wandb_log"}
        csv_record["timestamp"] = datetime.now().isoformat(timespec="seconds")
        csv_record["mock_run"] = args.mock_run
        _append_to_csv(csv_record, csv_path)
        if wandb_payload and run is not None:
            run.log(wandb_payload)

    def handle_error(error_record: dict) -> None:
        all_errors.append(error_record)
        error_entry = {**error_record}
        error_entry["timestamp"] = datetime.now().isoformat(timespec="seconds")
        error_entry["mock_run"] = args.mock_run
        error_entry["machine"] = machine
        error_entry["git_sha"] = git_sha
        _append_to_csv(error_entry, errors_csv_path)

    set_seed(args.seed)
    for dataset_name in tqdm(datasets, desc="Datasets"):
        logger.info("--- Evaluating dataset: %s ---", dataset_name)
        dataset_seeds = seeds_by_name[dataset_name]

        for run_index, seed_int in enumerate(dataset_seeds, start=1):
            logger.info(
                "Running seed %d (%d/%d) with conformity scores %s for dataset %s",
                seed_int,
                run_index,
                len(dataset_seeds),
                ",".join(args.conformity_scores),
                dataset_name,
            )
            set_seed(seed_int)
            try:
                results, errors = evaluate_talent_dataset(
                    dataset_name=dataset_name,
                    dataset_path=args.dataset_path,
                    models_to_run=args.models,
                    device=device,
                    confidence_level=args.confidence_level,
                    conformity_scores=args.conformity_scores,
                    seed=seed_int,
                    n_trials=args.n_trials,
                    logger=logger,
                    on_result=handle_result,
                    on_error=handle_error,
                    mock_run=args.mock_run,
                    cache_dir=str(cache_dir),
                    work_dir=str(work_dir),
                    machine=machine,
                    env_tag=env_tag,
                    git_sha=git_sha,
                    val_frac=args.val_frac,
                    overwrite=args.overwrite,
                    n_estimators=args.n_estimators,
                    softmax_temperature=args.softmax_temperature,
                    tag=args.tag,
                    hpo_objective=args.hpo_objective,
                )
                logger.info(
                    "Completed dataset %s seed %d with %d rows and %d errors",
                    dataset_name,
                    seed_int,
                    len(results),
                    len(errors),
                )
            except Exception as dataset_error:
                logger.exception(
                    "Dataset %s seed %d failed with an unrecoverable error: %s",
                    dataset_name,
                    seed_int,
                    dataset_error,
                )
                handle_error(
                    {
                        "dataset": dataset_name,
                        "task_type": None,
                        "model": None,
                        "tag": args.tag or "",
                        "seed": seed_int,
                        "conformity_score": "|".join(args.conformity_scores),
                        "error": repr(dataset_error),
                    }
                )
                continue

    logger.info(
        "Benchmark finished with %d rows and %d errors. Results saved to %s",
        len(all_results),
        len(all_errors),
        csv_path,
    )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
