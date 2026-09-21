import os
import torch
import pandas as pd
import argparse
import wandb

from utils import evaluate_on_openml, set_seed

seeds = [
    382114703,
    3773843095,
    1543291710,
    848852757,
    576485429,
    4208438384,
    2965207055,
    2386776174,
    1385245016,
    618360380,
    3721738935,
    2011376238,
    1472753441,
    2269463650,
    2330484666,
    391807351,
    802708881,
    2702360181,
    363512744,
    3147449495,
]


openml_datasets = [
    1479,
    43946,
    15,
    997,
    31,
    188,
    1046,
    1471,
    1476,
    45060,
    4534,
    32,
    45040,
    45074,
    1044,
    1053,
    1459,
    44122,
    45062,
    45023,
    375,
    4538,
    45553,
    41972,
    44130,
    1496,
    1507,
    803,
    182,
    42889,
    44,
    1475,
    44150,
    458,
    30,
    1497,
    1489,
    44124,
    44186,
    41146,
    44489,
    1037,
    42636,
    1557,
    28,
    1043,
    41156,  # X columns with only one distinct value
    40708,
    40497,  # X columns with only one distinct value
    40707,
    40713,
    40677,
    40678,
    3,
    46,
    40670,  # high gpu usage
    41145,  # high gpu usage
    45075,  # classes [-1, 1] instead of [0, 1]
    42178,  # y is an object instead of a category
    1589,  # as_frame fails -> ARFF dataset - Compressed Sparse Row
    41144,
    41143,
    40478,
    44091,
    1487,
    1548,
    45540,
    45539,
    45538,
    45537,
    45536,
    44528,
    36,
    1067,
    22,
    18,
    14,
    16,
    12,
    41721,
    41875,
    41882,
    40664,
    43442,
    45648,
    40646,
    42464,
    1501,
    23,
    1050,
    54,
    185,
    43895,
    1049,
    43812,
    1068,
    1552,
    1444,
    # 372,    # n_classes (y_true) and dimension of y_score is not matching -> error spliting the data
    # 1491,   # n_classes (y_true) and dimension of y_score is not matching -> error spliting the data
    # 1492,   # n_classes (y_true) and dimension of y_score is not matching -> error spliting the data
    # 1493,   # 100 classesm TabPFN supports up to 10
    # 40498,  # n_classes (y_true) and dimension of y_score is not matching -> error spliting the data
    # 40499,  # 11 classesm TabPFN supports up to 10
    # 41705   # X columns with only one distinct value -> column with one value and NaN &
    # 11 classesm TabPFN supports up to 10
    # 301,    # 2160 features, TabPFN supports up to 500
    # 20,     # 1648 features, TabPFN supports up to 500
]


def main():
    parser = argparse.ArgumentParser(description="Run Conformal Prediction Benchmark")
    parser.add_argument(
        "--dataset_id", type=int, required=True, help="OpenML dataset ID"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--confidence_level",
        type=float,
        default=0.9,
        help="Target confidence level for prediction sets",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tabular-uncertainty-benchmark",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="W&B entity name"
    )
    args = parser.parse_args()

    # Pick the best available device
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

    # Setup results directory
    out_folder = "results"
    os.makedirs(out_folder, exist_ok=True)

    # Set seed for reproducibility
    set_seed(args.seed)

    # Initialize W&B
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),  # Log all command-line arguments
    )
    print(f"Running on dataset {args.dataset_id} with seed {args.seed}")

    # The main evaluation function
    results = evaluate_on_openml(
        dataset_id=args.dataset_id,
        device=device,
        seed=args.seed,
    )

    # Log results to W&B and save locally
    # Combine metrics from all models into a single dictionary for W&B logging
    wandb_log_data = {}
    for res in results:
        wandb_log_data.update(res.pop("wandb_log"))

    wandb.log(wandb_log_data)

    # Save detailed results to a local CSV
    df = pd.DataFrame(results)
    df.to_csv(f"{out_folder}/{args.seed}_{args.dataset_id}.csv", index=False)

    run.finish()


if __name__ == "__main__":
    main()
