# How a cell was produced

This directory is the harness that wrote the probability cells, kept for reading rather
than for rerunning. It is not installed, nothing in `src/tabsets` imports from it, and it
is not a dependency of anything in the release. Reproducing a cell from it needs GPU
hours, the datasets, and twenty-two model packages with incompatible requirements;
recomputing a *set* or a *metric* from a cell needs none of that, which is the point of
releasing the cells.

## The split, and what the calibration rows are kept out of

A cell holds one model's class probabilities on a calibration split and on a test split.
The level a conformal set delivers is only meaningful if the calibration rows played no
part in producing the model, so this is the part worth checking rather than believing.

The test split is the source benchmark's own fixed test file and is never touched here.
Everything below happens inside the remaining pool.

1. **Pool into training and calibration**, 70/30, stratified on the label, seeded by the
   run's seed (`utils.py:1098`). The row indices ride along with the split, so
   `idx_cal` in the cell records exactly which rows these were.
2. **Hyper-parameter search, for the nine tuned models**, runs on a 75/25 split *of the
   training part alone* (`utils.py:1174`). The calibration rows are not in either side of
   it, so no hyper-parameter was chosen with any knowledge of them.
3. **Early stopping, for the epoch-trained deep models**, holds out a further fraction *of
   the training part* (`utils.py:1251`). Models that do not train by epochs are fitted on
   the whole training part (`utils.py:1261`).
4. **The fit** sees only what step 3 left it (`utils.py:1265`). For the foundation models
   this is the whole training part, and it is also their in-context data: they condition
   on the rows they are given, and those rows are the training part. The calibration rows
   are not among them.
5. **Inference happens exactly once per split** (`utils.py:1279` and `utils.py:1282`).
   Nothing refits, and no search, schedule or threshold is chosen after this point.

So the calibration rows are held out of fitting, out of tuning, out of early stopping and
out of the in-context data, and the cell records which rows they were.

## What is here

| | |
|---|---|
| `talent_benchmark.py` | the driver: one run is a dataset, a model and a seed |
| `utils.py` | the model adapters, the search, and the split above |
| `cache_io.py` | the cell format, and the index over a directory of cells |
| `conformal.py` | the sets the harness wrote at run time, superseded by `tabsets.sets` |
| `check_cache.py` | rebuilds a run's recorded row from its cell and compares |
| `generate_synthetic_datasets.py` | the synthetic datasets, regenerated from their seeds |
| `configs/default/` | the defaults, for the thirteen models the source benchmark configures |
| `configs/opt_space/` | the search spaces, for the nine tuned models |
| `tuned/` | the hyper-parameters the search chose, one file per tuned model |
| `environments/` | the package versions the runs were made under |
| `tests/` | the cell format round-trips, the search objective, the constant-output guard |

Eight of the twenty-two models have no file under `configs/`: the foundation models added
after the source benchmark was released run from their vendor defaults, and the checkpoint
each one used is named in the model table of the release.

## What is not here, and why

The tabular-model library this harness wraps is not vendored. It is sixty thousand lines
of someone else's code, it is available from its own repository under its own licence, and
copying it here would say that the release maintains it. `environments/` names the version
the runs used.

The datasets are not redistributed either. The cells carry the row indices into them, and
`generate_synthetic_datasets.py` regenerates the synthetic ones from their seeds.

Model weights are not redistributed. Every checkpoint is identified in the model table.
