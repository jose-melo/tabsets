# TabSets

Prediction sets for tabular classifiers, computed from cached probabilities.

A classifier that is right most of the time can still refuse to name any label at all.
At a requested level of 90 %, the conformal set of a confident model is sometimes empty,
and an empty set is not a cautious answer but a missing one. TabSets measures that: how
often a model commits to at least one label, and how well it covers the truth where it
does. The two order the model families in opposite directions, which is why the usual
single score hides the trade-off rather than showing it.

Everything here is a function of four arrays one run produces: the class probabilities
and labels on a calibration split, and the same on a test split. No model is refitted to
compute a set at a new level or under a different score.

```python
import tabsets

r = tabsets.evaluate(p_cal, y_cal, p_test, y_test, alpha=0.10, score="lac")
r["commitment"], r["sscs_plus"], r["empty_rate"], r["coverage"]
```

Sets can also be built with no labels anywhere. The threshold is solved on the model's
own probabilities, on the identity that the mass a calibrated model puts below a cut is
the error rate it predicts for itself below that cut:

```python
q = tabsets.labelfree_threshold(p_test, alpha=0.10)
s = tabsets.labelfree_sets(p_test, alpha=0.10)
s.guarantee          # None: no labels were used, so nothing is guaranteed
```

That field is not decoration. Sets calibrated on held-out labels report
`"split conformal: marginal, finite-sample, exchangeable calibration"`; sets built
without labels report `None`. Whether the level is nonetheless delivered is a property
of the model's probability scale, and it is the measurement the article reports rather
than a claim the library makes.

## Install

```bash
pip install tabsets                 # numpy, pandas, scipy, scikit-learn, pyarrow
pip install 'tabsets[test]'         # adds MAPIE, for the parity tests
```

## What is in the release

| | |
|---|---|
| `src/tabsets/` | the library: sets, metrics, the label-free threshold, the statistics, the cell index |
| `reproduce/` | the macro file, the tables and the figures of the article, regenerated from `reproduce/data` |
| `benchmark/` | the harness that wrote the cells, for reading rather than rerunning; its README documents the split |
| `reproduce/data/manifest.parquet` | one row per run: provenance, split sizes and hashes, tuning record, checkpoint |
| `reproduce/data/prereg.csv` | the analysis plan: the twenty analyses specified before the cache was read, with each outcome |

The calibration and test probabilities themselves are deposited separately, and are
needed only to recompute sets at a level or under a score the released tables do not
already carry. Set `TABSETS_CACHE` to where they were unpacked.

They come as a few parquet shards with an index beside them, rather than as one file per
run: 45,000 files is the right shape for a machine writing them one at a time and the
wrong shape for handing them to someone else. It saves no space, since the arrays are
float64 and stay float64 — a prediction set is a comparison against a threshold, and a
point sitting on that threshold would move if the value were rounded. Either layout is
read; `scripts/export_cache.py --verify` compares a sample of the shards against the
files they were packed from, array by array.

Prediction sets built from labelled calibration data reproduce MAPIE's split-conformal
classifier exactly on multiclass targets, which is what makes the released numbers
checkable against an independent implementation. On binary targets MAPIE declines the
adaptive, regularised and top-*k* scores; those sets are this package's own and say so
in `implementation`.

## The blocks

A comparison across models is only paired if every model is present on the same
datasets. A *block* is the subset of runs where that holds, recomputed from the manifest
rather than frozen, so it grows as runs are added:

```python
from tabsets import blocks
blocks.block("F1")          # every model, on the real datasets inside the conference envelope
```

Every pass takes `--block`, and the article names the block behind each table for the
same reason.

## Reproducing the article

```bash
make test                   # the suite; MAPIE parity included when MAPIE is installed
make reproduce              # the macro file, the tables and the figures, from data/
make runs                   # rebuild data/ itself, from the probability cells
```

`make reproduce` reads nothing but the tables in `reproduce/data`, and its output is
compared byte for byte against a committed copy, so a change in the data shows up as a
diff rather than as a silent difference. Only `make runs` needs the deposited
probabilities.

## Citing

The article is the citation; `CITATION.cff` carries it. The library and the data are
released under the terms in `LICENSE`.
