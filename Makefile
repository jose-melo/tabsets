PY ?= python3
BLOCK ?= F1
export PYTHONPATH := src

.PHONY: test reproduce runs manifest clean

test:
	$(PY) -m pytest tests -q

reproduce:
	$(PY) -m reproduce

runs:
	$(PY) -m tabsets summarize --block $(BLOCK) --out reproduce/data/runs_$(BLOCK).parquet

manifest:
	$(PY) -m tabsets manifest --out reproduce/data/manifest.parquet

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache reproduce/out
