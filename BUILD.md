# Build and reproduction instructions

Run all commands from `prototype/`. Python 3.12 or later is required (the pinned NumPy 2.5.1 declares `>=3.12`).

## Install

The default environment reproduces the cached analyses and figures without
installing a model runtime:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

To rebuild the model cache from the pinned Hugging Face revision, install the
additional inference dependencies:

```bash
.venv/bin/python -m pip install -e ".[dev,stageb]"
```

The direct numerical, plotting, test, and model dependencies are exactly pinned
in `pyproject.toml`. The committed cache records the model identifier, revision,
generation settings, seeds, and observed query latency.

## Verify

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider
```

The test suite checks benchmark hashes, expression and revision invariants,
ablation behavior, offline Stage-B normalization, and agreement between the
model pin and committed cache metadata.

## Reproduce the reported outputs

```bash
# Stage A: synthetic instrument validation
.venv/bin/python scripts/reproduce.py

# Stage B: primary real-model grid, using the committed cache
.venv/bin/python scripts/reproduce_stageb.py --stage b

# Stage C: consistency-gated grid on the frozen conversations
.venv/bin/python scripts/reproduce_stageb.py --stage c

# Stage-C robustness draw used in the manuscript
.venv/bin/python scripts/reproduce_stageb.py --stage c \
  --bench-seed 20260715 --outdir results/stagec-fresh

# Fact-clustered sensitivity analysis
.venv/bin/python scripts/sensitivity_fact_cluster.py

# Post hoc diagnostics behind the 2026-09 review (correction-path decomposition,
# acknowledgment breakdown, fact-clustered held-gap and logit-AUC intervals)
.venv/bin/python scripts/sensitivity_post_review.py
```

Each grid writes CSV files, event logs, four figures, `RESULTS.md`, and
`run_meta.json` to its output directory. The benchmark loader verifies every
frozen instance and label file against its SHA-256 manifest before a run.

Re-render the four Stage-B figures without recomputing metrics:

```bash
.venv/bin/python -c \
  'from pathlib import Path; from scripts.plot import make_figures; make_figures(Path("results/stageb"))'
```

Score an event log independently of the pipeline:

```bash
.venv/bin/python benchmark/score.py \
  results/stageb/events/full.jsonl benchmark/stageb-v1.0/labels.jsonl
```

## Rebuilding the model cache

The reported analysis should normally use the committed
`results/stageb/model_cache.json`. If a cache is built in a clean copy where
that file is absent, `reproduce_stageb.py` downloads only the exact model
revision recorded above, generates one greedy answer and eight sampled answers
per fact, and writes the cache before evaluating the grid. A cache with a
different model identifier or revision is rejected.
