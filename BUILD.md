# Building & running the prototype

This is the **implementation** of the plan in [`README.md`](README.md): the
Stage-A synthetic validation of the truth expression-and-revision layer.
Everything is test-driven and reproducible from a seed.

## Setup

```bash
cd prototype
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"    # numpy, matplotlib, pytest
```

## Run the tests

```bash
.venv/bin/python -m pytest -q
```

## Reproduce every measurement + figure (one command)

```bash
.venv/bin/python scripts/reproduce.py                 # full Stage-A run
.venv/bin/python scripts/reproduce.py --quick         # fast smoke run
```

Outputs land in `results/`: five CSVs (`fidelity`, `consistency`, `assertions`,
`capability`, `overhead`) plus `fidelity_bins.csv`, the four headline figures
(`fig1..fig4_*.png`), `run_meta.json`, per-arm event logs under
`results/events/` (the language-agnostic scorer's input), and a generated
`RESULTS.md` with the pre-registered instrument checks and honesty notes.
Re-render figures alone with `.venv/bin/python scripts/plot.py`.

## Freeze / verify the benchmark

```bash
.venv/bin/python benchmark/build_benchmark.py --freeze   # writes benchmark/v1.0/
```

Once frozen, `reproduce.py` loads `v1.0/` and errors out if any file's SHA-256
no longer matches `MANIFEST.json` (the PROTOCOL hash-freeze rule). Score any
event log independently of this package:

```bash
python3 benchmark/score.py results/events/full.jsonl benchmark/v1.0/labels.jsonl
```

## Module map (`truthllm/`)

| module | role | build-plan ref |
|---|---|---|
| `protocol.py` | PROTOCOL.md constants as code (single source) | §0 |
| `world.py` | ground-truth facts, city→region rule, retrieval docs | §1 |
| `basemodel.py` | corrupted-copy stub + hallucination path | §1 |
| `extractors.py` | signal + consistency confidence extractors | §1 |
| `state.py` | Claim, provenance, three-state assignment | §2 |
| `expression.py` | frozen gating table + rendering | §2 |
| `store.py` | belief store, triggers, acceptance rule, revision log | §3 |
| `arms.py` | the seven arm configurations | §4 |
| `pipeline.py` | turn loop; provenance by resolution path | §4 |
| `metrics.py` | ECE, contradiction, audit, assertions, overhead, bootstrap | §4 |
