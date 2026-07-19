# Truth prototype and evaluation

This directory contains the implementation and all mechanically scored
artifacts reported in [`../Truth_v1.tex`](../Truth_v1.tex). The system is an
expression-and-revision layer over a fixed base model: three epistemic states,
per-claim confidence and provenance, provenance-aware expression rules, and a
persistent belief store with logged revision and acknowledgment.

The prototype is not a trained model, a human-subjects study, or a product. It
does not measure trust, believability, perceived mind, or open-domain
consistency. Benchmark contradictions and false corrections are injected by a
seeded generator and scored against generator labels; no language model is used
as a judge.

## Evaluation stages

| Stage | Purpose | Inputs | Output directory |
| --- | --- | --- | --- |
| A | Validate the instrument against an LLM-free synthetic model with known ground truth | `benchmark/v1.0/` | `results/` |
| B | Evaluate all seven arms on a fixed 60-fact geography world and cached Qwen2.5-0.5B-Instruct outputs | `benchmark/stageb-v1.0/` | `results/stageb/` |
| C | Evaluate consistency-only gating under the separately committed Stage-C protocol | same frozen Stage-B conversations | `results/stagec/` |
| C fresh | Repeat Stage C on a newly seeded conversation set over the same 60 facts | fresh generated conversations | `results/stagec-fresh/` |
| Sensitivity | Recompute the Stage-C selection criteria with fact-clustered uncertainty | committed Stage-C event logs | `results/stagec/sensitivity_fact_cluster.csv` |

Stage B uses `Qwen/Qwen2.5-0.5B-Instruct` at revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Both the loader and the committed
cache enforce this revision. The cache makes the reported grids reproducible
without downloading or rerunning the model; rebuilding the cache requires the
optional `stageb` dependencies documented in [`BUILD.md`](BUILD.md).

## Implemented components

- `truthllm/world.py` and `basemodel.py`: synthetic world and corrupted-copy
  model used in Stage A.
- `truthllm/worldb.py` and `llm_model.py`: real geography world, exact model
  revision, output normalization, and cached real-model adapter.
- `truthllm/extractors.py`, `state.py`, and `expression.py`: confidence signals,
  epistemic states, and the frozen expression map.
- `truthllm/store.py`: revision triggers, correction-acceptance rule, revision
  log, and acknowledgment linkage.
- `truthllm/arms.py` and `pipeline.py`: the seven ablation configurations and
  conversation runner.
- `truthllm/metrics.py`: expression ECE and AUC, contradiction and correction
  measures, capability/coverage, overhead, and paired bootstrap intervals.
- `benchmark/`: seeded generators, frozen instances and labels, SHA-256
  manifests, and a language-agnostic event-log scorer.
- `scripts/`: reproduction, plotting, and fact-clustered sensitivity commands.

## Interpretation constraints

- The protocol is version-controlled but was not deposited in an independent
  preregistration registry. The Stage-B freeze and results first appear in the
  same commit; Stage-C and its fresh-draw protocol were committed separately
  before their runs. See [`PROTOCOL.md`](PROTOCOL.md).
- The capability criterion is inclusion of a paired 95% bootstrap confidence
  interval within `[-0.05, 0.05]`; it is not a formal two one-sided test.
- Conversation-clustered Stage-C criteria pass, but fact-clustered intervals
  are unresolved with only 60 distinct facts.
- The acknowledgment result is soundness in the direction utterance to accepted
  revision-log entry. It is not a completeness claim for all stored revisions.

See [`BUILD.md`](BUILD.md) for exact commands. The software is distributed
under the repository's MIT license; citation metadata are in
[`../CITATION.cff`](../CITATION.cff).
