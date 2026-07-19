# Truth Prototype Results — Stage B (pinned real model, final grid)

Model: `Qwen/Qwen2.5-0.5B-Instruct` @ revision `7ae557604adf67be50417f59c2c2f167def9a775` on mps · greedy answers, k=8 consistency samples (T=1.0, top-p=0.95). Benchmark: benchmark/stageb-v1.0 (frozen, sha-verified, seed 20260713); 120 conversations/arm; cluster bootstrap n=2000 (unit = conversation). Master seed `20260711`.

## Pre-specified margin verdicts (PROTOCOL.md)

| Check | Result |
| --- | --- |
| C2 margin (logit): full ECE <= controls - 0.02 | **FAIL** |
| C2 margin (consistency): full ECE <= controls - 0.02 | **FAIL** |
| C3 margin: stateless - store arms >= 0.03 | **FAIL** |
| C3 audit: 100% acknowledgment traceability | PASS |
| C3 corrections: false-accept <= true-accept - 0.10 | PASS |
| C4 margin: no_provenance - full >= 0.02 | **FAIL** |
| Capability equivalence (95% CI within +/-0.05, full vs uniform) | PASS |

**C5 under the ORIGINAL frozen check: NOT delivered** (verdict stands, never relabeled).

## Amendment 1 (disclosed post-hoc; see PROTOCOL.md) — expression-discrimination AUC

| arm | combined | logit | consistency |
| --- | ---: | ---: | ---: |
| full | 0.653 | 0.410 | 0.661 |
| uniform | 0.500 | 0.500 | 0.500 |
| always_hedged | 0.500 | 0.500 | 0.500 |
| threshold_only | 0.715 | 0.569 | 0.730 |

| Amended check | Result |
| --- | --- |
| Amendment 1 (logit): full AUC >= 0.60 and >= controls + 0.05 | **FAIL** |
| Amendment 1 (consistency): full AUC >= 0.60 and >= controls + 0.05 | PASS |

**C5 under Amendment 1 (amended manipulation check AND capability equivalence): NOT delivered** — reported with the amendment's post-hoc disclosure; the original FAIL verdicts above remain in force as the pre-specified outcome.

## Expression fidelity (C2) — expression-ECE, lower is better

| arm | combined | logit | consistency |
| --- | ---: | ---: | ---: |
| full | 0.269 | 0.254 | 0.257 |
| uniform | 0.006 | 0.035 | 0.003 |
| always_hedged | 0.602 | 0.572 | 0.607 |
| threshold_only | 0.125 | 0.030 | 0.116 |

## Consistency & revision (C3)

| arm | contradiction rate | acks | traced | accepted revs | true-corr accept | false-corr accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.0000 | 42 | 42 | 197 | 0.8750 | 0.4197 |
| uniform | 0.0000 | 61 | 61 | 195 | 0.8649 | 0.4249 |
| always_hedged | 0.0000 | 66 | 66 | 200 | 0.8718 | 0.4508 |
| threshold_only | 0.0000 | 69 | 69 | 203 | 0.8673 | 0.4560 |
| store_no_ack | 0.0000 | 0 | 0 | 201 | 0.8729 | 0.4560 |
| no_provenance | 0.0000 | 57 | 57 | 197 | 0.8707 | 0.4301 |
| stateless | 0.0286 | 0 | 0 | 0 | — | — |

## Assertions & capability (C4)

| arm | confidently-false rate | coverage | answer accuracy |
| --- | ---: | ---: | ---: |
| full | 0.0182 | 0.8325 | 0.9242 |
| uniform | 0.0867 | 0.9251 | 0.9063 |
| always_hedged | 0.0000 | 0.9213 | 0.9019 |
| threshold_only | 0.0284 | 0.9219 | 0.9048 |
| store_no_ack | 0.0187 | 0.8288 | 0.9212 |
| no_provenance | 0.0305 | 0.9058 | 0.9173 |
| stateless | 0.0187 | 0.8106 | 0.9274 |

**Capability equivalence (full − uniform, answer-when-given accuracy):** Δ = 0.0179, 95% CI [0.0059, 0.0295], margin ±0.05 → EQUIVALENT.

## Overhead

| config | median (ms) | lo | hi |
| --- | ---: | ---: | ---: |
| full_layer_machinery | 0.0792 | 0.0757 | 0.1045 |
| cached_lookup_baseline | 0.0051 | 0.0050 | 0.0062 |
| model_query_median_ms(one-time,from_cache_build) | 106.6 | — | — |

## Scope and interpretation notes

- **These are the paper's headline numbers** (Stage A validated the
  instrument only). One pinned 0.5B model, one machine — claims are scoped
  to this model class; no second model was run (R6).
- The world is a 60-fact real-geography table chosen for unambiguity; the
  model's error pattern is its own (no injected corruption). Provenance
  tags remain pipeline instrumentation; benchmark contradictions and
  false corrections are injected by construction and mechanically scored.
- The layer surfaces the model's epistemic state; it does not improve the
  model's calibration or knowledge. Over wrong parametric beliefs the
  layer asserts falsehoods fluently — see confidently-false rates.
- Coverage cost of gating is reported beside the reduction (the trade IS
  the result). No trust/believability/perceived-mind measurement here.
- Model inference is cached per fact (greedy + k samples once); grid
  logical outputs are deterministic given the cache + seeds; timing rows
  are wall-clock medians + intervals. Per-query model latency is reported
  from the cache build, not per-turn.

Full register: `run_meta.json` · protocol: `../../PROTOCOL.md` · wall 78.6s.
