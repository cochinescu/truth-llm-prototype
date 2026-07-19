# Truth Prototype Results — Stage C (consistency-gated configuration, separately committed pass)

Model: `Qwen/Qwen2.5-0.5B-Instruct` @ revision `7ae557604adf67be50417f59c2c2f167def9a775` on mps · greedy answers, k=8 consistency samples (T=1.0, top-p=0.95). Benchmark: benchmark/stageb-v1.0 (frozen, sha-verified, seed 20260713); 120 conversations/arm; cluster bootstrap n=2000 (unit = conversation). Master seed `20260711`.

## Pre-specified margin verdicts (PROTOCOL.md)

| Check | Result |
| --- | --- |
| Stage-C manipulation check (consistency): full AUC >= 0.60 and >= controls + 0.05 | PASS |
| C3 margin: stateless - store arms >= 0.03 | **FAIL** |
| C3 audit: 100% acknowledgment traceability | PASS |
| C3 corrections: false-accept <= true-accept - 0.10 | PASS |
| C4 margin: no_provenance - full >= 0.02 | **FAIL** |
| Capability equivalence (95% CI within +/-0.05, full vs uniform) | PASS |

**C5 under the Stage-C freeze conditions (pre-verdict summary below)**

## Expression-discrimination AUC (Stage-C manipulation check)

| arm | consistency |
| --- | ---: |
| full | 0.661 |
| uniform | 0.500 |
| always_hedged | 0.500 |
| threshold_only | 0.730 |

**C5 under the Stage-C freeze (manipulation check AND capability equivalence): DELIVERED** — evaluated per the pre-committed Stage-C freeze (PROTOCOL.md), whose prior-knowledge disclosure applies: the manipulation-check value was expected from Stage B; the equivalence outcome was unknown at freeze.

## Expression fidelity (C2) — expression-ECE, lower is better

| arm | consistency |
| --- | ---: |
| full | 0.257 |
| uniform | 0.003 |
| always_hedged | 0.607 |
| threshold_only | 0.116 |

## Consistency & revision (C3)

| arm | contradiction rate | acks | traced | accepted revs | true-corr accept | false-corr accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.0000 | 42 | 42 | 195 | 0.8707 | 0.4352 |
| uniform | 0.0000 | 71 | 71 | 205 | 0.8696 | 0.4508 |
| always_hedged | 0.0000 | 63 | 63 | 197 | 0.8649 | 0.4352 |
| threshold_only | 0.0000 | 70 | 70 | 204 | 0.8696 | 0.4404 |
| store_no_ack | 0.0000 | 0 | 0 | 194 | 0.8718 | 0.4352 |
| no_provenance | 0.0000 | 50 | 50 | 190 | 0.8673 | 0.4301 |
| stateless | 0.0229 | 0 | 0 | 0 | — | — |

## Assertions & capability (C4)

| arm | confidently-false rate | coverage | answer accuracy |
| --- | ---: | ---: | ---: |
| full | 0.0182 | 0.8309 | 0.9259 |
| uniform | 0.0952 | 0.9294 | 0.8975 |
| always_hedged | 0.0000 | 0.9197 | 0.9069 |
| threshold_only | 0.0278 | 0.9283 | 0.8986 |
| store_no_ack | 0.0182 | 0.8320 | 0.9196 |
| no_provenance | 0.0268 | 0.9085 | 0.9246 |
| stateless | 0.0187 | 0.7994 | 0.9304 |

**Capability equivalence (full − uniform, answer-when-given accuracy):** Δ = 0.0284, 95% CI [0.0155, 0.0420], margin ±0.05 → EQUIVALENT.

## Overhead

| config | median (ms) | lo | hi |
| --- | ---: | ---: | ---: |
| full_layer_machinery | 0.0773 | 0.0744 | 0.1103 |
| cached_lookup_baseline | 0.0049 | 0.0048 | 0.0051 |
| model_query_median_ms(one-time,from_cache_build) | 106.6 | — | — |

## Scope and interpretation notes

- **Stage-C numbers under the pre-committed Stage-C freeze** (see the
  PROTOCOL.md disclosure chain); Stage-A/B verdicts stand unchanged.
  One pinned 0.5B model, one machine; the extractor choice is the
  configuration's pre-committed design decision.
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

Full register: `run_meta.json` · protocol: `../../PROTOCOL.md` · wall 53.52s.
