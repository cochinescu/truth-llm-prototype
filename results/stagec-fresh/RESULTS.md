# Truth Prototype Results — Stage C (consistency-gated configuration, separately committed pass)

Model: `Qwen/Qwen2.5-0.5B-Instruct` @ revision `7ae557604adf67be50417f59c2c2f167def9a775` on mps · greedy answers, k=8 consistency samples (T=1.0, top-p=0.95). Benchmark: fresh in-memory draw, seed 20260715; 120 conversations/arm; cluster bootstrap n=2000 (unit = conversation). Master seed `20260711`.

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
| full | 0.677 |
| uniform | 0.500 |
| always_hedged | 0.500 |
| threshold_only | 0.720 |

**C5 under the Stage-C freeze (manipulation check AND capability equivalence): DELIVERED** — evaluated per the pre-committed Stage-C freeze (PROTOCOL.md), whose prior-knowledge disclosure applies: the manipulation-check value was expected from Stage B; the equivalence outcome was unknown at freeze.

## Expression fidelity (C2) — expression-ECE, lower is better

| arm | consistency |
| --- | ---: |
| full | 0.258 |
| uniform | 0.015 |
| always_hedged | 0.589 |
| threshold_only | 0.122 |

## Consistency & revision (C3)

| arm | contradiction rate | acks | traced | accepted revs | true-corr accept | false-corr accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.0000 | 56 | 56 | 222 | 0.8917 | 0.4818 |
| uniform | 0.0000 | 84 | 84 | 229 | 0.8938 | 0.4818 |
| always_hedged | 0.0000 | 87 | 87 | 232 | 0.8929 | 0.4682 |
| threshold_only | 0.0000 | 85 | 85 | 230 | 0.8850 | 0.5136 |
| store_no_ack | 0.0000 | 0 | 0 | 221 | 0.8974 | 0.4773 |
| no_provenance | 0.0000 | 65 | 65 | 218 | 0.8909 | 0.4727 |
| stateless | 0.0202 | 0 | 0 | 0 | — | — |

## Assertions & capability (C4)

| arm | confidently-false rate | coverage | answer accuracy |
| --- | ---: | ---: | ---: |
| full | 0.0180 | 0.8259 | 0.9071 |
| uniform | 0.1061 | 0.9195 | 0.8846 |
| always_hedged | 0.0000 | 0.9238 | 0.8893 |
| threshold_only | 0.0294 | 0.9266 | 0.8890 |
| store_no_ack | 0.0174 | 0.8188 | 0.9096 |
| no_provenance | 0.0288 | 0.8928 | 0.9171 |
| stateless | 0.0196 | 0.7851 | 0.9224 |

**Capability equivalence (full − uniform, answer-when-given accuracy):** Δ = 0.0225, 95% CI [0.0094, 0.0364], margin ±0.05 → EQUIVALENT.

## Overhead

| config | median (ms) | lo | hi |
| --- | ---: | ---: | ---: |
| full_layer_machinery | 0.0755 | 0.0729 | 0.0940 |
| cached_lookup_baseline | 0.0048 | 0.0047 | 0.0052 |
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

Full register: `run_meta.json` · protocol: `../../PROTOCOL.md` · wall 46.49s.
