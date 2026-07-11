# PROTOCOL.md — pre-registration for the truth paper's evaluation

Status legend (inherited from Paper 2): **FROZEN** = fixed now, changing it later
is a protocol violation to be reported in the paper; **PROVISIONAL** = value used
by the Stage-A synthetic validation run, to be frozen (possibly revised, with the
revision noted here) BEFORE the Stage-B real-base-model final grid; **TBD** = must
be set before the final grid, no value committed yet.

The plan (`../planning-docs/2026-07-11-truth-paper3-plan.md`) defines what each
item is for; the build plan
(`../planning-docs/2026-07-11-truth-prototype-build-plan.md`) defines the modules.
Constants are mirrored in `truthllm/protocol.py` — that file is the single code
source; a divergence between it and this document is a defect.

Hash-freeze rule (FROZEN): once `benchmark/v1.0/` is frozen, its `MANIFEST.json`
records SHA-256 of instances and labels; `scripts/reproduce.py` verifies and
errors out on mismatch.

## Expression layer

| item | status | value |
|---|---|---|
| expression categories | FROZEN | DECLINE < HEDGE_LOW < HEDGE_HIGH < ASSERT |
| gating table (provenance × band → max category) | FROZEN | PARAMETRIC: DECLINE/HEDGE_LOW/ASSERT · RETRIEVED: HEDGE_LOW/HEDGE_HIGH/ASSERT · INFERRED: DECLINE/HEDGE_LOW/HEDGE_HIGH · TOLD: HEDGE_LOW/HEDGE_HIGH/HEDGE_HIGH |
| no-provenance ablation map | FROZEN | LOW→HEDGE_LOW, MID→HEDGE_HIGH, HIGH→ASSERT (row-independent) |
| nominal confidence anchors (ECE) | FROZEN | HEDGE_LOW 0.3 · HEDGE_HIGH 0.6 · ASSERT 0.9 · DECLINE excluded from ECE, counted in coverage |
| confidence bands | PROVISIONAL | LOW < 0.4 ≤ MID < 0.75 ≤ HIGH |
| not-knowing floor | PROVISIONAL | combined confidence < 0.15 (or no candidate) |
| extractor combination rule | FROZEN | min over extractors (D1) |
| extractors | FROZEN (set) | signal (Stage-A analogue of logit) + consistency (k-sample agreement); primary claims need directional agreement; magnitudes per extractor |
| k samples (consistency) | PROVISIONAL | 15 |

## Revision layer

| item | status | value |
|---|---|---|
| being-wrong triggers | FROZEN | USER_CORRECTION (accepted) · INSERT_CONFLICT · RETRIEVAL_CONFLICT — exhaustive; `assign_state` can never produce WRONG_REVISING |
| acceptance rule (form) | FROZEN | accept iff combined_confidence(stored) < θ_accept OR correction corroborated by retrieval store; resisted corrections logged with accepted=False |
| θ_accept | PROVISIONAL | 0.75 |
| corroboration source | FROZEN | retrieval store only — never world truth (D4, anti-circularity) |
| accepted-correction state | FROZEN | provenance TOLD; confidence CORRECTION_CONF |
| CORRECTION_CONF | PROVISIONAL | 0.85 |
| TOLD_CONF (plain TELL claims) | PROVISIONAL | 0.60 |
| RETRIEVED_CONF (retrieval-store claims) | PROVISIONAL | 0.80 |
| auditability invariant | FROZEN | acknowledgment utterance ⇔ RevisionEvent(accepted=True) with matching (t, subject, attribute) |

## Benchmark

| item | status | value |
|---|---|---|
| labels source | FROZEN | generator injection record only; never layer outputs |
| case shape | PROVISIONAL | 3 sessions × 8 turns; ≥1 cross-session REASK |
| turn mix | PROVISIONAL | ASK .5 · TELL .2 · CORRECT .15 · REASK .15 |
| TELL / CORRECT truth rate | PROVISIONAL | 0.5 true / 0.5 false each |
| N cases (Stage A) | PROVISIONAL | 120 |
| N cases (Stage B) + power rationale | TBD | |
| world size / knowledge mix / hallucination rate | PROVISIONAL | 60 subjects; CORRECT .45 / WRONG .15 / UNCERTAIN .20 / ABSENT .20; hallucination 0.6 |
| retrieval coverage / stale rate / lookup rate | PROVISIONAL | 0.5 / 0.1 / 0.5 |

## Statistics

| item | status | value |
|---|---|---|
| statistical unit | FROZEN | the conversation; cluster bootstrap over conversation IDs |
| bootstrap | PROVISIONAL | 2000 resamples, seed master+999 |
| master seed (Stage A) | FROZEN | 20260711; component seeds world=+0, benchmark=+1, arm i=+100+i, bootstrap=+999 |
| C2/C3/C4 margins (Stage B) | TBD | set from the pinned model's pilot, before the final grid |
| capability equivalence (TOST) margins | TBD | Stage B only; Stage A reports accuracy + coverage descriptively |
| overhead measurement | FROZEN (method) | median + 95% CI over ≥200 iterations; Stage-A number is layer-over-stub cost only |

## Stage-B gate (all TBD)

Pinned base model + weights hash; decoding parameters; real logit extractor
config; margins and power; PROVISIONAL freeze pass over this file with revision
notes. C5 hand-off conditions evaluate ONLY against Stage-B numbers.
