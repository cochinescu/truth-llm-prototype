# PROTOCOL.md — version-controlled protocol for the truth paper's evaluation

This file is an author-maintained protocol, not an independently registered
preregistration. For Stage B, the freeze and result files first entered the
repository history in the same commit; their claimed ordering therefore rests
on the author's working-tree record. Stage-C and robustness-draw freezes were
committed separately before their corresponding runs. The manuscript uses
"pre-specified" for Stage B and reserves "pre-committed" for those later,
separately committed freezes.

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
| capability-equivalence CI margin | TBD | Stage B only; Stage A reports accuracy + coverage descriptively |
| overhead measurement | FROZEN (method) | median + 95% CI over ≥200 iterations; Stage-A number is layer-over-stub cost only |

## Stage-B freeze pass (2026-07-11 — recorded BEFORE the Stage-B grid ran)

| item | status | value |
|---|---|---|
| base model | FROZEN | `Qwen/Qwen2.5-0.5B-Instruct` (HF), revision `7ae557604adf67be50417f59c2c2f167def9a775`; loader and cache both enforce the revision; single model (R6 scoped: no second model this pass) |
| decoding | FROZEN | greedy for answers, max 12 new tokens; system prompt fixes "name only / say unknown" |
| extractors (Stage B) | FROZEN | `logit` = mean token probability of the generated answer; `consistency` = agreement of k temperature samples (T=1.0, top-p 0.95) with the greedy answer; combination rule stays min (D1) |
| k samples | FROZEN | 8 — **revision note:** Stage-A PROVISIONAL was 15; reduced for compute before the grid, per the legend |
| world (Stage B) | FROZEN | real geography facts (`truthllm/worldb.py`): 60 countries → capital (parametric), continent derived via public capital→continent rule; ambiguous capitals excluded by construction |
| benchmark (Stage B) | FROZEN | `benchmark/stageb-v1.0/` generated from WorldB with seed MASTER+2, hash-frozen BEFORE the grid; same case shape/turn mix as Stage A |
| answer normalization | FROZEN | lowercase, de-accent, strip punctuation/leading "the", snap to known capital by containment; decline markers → no candidate. Normalization never consults world truth |
| N cases | FROZEN | 120 (power note: Stage-A cluster-bootstrap 95% CI half-widths at 120 cases were ≈±0.02–0.03 on rates — sufficient to resolve the margins below) |
| C2 margin | FROZEN | ECE_full ≤ ECE_control − 0.02 for BOTH controls (uniform, always-hedged) under BOTH extractors |
| C3 margin | FROZEN | contradiction(stateless) − contradiction(each store arm) ≥ 0.03; ack audit = 100%; false-accept ≤ true-accept − 0.10 |
| C4 margin | FROZEN | confidently-false(no_provenance) − confidently-false(full) ≥ 0.02; coverage cost reported |
| capability equivalence | FROZEN | CI-inclusion criterion: 95% cluster-bootstrap CI of per-conversation Δ(answer-when-given accuracy, full − uniform) within ±0.05 → equivalent; this is not a formal two one-sided test; coverage reported separately |
| Stage-A PROVISIONAL values | FROZEN as-is | bands 0.40/0.75; floor 0.15; θ_accept 0.75; CORRECTION/TOLD/RETRIEVED conf 0.85/0.60/0.80; benchmark shape/mix; retrieval 0.5/0.1/0.5 — no revisions needed after Stage A |
| model cache | FROZEN (mechanism) | per-fact answers/confidences computed once, written to `results/stageb/model_cache.json`; the grid reads only the cache (rerun determinism) |

C5 hand-off conditions evaluate ONLY against Stage-B numbers: IF C2 margin met
(manipulation check) AND capability equivalence holds, the condition pair
(full vs threshold_only/uniform ablations, configs + outputs in the archive) is
delivered for Paper 5; otherwise the paper says so and the Paper-5 truth arm
stays blocked (no post-hoc relabeling).

## Amendment 1 (2026-07-11) — amended manipulation check

**Disclosure (FROZEN):** this amendment was adopted AFTER the Stage-B grid ran
and its anchor-ECE manipulation check failed. The original frozen check and its
FAIL verdict remain reported, unchanged, wherever results appear; the amendment
never replaces or relabels them. The amended metric was specified from the
structural analysis of the failure (anchor-degeneracy: any single-category
policy scores as calibrated whenever corpus accuracy sits near its anchor), not
tuned against arm outcomes; its margin is anchored to the chance value 0.5,
which every single-category policy attains by construction.

**Ordering record (FROZEN, for inspectability):** the metric definition and the
0.60 / +0.05 / both-extractors margin were written into this file and into
`truthllm/metrics.py` BEFORE `expression_auc` was computed on any arm (the
scoring code and this section land in the same commit as the re-scored results,
so the within-session ordering is attested here rather than by commit
separation). Full disclosure: during the ECE failure analysis a coarse
bin-based reading of the FULL arm's combined reliability bins had been made
(suggesting combined discrimination roughly in the 0.6--0.7 range); no
per-extractor, per-control, or threshold-only AUC existed or was estimable
before the freeze, and the per-extractor robustness requirement — the clause
the amended check ultimately FAILS — was frozen without any per-extractor
information.

| item | status | value |
|---|---|---|
| amended manipulation check | FROZEN | expression-discrimination AUC: P(rank(category of a correct expressed claim) > rank(incorrect)), ties ½; category ranks HEDGE_LOW < HEDGE_HIGH < ASSERT; DECLINEs excluded (coverage) |
| rationale | FROZEN | the framework requirement is evidence-CONGRUENCE ("evidence-sensitive expression rather than performative humility") — a covariation property; discrimination captures it, anchor-ECE conflates it with anchor placement |
| margin | FROZEN | AUC(full) ≥ 0.60 AND ≥ AUC(each control) + 0.05, under BOTH extractors (controls = uniform, always_hedged; both are exactly 0.5 structurally) |
| anchor-ECE | FROZEN | still computed and reported; its Stage-B FAIL verdict stands in all outputs |
| C5 under Amendment 1 | FROZEN | delivered IFF amended check passes AND capability equivalence (already PASS); delivery is reported as "under Amendment 1" with this disclosure cited |
| auditability reciprocal (review F1) | FROZEN | ack_completeness = acknowledged turns / accepted revisions, reported as a measured fraction; the 100% claim applies to the soundness direction (utterance → accepted entry) only |

## Stage-C freeze (2026-07-11 — committed BEFORE the Stage-C grid ran; commit separation attests ordering)

**Configuration (FROZEN):** the consistency-extractor-gated layer — identical to
Stage B in every respect (same pinned model + cache, same frozen
`benchmark/stageb-v1.0`, same world, same arms, same seeds scheme, same
N=120) except that ALL arms run with the consistency extractor as the sole
confidence source (`extractor_mode="consistency"`). This is the path forward
identified by Stage B's extractor-dependence finding and named in the paper as
requiring its own separately committed pass.

**Disclosure of prior knowledge (FROZEN — read before the verdicts):**
- KNOWN before this freeze, from Stage B's per-extractor robustness runs
  (fidelity arms only): consistency-mode AUC full 0.661 [0.604, 0.727],
  uniform/always-hedged exactly 0.5, threshold-only 0.730 [0.697, 0.770];
  consistency-mode ECE for the same four arms. The manipulation check below is
  therefore EXPECTED to pass; we register it anyway for completeness and CI
  re-verification, and we say so.
- UNKNOWN at freeze time: every outcome for the non-fidelity arms under
  consistency gating (Stage B ran store_no_ack / no_provenance / stateless in
  combined mode only); all C3/C4 rates, coverage, accuracy, corrections, and —
  decisive for delivery — **capability equivalence under consistency-only
  gating** (single-extractor state assignment changes which claims decline or
  assert, so coverage/accuracy shift unpredictably).
- **The C5 delivery decision therefore hinges on the equivalence outcome,
  which is unknown at freeze time.**

| item | status | value |
|---|---|---|
| manipulation check (Stage C) | FROZEN | AUC(full, consistency) ≥ 0.60 AND ≥ each control + 0.05. The per-extractor robustness clause of Amendment 1 does not apply: the extractor choice is the pre-committed design decision of this configuration, justified by Stage B's mechanism finding |
| C3 / corrections / C4 margins | FROZEN | identical to Stage B (0.03 / 0.10 / 0.02) for comparability; outcomes unknown |
| capability equivalence | FROZEN | paired cluster-bootstrap CI of Δ answer-when-given accuracy (full − uniform) within ±0.05; unknown at freeze |
| C5 delivery (Stage C) | FROZEN | delivered IFF manipulation check AND equivalence pass, reported "under the Stage-C freeze" with this disclosure chain cited; no relabeling of Stage-B verdicts |
| outputs | FROZEN | `results/stagec/`; Stage-B results untouched; model cache shared read-only from `results/stageb/model_cache.json` |
| binding-frequency diagnostic | FROZEN (descriptive) | fraction of cached facts where logit < consistency (the min operator binds to the logit extractor) — reported as Stage-B context per reviewer suggestion; no margin attached |

### Stage-C robustness draw (2026-07-11 — committed BEFORE it ran; answers the instance-reuse objection)

A reviewer objection: Stage C re-scores the same frozen benchmark instances
that informed the configuration choice, so its confirmatory weight on an
independent instance draw is unproven. The empirical answer, frozen here
before running: regenerate the benchmark with a **fresh seed, MASTER+4
(20260715)** — same generator, same world, same case shape/mix, new instances
and injections — and re-run the full Stage-C grid on it under the identical
frozen margins. Outputs to `results/stagec-fresh/`. Both outcomes are
reportable: a pass defeats the instance-reuse objection; a fail scopes the
delivery to the original instance set and is reported as such. Nothing about
the layer, model, cache, arms, seeds scheme, or margins changes.
