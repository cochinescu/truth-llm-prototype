# Truth — Prototype & Measured Evaluation

This folder holds the build and evaluation instructions for Paper 3's prototype:
the **metacognitive expression-and-revision layer** over a fixed base model. It is
intentionally separate from the manuscript (`../Truth_v1.tex`, not yet written —
results gate the manuscript, series rule) and from the plan
(`../planning-docs/2026-07-11-truth-paper3-plan.md`, which defines claims C1–C5,
the arms, and the honesty boundary this prototype serves).

**Key framing:** this is a *single-machine implementation plus mechanically scored
evaluation* of the truth layer — the three-state epistemic model, per-claim
(confidence, provenance) state, the expression function with the frozen
provenance-gating table, and the belief store + revision operator with auditable
acknowledgment. It is **not** a trained model, not a human study, and not a
product. One person (with AI help) can build and run all of it on a laptop. The
concrete deliverable is **five CSVs + four figures + `results/RESULTS.md`, all
regenerated from a single seeded command** (`scripts/reproduce.py`).

**Two-stage discipline (inherited from Paper 2 — read this first).** The layer's
claims are about *behavior of the layer*, not about any particular base model. So
the build runs in two stages:

- **Stage A (this build): synthetic validation.** The full pipeline runs against
  an **LLM-free synthetic base model** (`truthllm/basemodel.py`) whose knowledge
  is a seeded, deliberately corrupted copy of a synthetic world — so ground truth
  is knowable by construction, every metric is mechanically scorable, and the
  pipeline/metrics/tests are validated end-to-end. Stage-A numbers validate the
  *instrument*; they are never quoted as the paper's headline results.
- **Stage B (gated, later): the final grid** against one pinned small open-weight
  model, run only after `PROTOCOL.md`'s PROVISIONAL items are frozen and its TBD
  items are set. Stage-B produces the numbers the manuscript reports.

> **Honesty boundary (non-negotiable).** Every number here measures the *layer's
> expression and revision behavior* on synthetic or constructed inputs. The
> prototype does **not** measure trust, believability, perceived mind, or any
> human perception effect (Paper 5's job); it does **not** improve the base
> model's internal calibration (we surface, not improve); it does **not** measure
> open-domain consistency (the benchmark injects contradictions by construction —
> knowable cases). A well-behaved layer over a wrong model politely asserts
> falsehoods with well-ranked confidence. See `results/RESULTS.md` honesty notes
> and the plan's §3.

## 1. Scope — what to build (and what not to)

Build the smallest thing that exercises claims C2, C3, C4 end-to-end:

- **Synthetic world + base-model stub** — a seeded fact table (subjects ×
  attributes) and a base model holding a corrupted copy: per fact
  correct/wrong/uncertain/absent with a latent reliability, plus a hallucination
  path when asked about absent facts. This makes the three states real:
  absent = not-knowing, low-reliability = unsure, wrong-but-held = the revision
  target.
- **≥2 independent confidence extractors** (C2's requirement):
  `signal` (reads the stub's latent reliability + noise — Stage-A analogue of a
  logit extractor) and `consistency` (k resampled answers, agreement fraction —
  mechanism-faithful; ports unchanged to Stage B). Primary claims require
  directional agreement; magnitudes reported per extractor.
- **Provenance tagging as pipeline instrumentation** — PARAMETRIC / RETRIEVED /
  INFERRED / TOLD assigned by *which pipeline path produced the claim*, never by a
  classifier. A toy retrieval store (mostly correct, some stale) supplies
  RETRIEVED; one public derivation rule (city → region) supplies INFERRED; user
  TELL turns supply TOLD.
- **Expression function + the frozen gating table** — epistemic state →
  expression category (DECLINE / HEDGE_LOW / HEDGE_HIGH / ASSERT); the provenance
  class × confidence band → max-category table is FROZEN in `PROTOCOL.md` before
  code (plan §5 commitment).
- **Belief store + revision operator + acknowledgment** — session-persistent
  store; the being-wrong state entered by the three defined triggers only
  (accepted user correction / insertion conflict / retrieval contradiction); the
  correction-acceptance rule stated formally **including false-correction
  resistance** (an epistemic-humility layer that accepts every correction is
  sycophancy); every acknowledgment mechanically traceable to a revision-log
  entry.
- **The seven arms** (plan §6): full layer / uniform-confidence / always-hedged /
  threshold-only / store-without-acknowledgment / no-provenance / stateless.
- **Constructed multi-session contradiction benchmark** — seeded conversation
  scripts (ASK / TELL / CORRECT / re-ASK across sessions) with contradictions and
  true/false corrections injected by design; labels are the generator's injection
  record, never the layer's outputs (anti-circularity).
- **A measurement harness** with fixed seeds producing the five CSVs, four
  figures, and `RESULTS.md`.

**Do NOT build:**
- Any model training or fine-tuning; any prompt-engineering study.
- A UI, serving stack, or companion-app integration (Anima Felix is Paper 5's
  deployment; wiring it here would contaminate the matched-model identification).
- Real retrieval infrastructure — the retrieval store is an in-memory table.
- Any trust/believability/perception measurement — Paper 5.
- Stage B (real-model grid) in this pass — it is gated behind PROTOCOL.md.
- A general hallucination detector — confidence estimation is imported, not
  contributed (plan C1).

## 2. Tech choices (one lane, don't mix)

All Stage-A figures need only: seeded tables, resampling, counting, a bootstrap,
and matplotlib. **Pure `numpy` + `matplotlib` is sufficient** — no scipy unless a
specific statistic demands it, no LLM libraries until Stage B, no pandas (CSVs
written with the stdlib). Tests with `pytest`. Everything CPU, seconds not
minutes.

## 3. Layout

```
prototype/
  README.md            <- this file
  PROTOCOL.md          <- FROZEN / PROVISIONAL / TBD pre-registration (entropy legend)
  pyproject.toml
  truthllm/
    world.py           synthetic world: facts, city→region rule, retrieval docs
    basemodel.py       corrupted-copy stub: answer + latent reliability + hallucination
    extractors.py      signal + consistency extractors -> confidence in [0,1]
    state.py           Claim, Provenance, EpistemicState, three-state assignment
    expression.py      gating table (loaded from PROTOCOL constants) + rendering
    store.py           belief store, triggers, acceptance rule, revision log
    arms.py            the seven arm configurations
    pipeline.py        turn loop: benchmark script -> events, per arm
    metrics.py         expression-ECE, contradiction rate, ack audit, assertion
                       rate, capability, overhead (median+CI), cluster bootstrap
  benchmark/
    build_benchmark.py seeded generator -> versioned instances + labels
    score.py           language-agnostic scorer over an event log (JSON lines)
    v1.0/              frozen instances + labels + MANIFEST.json (sha256, params, seed)
  tests/               pytest, one file per module + test_reproduce.py
  results/             CSVs + figures + RESULTS.md + run_meta.json  (generated)
  scripts/
    reproduce.py       one seeded command -> every CSV + figure + RESULTS.md
    plot.py            CSVs -> the 4 figures
```

## 4. The evaluation — measurements → CSVs → figures

Anti-circularity rules, both inherited from Paper 1: (a) all labels come from the
generator's injection record; (b) every positive test has a negative control that
CAN fail — the uniform arm must NOT track accuracy (C2), the stateless arm must
NOT reduce contradictions (C3), and if an arm designed to fail passes, that is a
scoring bug, not a result.

| # | Measurement (claim) | Produces | Figure |
|---|---|---|---|
| 1 | **Expression fidelity (C2):** reliability diagram over expression categories; expression-ECE with cluster-bootstrap CIs (unit = conversation); full layer vs uniform vs always-hedged, under BOTH extractors with the directional-agreement rule | `fidelity.csv` | `fig1_reliability.png` |
| 2 | **Revision & consistency (C3):** long-horizon self-contradiction rate, store arms vs stateless; acknowledgment on its own two axes — (a) conditioning effect vs store-without-ack (may be null, reported either way), (b) auditability: every acknowledgment traces to a revision-log entry; plus correction handling: true-correction acceptance vs false-correction resistance | `consistency.csv` | `fig2_consistency.png` |
| 3 | **Provenance gating (C4):** confidently-asserted-false rate, full vs no-provenance; capability reported as answer-when-given accuracy AND coverage (declines counted separately, both shown — gating trades assertions for declines and the trade is reported, not hidden) | `assertions.csv`, `capability.csv` | `fig3_assertions.png` |
| 4 | **Overhead:** layer cost per turn vs stateless baseline, median + 95% CI over many iterations (Stage A measures layer-only cost over the stub; real-model overhead is a Stage-B number) | `overhead.csv` | `fig4_overhead.png` |

## 5. Milestones

1. **M1 — world + base-model stub + extractors.** Seeded world; corrupted copy;
   both extractors; test: extractor confidence correlates with latent reliability,
   consistency-extractor agreement ≈ reliability, all seeded-deterministic.
2. **M2 — state + provenance + expression.** Three-state assignment; provenance
   by pipeline path; gating table enforced; the plan's 2–4-turn worked example as
   an executable test.
3. **M3 — store + revision + acknowledgment.** The three triggers and nothing
   else enters being-wrong; acceptance rule incl. false-correction resistance;
   100% acknowledgment auditability by construction, tested.
4. **M4 — benchmark + arms + metrics → CSVs + figures + RESULTS.md.** Frozen
   PROTOCOL values; one-command seeded reproduction; honesty notes written.
5. **M4.5 — package the benchmark** as `benchmark/v1.0/` with MANIFEST.json
   (sha256 + params + seed) — a repackaging of M4's generator, not a rewrite.
6. **M5 — Stage B + fold-back (gated, NOT this build).** Freeze PROVISIONAL
   items, set TBDs (real-model margins, power), run the final grid on the pinned
   model, fold measured numbers into the manuscript via `/co-author` only.

## 6. Done criteria (Stage A)

- `pytest` green; `scripts/reproduce.py --seed <master>` regenerates every CSV,
  figure, and `RESULTS.md` from scratch.
- The negative controls fail where they must (uniform arm's ECE worse than full
  layer's; stateless arm's contradiction rate higher than store arms').
- Every acknowledgment event in every arm traces to a revision-log entry (100%).
- `results/RESULTS.md` carries the honesty notes: Stage-A synthetic; instrument
  validation, not headline results; trust/believability NOT measured; the
  quadrant-of-truth register (a null is a finding).
- `benchmark/v1.0/` frozen with MANIFEST hashes; `PROTOCOL.md` statuses accurate.
