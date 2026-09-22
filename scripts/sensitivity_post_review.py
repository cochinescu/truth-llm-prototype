"""Post hoc descriptive diagnostics requested by the 2026-09-22 pre-submission
review. Replays the frozen Stage-B and Stage-C `full` arms from the committed
per-fact cache (no new model samples) and writes
results/stageb/sensitivity_post_review.csv with:

  * the correction-acceptance decomposition behind the pre-specified mixed
    rates: corrections of unheld claims (accepted by rule), corrections that
    disagree with a held belief, and agreeing corrections (dropped by the
    scorer);
  * a subject-level (fact-clustered) bootstrap interval for the held-belief
    true-minus-false acceptance gap (Stage B);
  * a subject-level bootstrap interval for the Stage-B logit-gated
    expression-discrimination AUC (conversation-clustered in the paper);
  * the acknowledgment breakdown (rendered admissions vs. flagged ask turns),
    the accepted-revision composition, the HEDGE_HIGH provenance composition,
    and the realized retrieval-store composition.

The frozen verdicts are NOT recomputed or relabeled; this is a disclosed
post hoc analysis. Usage: python scripts/sensitivity_post_review.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.build_benchmark import load  # noqa: E402
from sensitivity_fact_cluster import _auc, _by_subject, _ci  # noqa: E402
from truthllm import protocol  # noqa: E402
from truthllm.arms import ARM_ORDER, ARMS  # noqa: E402
from truthllm.llm_model import LLMBaseModel  # noqa: E402
from truthllm.metrics import correction_scores, expression_auc  # noqa: E402
from truthllm.pipeline import run_conversation  # noqa: E402
from truthllm.store import RevisionTrigger  # noqa: E402
from truthllm.worldb import WorldB  # noqa: E402

MODES = ["combined", "logit", "consistency"]  # reproduce_stageb.EXTRACTOR_MODES


def _run(instances, arm, mode, model, docs, rules):
    i = ARM_ORDER.index(arm)
    rng = np.random.default_rng(protocol.SEED_ARM_BASE + i + 1000 * MODES.index(mode))
    return [(inst["case_id"], run_conversation(inst, ARMS[arm], model, docs, rng,
                                               extractor_mode=mode, rules=rules))
            for inst in instances]


def correction_decomposition(case_events, labels):
    """Counts keyed by (label, slice, accepted); slice in unheld/held/agreeing."""
    c = Counter()
    for cid, events in case_events:
        tl = labels[cid]
        for ev in events:
            if ev.turn_type != "CORRECT":
                continue
            lab = "true" if tl[ev.t]["correct_true"] else "false"
            corr = [r for r in ev.revision_events if r.trigger is RevisionTrigger.USER_CORRECTION]
            if not corr:
                c[(lab, "agreeing", False)] += 1
                continue
            r = corr[0]
            c[(lab, "unheld" if r.old_value is None else "held", bool(r.accepted))] += 1
    return c


def held_gap_fact_bootstrap(case_events, labels, n=protocol.BOOTSTRAP_N, seed=protocol.SEED_BOOTSTRAP):
    per = defaultdict(lambda: {"t": [], "f": []})
    for cid, events in case_events:
        tl = labels[cid]
        for ev in events:
            if ev.turn_type != "CORRECT":
                continue
            corr = [r for r in ev.revision_events
                    if r.trigger is RevisionTrigger.USER_CORRECTION and r.old_value is not None]
            if not corr:
                continue
            per[ev.subject]["t" if tl[ev.t]["correct_true"] else "f"].append(any(r.accepted for r in corr))
    rng = np.random.default_rng(seed)
    subs = sorted(per)
    k = len(subs)
    gaps = []
    for _ in range(n):
        pick = [subs[j] for j in rng.integers(0, k, size=k)]
        t = [x for s in pick for x in per[s]["t"]]
        f = [x for s in pick for x in per[s]["f"]]
        gaps.append((np.mean(t) if t else np.nan) - (np.mean(f) if f else np.nan))
    return _ci(gaps), k


def auc_fact_bootstrap(case_events, world, n=protocol.BOOTSTRAP_N, seed=protocol.SEED_BOOTSTRAP):
    fid, _ = _by_subject(case_events, world)
    rng = np.random.default_rng(seed)
    subs = sorted(fid)
    k = len(subs)
    aucs = []
    for _ in range(n):
        pick = [subs[j] for j in rng.integers(0, k, size=k)]
        aucs.append(_auc([r for s in pick for r in fid.get(s, [])]))
    return _ci(aucs), k


def main() -> int:
    world = WorldB()
    rules = world.public_rules()
    docs = world.retrieval_docs(seed=protocol.MASTER_SEED)
    instances, labels_rows, _ = load("stageb-v1.0", verify=True)
    labels = {r["case_id"]: r["turn_labels"] for r in labels_rows}
    model = LLMBaseModel(world.subjects, seed=protocol.MASTER_SEED,
                         cache_path=ROOT / "results" / "stageb" / "model_cache.json")
    rows: list[list] = []

    n_sub = len({s for (s, _a) in docs})
    stale = sum(1 for (s, a), v in docs.items() if v != world.truth(s, a))
    rows.append(["retrieval_store", "realized", "subjects_covered", n_sub, len(world.subjects), ""])
    rows.append(["retrieval_store", "realized", "stale_entries", stale, len(docs), ""])

    for stage, mode in (("stageb", "combined"), ("stagec", "consistency")):
        full = _run(instances, "full", mode, model, docs, rules)
        t_acc, f_acc = correction_scores(full, labels)
        rows.append([stage, "corrections", "scorer_true_accept_rate", f"{t_acc:.4f}", "", ""])
        rows.append([stage, "corrections", "scorer_false_accept_rate", f"{f_acc:.4f}", "", ""])
        dec = correction_decomposition(full, labels)
        for lab in ("true", "false"):
            for sl in ("unheld", "held"):
                acc = dec[(lab, sl, True)]
                tot = acc + dec[(lab, sl, False)]
                rows.append([stage, "corrections", f"{lab}_{sl}_accepted", acc, tot,
                             f"{acc / tot:.4f}" if tot else ""])
            rows.append([stage, "corrections", f"{lab}_agreeing_dropped", dec[(lab, "agreeing", False)], "", ""])
        (lo, hi), k = held_gap_fact_bootstrap(full, labels)
        held_t = dec[("true", "held", True)] / (dec[("true", "held", True)] + dec[("true", "held", False)])
        held_f = dec[("false", "held", True)] / (dec[("false", "held", True)] + dec[("false", "held", False)])
        rows.append([stage, "corrections", "held_gap_true_minus_false", f"{held_t - held_f:.4f}",
                     f"[{lo:.4f}, {hi:.4f}]", f"fact-clustered 95% CI, {k} subjects"])

        acc_rev = Counter()
        for _cid, events in full:
            for ev in events:
                for r in ev.revision_events:
                    if r.accepted:
                        acc_rev[(r.trigger.name, "unheld" if r.old_value is None else "held")] += 1
        for (trig, sl), n in sorted(acc_rev.items()):
            rows.append([stage, "accepted_revisions", f"{trig}_{sl}", n, "", ""])
        rendered = flagged = never_uttered = 0
        for _cid, events in full:
            for k_, ev in enumerate(events):
                if not ev.acknowledged:
                    continue
                if ev.turn_type == "CORRECT":
                    rendered += 1
                    old = re.search(r"I said (.*?), it is", ev.expressed_value).group(1)
                    if not any(e.subject == ev.subject and e.attribute == ev.attribute
                               and e.expressed_value and old in e.expressed_value for e in events[:k_]):
                        never_uttered += 1
                else:
                    flagged += 1
        rows.append([stage, "acknowledgments", "rendered_admissions_on_correct_turns", rendered, "", ""])
        rows.append([stage, "acknowledgments", "flag_only_on_ask_turns", flagged, "", ""])
        rows.append([stage, "acknowledgments", "rendered_admissions_prior_value_never_uttered", never_uttered, rendered, ""])
        hh = Counter(ev.provenance for _c, evs in full for ev in evs
                     if ev.turn_type in ("ASK", "REASK") and ev.category == "HEDGE_HIGH")
        for prov, n in sorted(hh.items()):
            rows.append([stage, "hedge_high_by_provenance", prov, n, "", ""])
        ret = Counter(ev.category for _c, evs in full for ev in evs
                      if ev.turn_type in ("ASK", "REASK") and ev.provenance == "RETRIEVED")
        for cat, n in sorted(ret.items()):
            rows.append([stage, "retrieved_by_category", cat, n, "", ""])

    logit = _run(instances, "full", "logit", model, docs, rules)
    est = expression_auc(logit, world)
    (lo, hi), k = auc_fact_bootstrap(logit, world)
    rows.append(["stageb", "discrimination", "full_logit_auc", f"{est:.4f}", f"[{lo:.4f}, {hi:.4f}]",
                 f"fact-clustered 95% CI, {k} subjects"])

    # floor redundancy: the full configuration with three_state=False (everything
    # else identical) must reproduce every Stage-B full-arm event
    from truthllm.arms import ArmConfig
    i = ARM_ORDER.index("full")
    rng = np.random.default_rng(protocol.SEED_ARM_BASE + i)
    nofloor = [(inst["case_id"], run_conversation(inst, ArmConfig("full", "table", False, True, True, True),
                                                  model, docs, rng, extractor_mode="combined", rules=rules))
               for inst in instances]

    fullb = _run(instances, "full", "combined", model, docs, rules)
    # dataclass equality: every TurnEvent field, including nested revision events
    rows.append(["stageb", "floor_redundancy", "full_three_state_off_identical_events",
                 str(fullb == nofloor), "", ""])
    # independent retrieval-sampling streams: committed full vs store_no_ack logs
    import json

    def _load(p):
        return {(e["case_id"], e["t"]): e for e in map(json.loads, open(p))}
    a = _load(ROOT / "results" / "stageb" / "events" / "full.jsonl")
    b = _load(ROOT / "results" / "stageb" / "events" / "store_no_ack.jsonl")
    asks = [k for k in a if a[k]["turn_type"] in ("ASK", "REASK")]
    diff = sum(1 for k in asks
               if (a[k]["category"], a[k]["expressed_value"]) != (b[k]["category"], b[k]["expressed_value"]))
    rows.append(["stageb", "arm_seed_streams", "full_vs_store_no_ack_ask_turn_output_differences",
                 diff, len(asks), ""])

    out = ROOT / "results" / "stageb" / "sensitivity_post_review.csv"
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "block", "quantity", "value", "denominator_or_ci", "note"])
        w.writerows(rows)
    print(f"wrote {out}")
    for r in rows:
        print(*r, sep=" | ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
