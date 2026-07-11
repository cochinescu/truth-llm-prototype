"""Fact-clustered (subject-level) sensitivity analysis for the Stage-C gates.

Review finding (final round): the pre-registered statistical unit is the
conversation, but per-fact model caching makes observations dependent at the
fact level — 120 conversations recycle the same 60 cached facts. This script
recomputes the two delivery gates (manipulation AUC; capability-equivalence
delta) with a subject-level cluster bootstrap, on both the frozen Stage-C
benchmark and the fresh robustness draw, and writes
results/stagec/sensitivity_fact_cluster.csv. Also emits conversation-cluster
CIs for the correction-acceptance rates (reported without intervals before).

The frozen verdicts are NOT recomputed or relabeled; this is a disclosed
post-hoc sensitivity analysis.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.build_benchmark import generate, load  # noqa: E402
from truthllm import protocol  # noqa: E402
from truthllm.arms import ARM_ORDER, ARMS  # noqa: E402
from truthllm.llm_model import LLMBaseModel  # noqa: E402
from truthllm.metrics import _CAT_RANK, correction_scores  # noqa: E402
from truthllm.pipeline import run_conversation  # noqa: E402
from truthllm.worldb import WorldB  # noqa: E402

FRESH_SEED = 20260715  # PROTOCOL robustness-draw seed


def _run_arm(instances, arm, model, docs, rules):
    i = ARM_ORDER.index(arm)
    rng = np.random.default_rng(protocol.SEED_ARM_BASE + i + 1000 * 2)  # consistency mode
    return [(inst["case_id"],
             run_conversation(inst, ARMS[arm], model, docs, rng,
                              extractor_mode="consistency", rules=rules))
            for inst in instances]


def _by_subject(case_events, world):
    fid, acc = defaultdict(list), defaultdict(list)
    for _, events in case_events:
        for ev in events:
            if ev.turn_type in ("ASK", "REASK") and ev.expressed_value is not None:
                ok = ev.expressed_value == world.truth(ev.subject, ev.attribute)
                acc[ev.subject].append(ok)
                if ev.category in _CAT_RANK:
                    fid[ev.subject].append((_CAT_RANK[ev.category], ok))
    return fid, acc


def _auc(recs):
    cor = [r for r, ok in recs if ok]
    inc = [r for r, ok in recs if not ok]
    if not cor or not inc:
        return float("nan")
    ci = Counter(inc)
    w = t = 0
    for rc in cor:
        for ri, n in ci.items():
            if rc > ri:
                w += n
            elif rc == ri:
                t += n
    return (w + 0.5 * t) / (len(cor) * len(inc))


def _ci(a):
    a = np.sort(np.asarray([x for x in a if x == x]))
    return float(a[int(0.025 * len(a))]), float(a[min(int(0.975 * len(a)), len(a) - 1)])


def subject_bootstrap(fid_full, acc_full, acc_uni, n=protocol.BOOTSTRAP_N,
                      seed=protocol.SEED_BOOTSTRAP):
    rng = np.random.default_rng(seed)
    subs = sorted(set(fid_full) | set(acc_full) | set(acc_uni))
    k = len(subs)
    aucs, deltas = [], []
    for _ in range(n):
        pick = [subs[j] for j in rng.integers(0, k, size=k)]
        aucs.append(_auc([r for s in pick for r in fid_full.get(s, [])]))
        af = [x for s in pick for x in acc_full.get(s, [])]
        au = [x for s in pick for x in acc_uni.get(s, [])]
        deltas.append((np.mean(af) if af else np.nan) - (np.mean(au) if au else np.nan))
    return _ci(aucs), _ci(deltas)


def conversation_bootstrap_corrections(case_events, labels, n=protocol.BOOTSTRAP_N,
                                       seed=protocol.SEED_BOOTSTRAP):
    rng = np.random.default_rng(seed)
    k = len(case_events)
    ts, fs = [], []
    for _ in range(n):
        idx = rng.integers(0, k, size=k)
        t, f = correction_scores([case_events[i] for i in idx], labels)
        ts.append(t)
        fs.append(f)
    return _ci(ts), _ci(fs)


def main() -> int:
    world = WorldB()
    rules = world.public_rules()
    docs = world.retrieval_docs(seed=protocol.MASTER_SEED)
    model = LLMBaseModel(world.subjects, seed=protocol.MASTER_SEED,
                         cache_path=ROOT / "results" / "stageb" / "model_cache.json")

    frozen_inst, frozen_lab, _ = load("stageb-v1.0", verify=True)
    fresh_inst, fresh_lab = generate(world, seed=FRESH_SEED)

    rows = [["bench", "quantity", "point", "fact_ci_lo", "fact_ci_hi"]]
    corr_rows = [["bench", "quantity", "point", "conv_ci_lo", "conv_ci_hi"]]
    for name, instances, labels_rows in [("stagec-frozen", frozen_inst, frozen_lab),
                                         ("fresh-20260715", fresh_inst, fresh_lab)]:
        labels = {r["case_id"]: r["turn_labels"] for r in labels_rows}
        full = _run_arm(instances, "full", model, docs, rules)
        uni = _run_arm(instances, "uniform", model, docs, rules)
        fid_f, acc_f = _by_subject(full, world)
        _, acc_u = _by_subject(uni, world)
        point_auc = _auc([r for rs in fid_f.values() for r in rs])
        pa = (np.mean([x for xs in acc_f.values() for x in xs])
              - np.mean([x for xs in acc_u.values() for x in xs]))
        (alo, ahi), (dlo, dhi) = subject_bootstrap(fid_f, acc_f, acc_u)
        rows.append([name, "manipulation_auc_full", f"{point_auc:.4f}", f"{alo:.4f}", f"{ahi:.4f}"])
        rows.append([name, "equivalence_delta", f"{pa:.4f}", f"{dlo:.4f}", f"{dhi:.4f}"])
        t_pt, f_pt = correction_scores(full, labels)
        (tlo, thi), (flo, fhi) = conversation_bootstrap_corrections(full, labels)
        corr_rows.append([name, "true_accept", f"{t_pt:.4f}", f"{tlo:.4f}", f"{thi:.4f}"])
        corr_rows.append([name, "false_accept", f"{f_pt:.4f}", f"{flo:.4f}", f"{fhi:.4f}"])

    out = ROOT / "results" / "stagec"
    with open(out / "sensitivity_fact_cluster.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(out / "correction_rate_cis.csv", "w", newline="") as f:
        csv.writer(f).writerows(corr_rows)
    for r in rows[1:]:
        print(*r)
    for r in corr_rows[1:]:
        print(*r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
