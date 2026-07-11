"""Mechanical scoring. Every metric reads TurnEvents + the world/labels record;
none reads the layer's internals beyond the event log (the scorer contract).

Statistical unit = the conversation; CIs via cluster bootstrap over case ids
(PROTOCOL: FROZEN). All estimators return plain floats/dicts ready for CSV.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from . import protocol
from .pipeline import TurnEvent
from .store import RevisionTrigger
from .world import World

CaseEvents = list[tuple[str, list[TurnEvent]]]  # (case_id, events)


# --- C2: expression fidelity ---------------------------------------------------

def _fidelity_records(case_events: CaseEvents, world: World) -> dict[str, list[tuple[str, bool]]]:
    """case_id -> [(category, correct)] over asserted content (DECLINE/ACK excluded)."""
    out: dict[str, list[tuple[str, bool]]] = {}
    for cid, events in case_events:
        recs = []
        for ev in events:
            if ev.turn_type in ("ASK", "REASK") and ev.expressed_value is not None \
                    and ev.category in protocol.NOMINAL_CONF:
                recs.append((ev.category, ev.expressed_value == world.truth(ev.subject, ev.attribute)))
        out[cid] = recs
    return out


def expression_ece(case_events: CaseEvents, world: World) -> tuple[float, dict[str, tuple[float, float, int]]]:
    """Returns (ece, per_category {cat: (empirical_acc, nominal, n)})."""
    recs = [r for rs in _fidelity_records(case_events, world).values() for r in rs]
    return _ece_from_records(recs)


def _ece_from_records(recs: list[tuple[str, bool]]) -> tuple[float, dict[str, tuple[float, float, int]]]:
    if not recs:
        return float("nan"), {}
    total = len(recs)
    ece = 0.0
    bins: dict[str, tuple[float, float, int]] = {}
    for cat, nominal in protocol.NOMINAL_CONF.items():
        sub = [ok for c, ok in recs if c == cat]
        if not sub:
            continue
        acc = float(np.mean(sub))
        bins[cat] = (acc, nominal, len(sub))
        ece += (len(sub) / total) * abs(acc - nominal)
    return ece, bins


_CAT_RANK = {"HEDGE_LOW": 1, "HEDGE_HIGH": 2, "ASSERT": 3}


def expression_auc(case_events: CaseEvents, world: World) -> float:
    """Amendment 1 (PROTOCOL): expression-discrimination AUC — the probability
    that a uniformly drawn CORRECT expressed claim carries a strictly higher
    expression category than an incorrect one (ties count 1/2). Any
    single-category policy scores exactly 0.5 regardless of corpus accuracy,
    which is what makes this degeneracy-proof where anchor-ECE is not.
    DECLINEs remain excluded (no asserted content; counted in coverage)."""
    recs = [r for rs in _fidelity_records(case_events, world).values() for r in rs]
    correct = [_CAT_RANK[c] for c, ok in recs if ok]
    incorrect = [_CAT_RANK[c] for c, ok in recs if not ok]
    if not correct or not incorrect:
        return float("nan")
    from collections import Counter
    ci = Counter(incorrect)
    wins = ties = 0
    for rc in correct:
        for ri, n in ci.items():
            if rc > ri:
                wins += n
            elif rc == ri:
                ties += n
    return (wins + 0.5 * ties) / (len(correct) * len(incorrect))


# --- C3: consistency, acknowledgment, corrections -------------------------------

def _contradictions(events: list[TurnEvent]) -> tuple[int, int]:
    """(contradictions, opportunities) for one conversation."""
    accepted: list[tuple[int, str, str]] = []
    for ev in events:
        for r in ev.revision_events:
            if r.accepted:
                accepted.append((r.t, r.subject, r.attribute))
    last_said: dict[tuple[str, str], tuple[int, str]] = {}
    contradictions = opportunities = 0
    for ev in events:
        if ev.turn_type not in ("ASK", "REASK") or ev.expressed_value is None:
            continue
        key = (ev.subject, ev.attribute)
        if ev.turn_type == "REASK" and key in last_said:
            t_prev, prev_value = last_said[key]
            opportunities += 1
            if ev.expressed_value != prev_value and not any(
                    s == ev.subject and a == ev.attribute and t_prev < t <= ev.t
                    for t, s, a in accepted):
                contradictions += 1
        last_said[key] = (ev.t, ev.expressed_value)
    return contradictions, opportunities


def contradiction_rate(case_events: CaseEvents) -> float:
    c = sum(_contradictions(evs)[0] for _, evs in case_events)
    o = sum(_contradictions(evs)[1] for _, evs in case_events)
    return c / o if o else float("nan")


def ack_audit(case_events: CaseEvents) -> tuple[int, int]:
    """(n_acknowledgments, n_traced_to_accepted_revision) — C3b soundness
    direction: every utterance traces to an accepted revision."""
    n_acks = n_traced = 0
    for _, events in case_events:
        accepted = {(r.t, r.subject, r.attribute)
                    for ev in events for r in ev.revision_events if r.accepted}
        for ev in events:
            if ev.acknowledged:
                n_acks += 1
                if ev.ack_key in accepted:
                    n_traced += 1
    return n_acks, n_traced


def ack_completeness(case_events: CaseEvents) -> tuple[int, int]:
    """(n_accepted_revisions, n_acknowledged_turns) — the reciprocal
    direction: how many accepted state changes were verbally acknowledged.
    Reported as a fraction, not asserted as 100%: store-maintenance inserts
    that occur after the turn's expression are logged but not uttered."""
    n_accepted = n_acked = 0
    for _, events in case_events:
        for ev in events:
            n_accepted += sum(1 for r in ev.revision_events if r.accepted)
            n_acked += bool(ev.acknowledged)
    return n_accepted, n_acked


def correction_scores(case_events: CaseEvents, labels: dict[str, list[dict]]) -> tuple[float, float]:
    """(true_accept_rate, false_accept_rate) over real-disagreement corrections."""
    true_n = true_acc = false_n = false_acc = 0
    for cid, events in case_events:
        turn_labels = labels[cid]
        for ev in events:
            if ev.turn_type != "CORRECT":
                continue
            lab = turn_labels[ev.t]
            corr_events = [r for r in ev.revision_events
                           if r.trigger is RevisionTrigger.USER_CORRECTION]
            if not corr_events:
                continue  # no stored disagreement (or stateless) — no acceptance decision
            accepted = any(r.accepted for r in corr_events)
            if lab["correct_true"]:
                true_n += 1
                true_acc += accepted
            else:
                false_n += 1
                false_acc += accepted
    return (true_acc / true_n if true_n else float("nan"),
            false_acc / false_n if false_n else float("nan"))


# --- C4: assertions, capability --------------------------------------------------

def assertion_rates(case_events: CaseEvents, world: World) -> dict[str, float]:
    asks = expressed = correct = conf_false = 0
    for _, events in case_events:
        for ev in events:
            if ev.turn_type not in ("ASK", "REASK"):
                continue
            asks += 1
            if ev.expressed_value is None:
                continue
            expressed += 1
            ok = ev.expressed_value == world.truth(ev.subject, ev.attribute)
            correct += ok
            if ev.category == "ASSERT" and not ok:
                conf_false += 1
    return {
        "confidently_false_rate": conf_false / asks if asks else float("nan"),
        "coverage": expressed / asks if asks else float("nan"),
        "answer_accuracy": correct / expressed if expressed else float("nan"),
    }


# --- Overhead --------------------------------------------------------------------

def overhead_ms(fn: Callable[[], object], iters: int = protocol.OVERHEAD_ITERS) -> tuple[float, float, float]:
    """(median_ms, ci_lo, ci_hi) over iters runs; never a single number (PROTOCOL)."""
    times = []
    fn()  # warmup
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.sort(np.asarray(times))
    lo, hi = int(0.025 * len(arr)), min(int(0.975 * len(arr)), len(arr) - 1)
    return float(np.median(arr)), float(arr[lo]), float(arr[hi])


# --- Capability equivalence (TOST-style, PROTOCOL Stage-B) -------------------------

def paired_difference_ci(
    case_events_a: CaseEvents,
    case_events_b: CaseEvents,
    stat: Callable[[CaseEvents], float],
    n: int = protocol.BOOTSTRAP_N,
    seed: int = protocol.SEED_BOOTSTRAP,
) -> tuple[float, float, float]:
    """(estimate, ci_lo, ci_hi) for stat(A) - stat(B), resampling the SAME
    conversation ids in both arms (paired; the equivalence test's engine)."""
    assert [c for c, _ in case_events_a] == [c for c, _ in case_events_b]
    rng = np.random.default_rng(seed)
    est = stat(case_events_a) - stat(case_events_b)
    k = len(case_events_a)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, k, size=k)
        diffs.append(stat([case_events_a[i] for i in idx])
                     - stat([case_events_b[i] for i in idx]))
    arr = np.sort(np.asarray(diffs))
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return est, float("nan"), float("nan")
    lo, hi = int(0.025 * len(arr)), min(int(0.975 * len(arr)), len(arr) - 1)
    return est, float(arr[lo]), float(arr[hi])


# --- Cluster bootstrap -------------------------------------------------------------

def cluster_bootstrap(
    case_events: CaseEvents,
    stat: Callable[[CaseEvents], float],
    n: int = protocol.BOOTSTRAP_N,
    seed: int = protocol.SEED_BOOTSTRAP,
) -> tuple[float, float, float]:
    """(estimate, ci_lo, ci_hi); resamples whole conversations (the statistical unit)."""
    rng = np.random.default_rng(seed)
    est = stat(case_events)
    stats = []
    k = len(case_events)
    for _ in range(n):
        idx = rng.integers(0, k, size=k)
        stats.append(stat([case_events[i] for i in idx]))
    arr = np.sort(np.asarray(stats))
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return est, float("nan"), float("nan")
    lo, hi = int(0.025 * len(arr)), min(int(0.975 * len(arr)), len(arr) - 1)
    return est, float(arr[lo]), float(arr[hi])
