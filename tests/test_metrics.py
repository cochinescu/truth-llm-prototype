import numpy as np

from truthllm import protocol
from truthllm.metrics import (_ece_from_records, ack_audit, cluster_bootstrap,
                              contradiction_rate)
from truthllm.pipeline import TurnEvent
from truthllm.store import RevisionEvent, RevisionTrigger


def _ask(t, s="e00", a="color", value="red", category="ASSERT", ttype="ASK"):
    ev = TurnEvent(t, 0, ttype, s, a, "parametric", "PARAMETRIC", "CONFIDENT")
    ev.category = category
    ev.expressed_value = value
    return ev


def test_ece_fixture_exact():
    # 4 ASSERTs (3 correct) + 2 HEDGE_LOWs (1 correct)
    recs = [("ASSERT", True)] * 3 + [("ASSERT", False)] + [("HEDGE_LOW", True), ("HEDGE_LOW", False)]
    ece, bins = _ece_from_records(recs)
    want = (4 / 6) * abs(0.75 - protocol.NOMINAL_CONF["ASSERT"]) \
         + (2 / 6) * abs(0.5 - protocol.NOMINAL_CONF["HEDGE_LOW"])
    assert abs(ece - want) < 1e-12
    assert bins["ASSERT"][2] == 4 and bins["HEDGE_LOW"][2] == 2


def test_contradiction_fixture():
    # say red at t=0, flip to blue at t=2 with no revision -> 1 contradiction
    flip = _ask(2, value="blue", ttype="REASK")
    assert contradiction_rate([("c0", [_ask(0), flip])]) == 1.0
    # same flip legitimized by an accepted revision at t=1 -> 0
    rev = RevisionEvent(1, "e00", "color", "red", "blue", RevisionTrigger.USER_CORRECTION, True)
    tell = TurnEvent(1, 0, "CORRECT", "e00", "color", None, None, None)
    tell.revision_events = [rev]
    assert contradiction_rate([("c0", [_ask(0), tell, _ask(2, value="blue", ttype="REASK")])]) == 0.0
    # consistent reask -> 0 over 1 opportunity
    assert contradiction_rate([("c0", [_ask(0), _ask(2, ttype="REASK")])]) == 0.0


def test_ack_audit_traces():
    rev = RevisionEvent(1, "e00", "color", "red", "blue", RevisionTrigger.USER_CORRECTION, True)
    ev = TurnEvent(1, 0, "CORRECT", "e00", "color", None, None, None)
    ev.revision_events = [rev]
    ev.acknowledged = True
    ev.ack_key = (1, "e00", "color")
    assert ack_audit([("c0", [ev])]) == (1, 1)
    ev_bad = TurnEvent(2, 0, "CORRECT", "e00", "color", None, None, None)
    ev_bad.acknowledged = True
    ev_bad.ack_key = (99, "e00", "color")  # untraceable
    assert ack_audit([("c0", [ev, ev_bad])]) == (2, 1)


def test_bootstrap_seeded_and_brackets_estimate():
    cases = [(f"c{i}", [_ask(0, value="red" if i % 2 else "blue", ttype="ASK"),
                        _ask(2, value="red", ttype="REASK")]) for i in range(20)]
    est, lo, hi = cluster_bootstrap(cases, contradiction_rate, n=200, seed=7)
    est2, lo2, hi2 = cluster_bootstrap(cases, contradiction_rate, n=200, seed=7)
    assert (est, lo, hi) == (est2, lo2, hi2)
    assert lo <= est <= hi
