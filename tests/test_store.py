import numpy as np

from truthllm import protocol
from truthllm.state import Claim, Provenance
from truthllm.store import BeliefStore, RevisionTrigger


def _claim(value="red", conf=0.9, prov=Provenance.PARAMETRIC):
    return Claim("e00", "color", value, prov, {"signal": conf, "consistency": conf})


def test_identical_retell_never_fires_a_trigger():
    st = BeliefStore()
    st.insert(_claim("red"), t=0)
    assert st.insert(_claim("red"), t=1) is None
    assert st.log == []


def test_insert_conflict_resolves_by_confidence():
    st = BeliefStore()
    st.insert(_claim("red", conf=0.5), t=0)
    ev = st.insert(_claim("blue", conf=0.9), t=1)
    assert ev.trigger is RevisionTrigger.INSERT_CONFLICT and ev.accepted
    assert st.get("e00", "color").claim.value == "blue"
    ev2 = st.insert(_claim("green", conf=0.2), t=2)
    assert ev2 is not None and not ev2.accepted
    assert st.get("e00", "color").claim.value == "blue"


def test_acceptance_rule_truth_table():
    # low-confidence stored + uncorroborated -> accept
    st = BeliefStore()
    st.insert(_claim("red", conf=protocol.THETA_ACCEPT - 0.1), t=0)
    assert st.correct("e00", "color", "blue", corroborated=False, t=1).accepted
    # high-confidence stored + uncorroborated -> resist (false-correction defense)
    st = BeliefStore()
    st.insert(_claim("red", conf=protocol.THETA_ACCEPT + 0.1), t=0)
    ev = st.correct("e00", "color", "blue", corroborated=False, t=1)
    assert not ev.accepted
    assert st.get("e00", "color").claim.value == "red"
    # high-confidence stored + corroborated -> accept
    st = BeliefStore()
    st.insert(_claim("red", conf=protocol.THETA_ACCEPT + 0.1), t=0)
    assert st.correct("e00", "color", "blue", corroborated=True, t=1).accepted
    assert st.get("e00", "color").claim.provenance is Provenance.TOLD


def test_retrieval_conflict_theta_rule():
    st = BeliefStore()
    st.insert(_claim("red", conf=0.5), t=0)
    ev = st.retrieval_conflict("e00", "color", "blue", t=1)
    assert ev.trigger is RevisionTrigger.RETRIEVAL_CONFLICT and ev.accepted
    assert st.get("e00", "color").claim.provenance is Provenance.RETRIEVED
    st2 = BeliefStore()
    st2.insert(_claim("red", conf=0.95), t=0)
    assert not st2.retrieval_conflict("e00", "color", "blue", t=1).accepted


def test_audit_invariant_fuzz():
    """Both directions over 200 random operations: accepted events are exactly
    the belief changes; resisted events never change the store."""
    rng = np.random.default_rng(5)
    st = BeliefStore()
    values = ["red", "blue", "green"]
    snapshot = {}
    for t in range(200):
        v = values[rng.integers(3)]
        conf = float(rng.uniform(0.1, 1.0))
        op = rng.integers(3)
        before = st.get("e00", "color")
        before_v = before.claim.value if before else None
        if op == 0:
            st.insert(_claim(v, conf=conf), t=t)
        elif op == 1:
            st.correct("e00", "color", v, corroborated=bool(rng.random() < 0.3), t=t)
        else:
            st.retrieval_conflict("e00", "color", v, t=t)
        after = st.get("e00", "color")
        after_v = after.claim.value if after else None
        changed = before_v is not None and after_v != before_v
        new_events = [e for e in st.log if e.t == t]
        if changed:
            assert any(e.accepted for e in new_events), f"unlogged change at t={t}"
        for e in new_events:
            if not e.accepted:
                snapshot[t] = (before_v, after_v)
    # resisted-only turns never changed the store value
    for t, (b, a) in snapshot.items():
        events_t = [e for e in st.log if e.t == t]
        if not any(e.accepted for e in events_t):
            assert b == a


def test_worked_example_four_turns():
    """The plan's §3 worked example: ASK -> TELL(conflict, accepted) ->
    re-ASK (revised, consistent) -> ASK absent (decline path upstream)."""
    from truthllm.expression import express
    from truthllm.state import assign_state

    st = BeliefStore()
    # T1: ASK — parametric HIGH claim, asserted, stored
    c1 = _claim("red", conf=0.9)
    s1 = assign_state(c1)
    e1 = express(c1, s1, "table", True)
    assert (s1.value, e1.category) == ("CONFIDENT", "ASSERT")
    st.insert(c1, t=1)
    # T2: TELL conflicting with low enough stored conf? stored is 0.9 -> resist path
    # via correction without corroboration:
    ev2 = st.correct("e00", "color", "blue", corroborated=False, t=2)
    assert not ev2.accepted  # humility is not sycophancy
    # corroborated correction is accepted and acknowledged
    ev2b = st.correct("e00", "color", "blue", corroborated=True, t=3)
    assert ev2b.accepted and ev2b.old_value == "red"
    # T3: re-ASK — the store now answers "blue": no self-contradiction
    assert st.get("e00", "color").claim.value == "blue"
    assert st.get("e00", "color").claim.provenance is Provenance.TOLD
    # T4: ASK about an unheld fact — NOT_KNOWING -> DECLINE
    e4 = express(None, assign_state(None), "table", True)
    assert e4.category == "DECLINE"
    # audit: exactly one accepted revision, traceable
    accepted = [e for e in st.log if e.accepted]
    assert len(accepted) == 1 and accepted[0].t == 3
