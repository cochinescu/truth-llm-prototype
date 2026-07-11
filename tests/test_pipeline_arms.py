import numpy as np

from benchmark.build_benchmark import generate
from truthllm.arms import ARMS
from truthllm.basemodel import SyntheticBaseModel
from truthllm.pipeline import run_conversation
from truthllm.world import World


def _setup(seed=41, n=10):
    w = World(seed=seed)
    m = SyntheticBaseModel(w, seed=seed)
    docs = w.retrieval_docs(seed=seed)
    instances, labels = generate(w, n_cases=n, seed=seed)
    return w, m, docs, instances, labels


def _run(arm, seed=41, n=10):
    w, m, docs, instances, _ = _setup(seed, n)
    rng = np.random.default_rng(seed)
    return [(i["case_id"], run_conversation(i, ARMS[arm], m, docs, rng)) for i in instances], w


def test_provenance_matches_resolution_path():
    ces, _ = _run("full")
    seen = set()
    for _, events in ces:
        for ev in events:
            if ev.turn_type in ("ASK", "REASK") and ev.resolution:
                seen.add((ev.resolution, ev.provenance))
                if ev.resolution == "parametric":
                    assert ev.provenance == "PARAMETRIC"
                elif ev.resolution == "retrieval":
                    assert ev.provenance == "RETRIEVED"
                elif ev.resolution == "inference":
                    assert ev.provenance == "INFERRED"
    assert {"parametric", "inference"} <= {p for p, _ in seen}


def test_stateless_never_resolves_from_store_and_never_acknowledges():
    ces, _ = _run("stateless")
    for _, events in ces:
        for ev in events:
            assert ev.resolution != "store"
            assert not ev.acknowledged


def test_uniform_asserts_all_content_and_full_declines_some():
    ces_u, _ = _run("uniform")
    cats_u = {ev.category for _, evs in ces_u for ev in evs
              if ev.turn_type in ("ASK", "REASK") and ev.expressed_value is not None}
    assert cats_u == {"ASSERT"}
    ces_f, _ = _run("full")
    cats_f = [ev.category for _, evs in ces_f for ev in evs if ev.turn_type in ("ASK", "REASK")]
    assert "DECLINE" in cats_f and any(c != "DECLINE" for c in cats_f)


def test_store_no_ack_revises_silently():
    ces, _ = _run("store_no_ack")
    accepted = sum(r.accepted for _, evs in ces for ev in evs for r in ev.revision_events)
    acks = sum(ev.acknowledged for _, evs in ces for ev in evs)
    assert accepted > 0 and acks == 0


def test_arm_runs_seeded_deterministic():
    a, _ = _run("full", seed=42)
    b, _ = _run("full", seed=42)
    sig = lambda ces: [(ev.t, ev.category, ev.expressed_value)
                       for _, evs in ces for ev in evs]
    assert sig(a) == sig(b)
