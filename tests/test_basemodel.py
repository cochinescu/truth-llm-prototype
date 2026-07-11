import numpy as np

from truthllm import protocol
from truthllm.basemodel import PARAMETRIC_ATTRS, SyntheticBaseModel
from truthllm.world import World


def _model(seed=11):
    w = World(seed=seed)
    return w, SyntheticBaseModel(w, seed=seed)


def test_mix_within_tolerance():
    w, m = _model()
    n = len(w.subjects) * len(PARAMETRIC_ATTRS)
    for status, rate in protocol.KNOWLEDGE_MIX.items():
        got = sum(v == status for v in m.status.values()) / n
        assert abs(got - rate) <= 0.08, (status, got, rate)


def test_absent_facts_may_hallucinate_with_moderate_reliability():
    w, m = _model()
    absent = [k for k, v in m.status.items() if v == "ABSENT"]
    halluc = [k for k in absent if k in m.held]
    assert halluc, "hallucination path never taken"
    lo, hi = protocol.RELIABILITY_HALLUC
    for k in halluc:
        assert lo <= m.held[k].latent_r <= hi
    assert any(k not in m.held for k in absent), "some ABSENT facts stay truly absent"


def test_region_never_parametric_and_queries_pure():
    w, m = _model()
    for s in w.subjects[:5]:
        assert m.query(s, "region") is None
        assert m.query(s, "city") == m.query(s, "city")  # pure lookup


def test_sample_agreement_tracks_reliability():
    w, m = _model()
    rng = np.random.default_rng(0)
    key = max((k for k in m.held), key=lambda k: m.held[k].latent_r)
    ans = m.held[key]
    agree = np.mean([m.sample(*key, rng) == ans.value for _ in range(300)])
    assert abs(agree - ans.latent_r) < 0.1
