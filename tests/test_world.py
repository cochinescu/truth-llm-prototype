from truthllm.world import REGION_OF, World


def test_seeded_determinism():
    w1, w2 = World(seed=1), World(seed=1)
    assert w1.facts == w2.facts
    assert World(seed=2).facts != w1.facts


def test_region_rule_consistency():
    w = World(seed=3)
    for s in w.subjects:
        assert w.truth(s, "region") == REGION_OF[w.truth(s, "city")]


def test_retrieval_docs_deterministic_and_partially_stale():
    w = World(seed=4)
    d1 = w.retrieval_docs(seed=4)
    d2 = w.retrieval_docs(seed=4)
    assert d1 == d2
    stale = sum(v != w.facts[k] for k, v in d1.items())
    assert 0 < len(d1) < len(w.subjects)  # partial coverage
    assert 0 <= stale < len(d1)  # stale docs are wrong, and a minority
