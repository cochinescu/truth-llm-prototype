"""Stage-B components that need no model download: the real-fact world and
the answer normalization/snapping layer."""

from truthllm.llm_model import normalize, snap
from truthllm.worldb import CAPITALS, CONTINENT_OF_CAPITAL, FACTS, WorldB


def test_fact_table_integrity():
    w = WorldB()
    assert len(w.subjects) == 60 and len(set(w.subjects)) == 60
    assert len(set(CAPITALS)) == 60  # capitals unique (Djibouti city == country is fine)
    for s, cap, cont in FACTS:
        assert w.truth(s, "capital") == cap
        assert w.truth(s, "continent") == cont
        assert CONTINENT_OF_CAPITAL[cap] == cont


def test_public_rules_mirror_stage_a_shape():
    r = WorldB().public_rules()
    assert r.derived_attr == "continent" and r.base_attr == "capital"
    assert r.mapping["Paris"] == "Europe" and r.mapping["Dodoma"] == "Africa"


def test_normalize_strips_accents_case_punctuation():
    assert normalize("  Brasília! ") == "brasilia"
    assert normalize("The Hague") == "hague"
    assert normalize("Washington, D.C.") == "washington d c"


def test_snap_to_known_capital_and_declines():
    assert snap("Paris") == "Paris"
    assert snap("The capital is Paris.") == "Paris"
    assert snap("paris, france") == "Paris"
    assert snap("Ulaanbaatar") == "Ulaanbaatar"
    assert snap("I don't know") is None
    assert snap("unknown") is None
    assert snap("") is None
    # unknown place stays as itself (scored false downstream, never snapped to truth)
    assert snap("Gotham City") == "gotham city"


def test_retrieval_docs_partial_and_stale():
    w = WorldB()
    docs = w.retrieval_docs(seed=5)
    assert 0 < len(docs) < 60
    stale = sum(v != w.facts[k] for k, v in docs.items())
    assert 0 <= stale < len(docs)
