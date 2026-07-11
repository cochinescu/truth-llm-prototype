from truthllm import protocol
from truthllm.expression import express
from truthllm.state import Claim, Provenance, assign_state

BAND_REP = {"LOW": 0.2, "MID": 0.5, "HIGH": 0.9}


def _claim(prov, conf):
    return Claim("e00", "color", "red", Provenance[prov], {"signal": conf})


def test_every_gating_cell_matches_the_frozen_table():
    for prov, row in protocol.GATING_TABLE.items():
        for b, want in row.items():
            claim = _claim(prov, BAND_REP[b])
            got = express(claim, assign_state(claim), "table", three_state=True)
            assert got.category == want, (prov, b, got.category, want)


def test_ablation_map_delta_cells_only():
    """The confmap differs from the table exactly where provenance matters."""
    deltas = []
    for prov, row in protocol.GATING_TABLE.items():
        for b in row:
            claim = _claim(prov, BAND_REP[b])
            state = assign_state(claim)
            table = express(claim, state, "table", True).category
            confmap = express(claim, state, "confmap", True).category
            if table != confmap:
                deltas.append((prov, b))
    # RETRIEVED coincides with the ablation map by design (retrieval-grounded
    # claims are the reference case); TOLD/LOW likewise. The deltas are where
    # provenance information changes behavior:
    expected = {("PARAMETRIC", "LOW"), ("PARAMETRIC", "MID"),
                ("INFERRED", "LOW"), ("INFERRED", "MID"), ("INFERRED", "HIGH"),
                ("TOLD", "HIGH")}
    assert set(deltas) == expected, set(deltas) ^ expected


def test_uniform_asserts_and_always_hedged_hedges():
    claim = _claim("PARAMETRIC", 0.2)
    state = assign_state(claim)
    assert express(claim, state, "uniform", False).category == "ASSERT"
    assert express(claim, state, "always_hedged", False).category == "HEDGE_LOW"


def test_no_candidate_always_declines():
    for gating in ["table", "confmap", "uniform", "always_hedged"]:
        got = express(None, assign_state(None), gating, three_state=False)
        assert got.category == "DECLINE" and got.value is None


def test_three_state_floor_declines_only_when_enabled():
    claim = _claim("RETRIEVED", protocol.NOT_KNOWING_FLOOR - 0.01)
    state = assign_state(claim)
    assert express(claim, state, "table", True).category == "DECLINE"
    assert express(claim, state, "confmap", False).category == "HEDGE_LOW"
