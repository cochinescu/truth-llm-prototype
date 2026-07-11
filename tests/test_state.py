import pytest

from truthllm import protocol
from truthllm.state import (Claim, EpistemicClass, Provenance, assign_state,
                            band, combined_confidence)


def _claim(**conf):
    return Claim("e00", "color", "red", Provenance.PARAMETRIC, conf)


def test_band_edges_exact():
    assert band(protocol.CONF_LOW - 1e-9) == "LOW"
    assert band(protocol.CONF_LOW) == "MID"
    assert band(protocol.CONF_HIGH - 1e-9) == "MID"
    assert band(protocol.CONF_HIGH) == "HIGH"


def test_min_combination_rule():
    assert combined_confidence(_claim(signal=0.9, consistency=0.3)) == 0.3


def test_three_state_assignment():
    assert assign_state(None) is EpistemicClass.NOT_KNOWING
    assert assign_state(_claim(signal=protocol.NOT_KNOWING_FLOOR - 0.01)) is EpistemicClass.NOT_KNOWING
    assert assign_state(_claim(signal=0.5)) is EpistemicClass.UNSURE
    assert assign_state(_claim(signal=0.9)) is EpistemicClass.CONFIDENT


def test_assign_state_can_never_return_wrong_revising():
    for c in [0.0, 0.1, 0.4, 0.75, 1.0]:
        assert assign_state(_claim(signal=c)) is not EpistemicClass.WRONG_REVISING
    with pytest.raises(Exception):
        # WRONG_REVISING has no assignment path from confidence at all
        assign_state("not a claim")  # type: ignore[arg-type]
