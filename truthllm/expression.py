"""Expression function: epistemic state -> expression category + rendered text.

The gating table and the no-provenance ablation map are PROTOCOL constants
(FROZEN); this module only applies them per the active arm configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import protocol
from .state import Claim, EpistemicClass, band, combined_confidence

CATEGORIES = ["DECLINE", "HEDGE_LOW", "HEDGE_HIGH", "ASSERT"]  # ordered, FROZEN

_TEMPLATES = {
    "DECLINE": "I don't know.",
    "HEDGE_LOW": "I'm not sure — possibly {value}.",
    "HEDGE_HIGH": "I believe {value}.",
    "ASSERT": "{value}.",
}


@dataclass(frozen=True)
class ExpressionEvent:
    category: str
    text: str
    value: str | None  # None iff DECLINE


def express(claim: Claim | None, state: EpistemicClass, gating: str, three_state: bool) -> ExpressionEvent:
    """gating in {"table", "confmap", "uniform", "always_hedged"} (see arms.py)."""
    if claim is None:
        return ExpressionEvent("DECLINE", _TEMPLATES["DECLINE"], None)
    if three_state and state is EpistemicClass.NOT_KNOWING:
        return ExpressionEvent("DECLINE", _TEMPLATES["DECLINE"], None)

    if gating == "uniform":
        category = "ASSERT"
    elif gating == "always_hedged":
        category = "HEDGE_LOW"
    elif gating == "confmap":
        category = protocol.ABLATION_MAP[band(combined_confidence(claim))]
    elif gating == "table":
        category = protocol.GATING_TABLE[claim.provenance.value][band(combined_confidence(claim))]
    else:
        raise ValueError(f"unknown gating mode {gating!r}")

    if category == "DECLINE":
        return ExpressionEvent("DECLINE", _TEMPLATES["DECLINE"], None)
    return ExpressionEvent(category, _TEMPLATES[category].format(value=claim.value), claim.value)


def acknowledgment_text(subject: str, attribute: str, old: str, new: str) -> str:
    return f"I was wrong about {subject}'s {attribute} — I said {old}, it is {new}."
