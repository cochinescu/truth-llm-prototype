"""Per-claim epistemic state: (confidence per extractor, provenance) + the
three-state assignment. WRONG_REVISING is entered by store triggers only —
assign_state can never produce it (PROTOCOL: FROZEN)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from . import protocol


class Provenance(Enum):
    PARAMETRIC = "PARAMETRIC"
    RETRIEVED = "RETRIEVED"
    INFERRED = "INFERRED"
    TOLD = "TOLD"


class EpistemicClass(Enum):
    NOT_KNOWING = "NOT_KNOWING"
    UNSURE = "UNSURE"
    CONFIDENT = "CONFIDENT"
    WRONG_REVISING = "WRONG_REVISING"


@dataclass
class Claim:
    subject: str
    attribute: str
    value: str
    provenance: Provenance
    confidence: dict[str, float] = field(default_factory=dict)  # extractor -> [0,1]


def combined_confidence(claim: Claim) -> float:
    """min over extractors — the conservative combination rule (PROTOCOL D1, FROZEN)."""
    return min(claim.confidence.values())


def band(conf: float) -> str:
    if conf < protocol.CONF_LOW:
        return "LOW"
    if conf < protocol.CONF_HIGH:
        return "MID"
    return "HIGH"


def assign_state(claim: Claim | None) -> EpistemicClass:
    if claim is None:
        return EpistemicClass.NOT_KNOWING
    c = combined_confidence(claim)
    if c < protocol.NOT_KNOWING_FLOOR:
        return EpistemicClass.NOT_KNOWING
    if c < protocol.CONF_HIGH:
        return EpistemicClass.UNSURE
    return EpistemicClass.CONFIDENT
