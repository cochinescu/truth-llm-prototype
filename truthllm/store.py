"""Belief store + revision operator + revision log.

The being-wrong state is entered by exactly three triggers (PROTOCOL: FROZEN):
USER_CORRECTION (accepted), INSERT_CONFLICT, RETRIEVAL_CONFLICT. The log is the
C3b audit artifact: every acknowledgment utterance must trace to an accepted
RevisionEvent, and the store enforces the acceptance rule — it never consults
world truth (anti-circularity, PROTOCOL D4).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from . import protocol
from .state import Claim, EpistemicClass, Provenance, combined_confidence


class RevisionTrigger(Enum):
    USER_CORRECTION = "USER_CORRECTION"
    INSERT_CONFLICT = "INSERT_CONFLICT"
    RETRIEVAL_CONFLICT = "RETRIEVAL_CONFLICT"


@dataclass
class Belief:
    claim: Claim
    t_stored: int


@dataclass(frozen=True)
class RevisionEvent:
    t: int
    subject: str
    attribute: str
    old_value: str | None
    new_value: str
    trigger: RevisionTrigger
    accepted: bool  # False = resisted (belief kept); logged either way


class BeliefStore:
    def __init__(self) -> None:
        self._beliefs: dict[tuple[str, str], Belief] = {}
        self.log: list[RevisionEvent] = []

    def get(self, subject: str, attribute: str) -> Belief | None:
        return self._beliefs.get((subject, attribute))

    def insert(self, claim: Claim, t: int) -> RevisionEvent | None:
        """Store a claim; a conflicting existing belief fires INSERT_CONFLICT.

        Resolution: the higher combined confidence wins. Identical values never
        fire a trigger (re-telling the same thing is not a revision).
        """
        key = (claim.subject, claim.attribute)
        existing = self._beliefs.get(key)
        if existing is None or existing.claim.value == claim.value:
            self._beliefs[key] = Belief(claim, t)
            return None
        accepted = combined_confidence(claim) > combined_confidence(existing.claim)
        event = RevisionEvent(
            t, claim.subject, claim.attribute,
            existing.claim.value, claim.value, RevisionTrigger.INSERT_CONFLICT, accepted,
        )
        self.log.append(event)
        if accepted:
            self._beliefs[key] = Belief(claim, t)
        return event

    def correct(
        self, subject: str, attribute: str, new_value: str, corroborated: bool, t: int
    ) -> RevisionEvent:
        """The acceptance rule (PROTOCOL: FROZEN form).

        Accept iff stored confidence < THETA_ACCEPT or the correction is
        corroborated by the retrieval store. A correction of an unheld belief is
        accepted trivially (nothing to defend).
        """
        existing = self._beliefs.get((subject, attribute))
        if existing is not None and existing.claim.value == new_value:
            # agreeing "correction" — nothing to revise, log nothing
            return RevisionEvent(t, subject, attribute, new_value, new_value,
                                 RevisionTrigger.USER_CORRECTION, False)
        if existing is None:
            accepted = True
            old_value = None
        else:
            accepted = combined_confidence(existing.claim) < protocol.THETA_ACCEPT or corroborated
            old_value = existing.claim.value
        event = RevisionEvent(t, subject, attribute, old_value, new_value,
                              RevisionTrigger.USER_CORRECTION, accepted)
        self.log.append(event)
        if accepted:
            claim = Claim(subject, attribute, new_value, Provenance.TOLD,
                          {"signal": protocol.CORRECTION_CONF, "consistency": protocol.CORRECTION_CONF})
            self._beliefs[(subject, attribute)] = Belief(claim, t)
        return event

    def retrieval_conflict(self, subject: str, attribute: str, retrieved_value: str, t: int) -> RevisionEvent | None:
        """A retrieval result contradicting a stored belief; same theta rule."""
        existing = self._beliefs.get((subject, attribute))
        if existing is None or existing.claim.value == retrieved_value:
            return None
        accepted = combined_confidence(existing.claim) < protocol.THETA_ACCEPT
        event = RevisionEvent(t, subject, attribute, existing.claim.value, retrieved_value,
                              RevisionTrigger.RETRIEVAL_CONFLICT, accepted)
        self.log.append(event)
        if accepted:
            claim = Claim(subject, attribute, retrieved_value, Provenance.RETRIEVED,
                          {"signal": protocol.RETRIEVED_CONF, "consistency": protocol.RETRIEVED_CONF})
            self._beliefs[(subject, attribute)] = Belief(claim, t)
        return event

    def accepted_revisions_between(self, subject: str, attribute: str, t_from: int, t_to: int) -> list[RevisionEvent]:
        return [e for e in self.log
                if e.accepted and e.subject == subject and e.attribute == attribute
                and t_from < e.t <= t_to]
