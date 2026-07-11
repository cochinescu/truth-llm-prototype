"""The turn loop: run one benchmark conversation through one arm.

Answer resolution order (PROTOCOL: FROZEN): store hit -> parametric ->
retrieval -> inference (region via the public city->region rule) -> absent.
Provenance is assigned by the path that produced the claim — pipeline
instrumentation, never a classifier.

Each turn yields a TurnEvent; the serialized event log is the scorer's input.
The pipeline never consults world truth — correctness is stamped on afterwards
by the metrics layer (anti-circularity).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import protocol
from .arms import ArmConfig
from .expression import ExpressionEvent, acknowledgment_text, express
from .extractors import extract_all
from .state import Claim, EpistemicClass, Provenance, assign_state
from .store import BeliefStore, RevisionEvent
from .world import PublicRules


@dataclass
class TurnEvent:
    t: int
    session: int
    turn_type: str                 # ASK | TELL | CORRECT | REASK
    subject: str
    attribute: str
    resolution: str | None         # store|parametric|retrieval|inference|None
    provenance: str | None
    state: str | None
    confidence: dict[str, float] = field(default_factory=dict)
    category: str | None = None
    expressed_value: str | None = None
    acknowledged: bool = False
    ack_key: tuple[int, str, str] | None = None  # (t, subject, attribute) of the revision event
    revision_events: list[RevisionEvent] = field(default_factory=list)


def _resolve(
    subject: str,
    attribute: str,
    cfg: ArmConfig,
    store: BeliefStore | None,
    model,
    docs: dict[tuple[str, str], str],
    rules: PublicRules,
    rng: np.random.Generator,
    t: int,
    events_out: list[RevisionEvent],
    depth: int = 0,
) -> tuple[Claim | None, str | None]:
    """Returns (claim, resolution_path)."""
    # 1. store hit (+ probabilistic retrieval verification, the RETRIEVAL_CONFLICT source)
    if store is not None:
        belief = store.get(subject, attribute)
        if belief is not None:
            if (subject, attribute) in docs and rng.random() < protocol.RETRIEVAL_RATE:
                ev = store.retrieval_conflict(subject, attribute, docs[(subject, attribute)], t)
                if ev is not None:
                    events_out.append(ev)
                    if ev.accepted:
                        return store.get(subject, attribute).claim, "store"
            return belief.claim, "store"
    # 2. parametric
    answer = model.query(subject, attribute)
    if answer is not None:
        extract = getattr(model, "extract", None)  # Stage-B adapters supply their own
        conf = (extract(subject, attribute, answer, rng) if extract
                else extract_all(model, subject, attribute, answer, rng))
        if _EXTRACTOR_MODE[0] != "combined":
            conf = {_EXTRACTOR_MODE[0]: conf[_EXTRACTOR_MODE[0]]}
        claim = Claim(subject, attribute, answer.value, Provenance.PARAMETRIC, conf)
        # UNSURE parametric answers may be upgraded by retrieval
        if (assign_state(claim) is not EpistemicClass.CONFIDENT
                and (subject, attribute) in docs and rng.random() < protocol.RETRIEVAL_RATE):
            return _retrieved_claim(subject, attribute, docs), "retrieval"
        return claim, "parametric"
    # 3. retrieval
    if (subject, attribute) in docs and rng.random() < protocol.RETRIEVAL_RATE:
        return _retrieved_claim(subject, attribute, docs), "retrieval"
    # 4. inference: derived attribute via the public rule (one level, base first)
    if attribute == rules.derived_attr and depth == 0:
        base_claim, _ = _resolve(subject, rules.base_attr, cfg, store, model, docs,
                                 rules, rng, t, events_out, depth=1)
        if base_claim is not None and base_claim.value in rules.mapping:
            return Claim(subject, rules.derived_attr, rules.mapping[base_claim.value],
                         Provenance.INFERRED, dict(base_claim.confidence)), "inference"
    return None, None


def _retrieved_claim(subject: str, attribute: str, docs: dict[tuple[str, str], str]) -> Claim:
    return Claim(subject, attribute, docs[(subject, attribute)], Provenance.RETRIEVED,
                 {"signal": protocol.RETRIEVED_CONF, "consistency": protocol.RETRIEVED_CONF})


# Extractor mode for the current run: "combined" (min over both — the deployed
# configuration, PROTOCOL D1) or a single extractor name (C2's per-extractor
# robustness runs). Module-level because it is a run-wide setting, not per-turn.
_EXTRACTOR_MODE = ["combined"]


def run_conversation(
    script: dict,
    cfg: ArmConfig,
    model,
    docs: dict[tuple[str, str], str],
    rng: np.random.Generator,
    extractor_mode: str = "combined",
    rules: PublicRules | None = None,
) -> list[TurnEvent]:
    _EXTRACTOR_MODE[0] = extractor_mode
    if rules is None:  # Stage-A default: the synthetic world's city->region rule
        from .world import REGION_OF
        rules = PublicRules("region", "city", dict(REGION_OF))
    store = BeliefStore() if cfg.store else None
    events: list[TurnEvent] = []
    for turn in script["turns"]:
        t, session = turn["t"], turn["session"]
        ttype, s, a = turn["type"], turn["subject"], turn["attribute"]
        ev = TurnEvent(t, session, ttype, s, a, None, None, None)

        if ttype in ("ASK", "REASK"):
            revs: list[RevisionEvent] = []
            claim, path = _resolve(s, a, cfg, store, model, docs, rules, rng, t, revs)
            ev.revision_events = revs
            state = assign_state(claim)
            expr: ExpressionEvent = express(claim, state, cfg.gating, cfg.three_state)
            ev.resolution = path
            ev.provenance = claim.provenance.value if claim else None
            ev.state = state.value
            ev.confidence = dict(claim.confidence) if claim else {}
            ev.category = expr.category
            ev.expressed_value = expr.value
            if revs and cfg.acknowledge and any(r.accepted for r in revs):
                r = next(r for r in revs if r.accepted)
                ev.acknowledged = True
                ev.ack_key = (r.t, r.subject, r.attribute)
            if store is not None and claim is not None and expr.value is not None:
                ins = store.insert(claim, t)
                if ins is not None:
                    ev.revision_events.append(ins)

        elif ttype == "TELL":
            value = turn["value"]
            claim = Claim(s, a, value, Provenance.TOLD,
                          {"signal": protocol.TOLD_CONF, "consistency": protocol.TOLD_CONF})
            if store is not None:
                ins = store.insert(claim, t)
                if ins is not None:
                    ev.revision_events.append(ins)
                    if ins.accepted and cfg.acknowledge:
                        ev.acknowledged = True
                        ev.ack_key = (ins.t, ins.subject, ins.attribute)
            ev.provenance = Provenance.TOLD.value

        elif ttype == "CORRECT":
            value = turn["value"]
            if store is not None:
                corroborated = docs.get((s, a)) == value
                rev = store.correct(s, a, value, corroborated, t)
                if rev.old_value != rev.new_value:  # real disagreement, logged
                    ev.revision_events.append(rev)
                if rev.accepted and cfg.acknowledge and rev.old_value is not None:
                    ev.acknowledged = True
                    ev.ack_key = (rev.t, rev.subject, rev.attribute)
                    ev.category = "ACK"
                    ev.expressed_value = acknowledgment_text(s, a, rev.old_value, rev.new_value)
            # stateless: the correction has nowhere to persist — by design

        events.append(ev)
    return events
