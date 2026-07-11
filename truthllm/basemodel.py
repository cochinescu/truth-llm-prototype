"""LLM-free synthetic base model: a seeded, corrupted copy of the world.

Per known fact one of CORRECT / WRONG / UNCERTAIN / ABSENT with a latent
reliability r. ABSENT facts may be held as confident guesses (the hallucination
path C4 gates). All corruption is drawn once at init so queries are pure
lookups and the whole model state is reproducible from its seed.

The model knows color and city only; region is never parametric, so the
inference path (city -> region via the public rule) is always exercised.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import protocol
from .world import VALUES, World

PARAMETRIC_ATTRS = ("color", "city")


@dataclass(frozen=True)
class Answer:
    value: str
    latent_r: float  # ground-truth reliability of this held answer (Stage-A signal source)


class SyntheticBaseModel:
    def __init__(self, world: World, seed: int, mix: dict[str, float] = protocol.KNOWLEDGE_MIX):
        rng = np.random.default_rng(seed)
        self.held: dict[tuple[str, str], Answer] = {}
        self.status: dict[tuple[str, str], str] = {}
        statuses = list(mix.keys())
        probs = np.array([mix[k] for k in statuses])
        for s in world.subjects:
            for a in PARAMETRIC_ATTRS:
                truth = world.facts[(s, a)]
                status = statuses[rng.choice(len(statuses), p=probs)]
                self.status[(s, a)] = status
                if status == "CORRECT":
                    r = rng.uniform(*protocol.RELIABILITY_HIGH)
                    self.held[(s, a)] = Answer(truth, r)
                elif status == "WRONG":
                    r = rng.uniform(*protocol.RELIABILITY_HIGH)
                    others = [v for v in VALUES[a] if v != truth]
                    self.held[(s, a)] = Answer(others[rng.integers(len(others))], r)
                elif status == "UNCERTAIN":
                    r = rng.uniform(*protocol.RELIABILITY_LOW)
                    if rng.random() < 0.5:
                        self.held[(s, a)] = Answer(truth, r)
                    else:
                        others = [v for v in VALUES[a] if v != truth]
                        self.held[(s, a)] = Answer(others[rng.integers(len(others))], r)
                else:  # ABSENT — maybe held as a confident guess (hallucination)
                    if rng.random() < protocol.HALLUCINATION_RATE:
                        r = rng.uniform(*protocol.RELIABILITY_HALLUC)
                        vals = VALUES[a]
                        self.held[(s, a)] = Answer(vals[rng.integers(len(vals))], r)

    def query(self, subject: str, attribute: str) -> Answer | None:
        """Pure lookup; None means the model produces no candidate at all."""
        if attribute not in PARAMETRIC_ATTRS:
            return None
        return self.held.get((subject, attribute))

    def sample(self, subject: str, attribute: str, rng: np.random.Generator) -> str | None:
        """One stochastic sample: the held value with prob r, else a random value."""
        ans = self.query(subject, attribute)
        if ans is None:
            return None
        if rng.random() < ans.latent_r:
            return ans.value
        vals = VALUES[attribute]
        return vals[rng.integers(len(vals))]
