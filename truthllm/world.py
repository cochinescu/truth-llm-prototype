"""Synthetic world: the ground-truth fact table and the public derivation rule.

The world is the generator's record — labels and correctness scoring come from
here. The LAYER never reads world truth (anti-circularity, PROTOCOL D4); only
the benchmark generator and the metrics do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import protocol

COLORS = ["red", "blue", "green", "amber", "violet", "teal", "coral", "slate"]
CITIES = [
    "arlon", "brivec", "corda", "duvane", "elmet", "farrow",
    "gilder", "havre", "istra", "jorvik", "kessel", "lorne",
]
REGIONS = ["north", "south", "east", "west"]
# Public rule table: part of the world spec, given to the layer (INFERRED path).
REGION_OF = {city: REGIONS[i % len(REGIONS)] for i, city in enumerate(CITIES)}

ATTRIBUTES = ["color", "city", "region"]  # region is derived, never stored directly
VALUES = {"color": COLORS, "city": CITIES, "region": REGIONS}


@dataclass(frozen=True)
class PublicRules:
    """The derivation rule handed to the LAYER (public world spec — not truth).

    Keeping this a separate object preserves anti-circularity: the pipeline
    receives rules and retrieval docs, never the World object itself.
    """
    derived_attr: str
    base_attr: str
    mapping: dict


class World:
    tellable_attrs = ("color", "city")
    askable_attrs = ("color", "city", "region")
    values = VALUES

    def __init__(self, n_subjects: int = protocol.N_SUBJECTS, seed: int = protocol.SEED_WORLD):
        rng = np.random.default_rng(seed)
        self.subjects = [f"e{i:02d}" for i in range(n_subjects)]
        self.facts: dict[tuple[str, str], str] = {}
        for s in self.subjects:
            self.facts[(s, "color")] = COLORS[rng.integers(len(COLORS))]
            self.facts[(s, "city")] = CITIES[rng.integers(len(CITIES))]

    def public_rules(self) -> PublicRules:
        return PublicRules("region", "city", dict(REGION_OF))

    def truth(self, subject: str, attribute: str) -> str | None:
        if attribute == "region":
            city = self.facts.get((subject, "city"))
            return REGION_OF[city] if city else None
        return self.facts.get((subject, attribute))

    def retrieval_docs(
        self,
        coverage: float = protocol.RETRIEVAL_COVERAGE,
        stale_rate: float = protocol.RETRIEVAL_STALE,
        seed: int = protocol.SEED_WORLD,
    ) -> dict[tuple[str, str], str]:
        """City facts available as 'documents'; a stale fraction are wrong."""
        rng = np.random.default_rng(seed + 7)
        docs: dict[tuple[str, str], str] = {}
        for s in self.subjects:
            if rng.random() < coverage:
                true_city = self.facts[(s, "city")]
                if rng.random() < stale_rate:
                    others = [c for c in CITIES if c != true_city]
                    docs[(s, "city")] = others[rng.integers(len(others))]
                else:
                    docs[(s, "city")] = true_city
        return docs
