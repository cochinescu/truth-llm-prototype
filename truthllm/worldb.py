"""Stage-B world: real, stable geography facts.

Same interface as world.World, but facts are real (country -> capital), so a
real base model has genuine parametric knowledge with a natural error/uncertainty
pattern (easy capitals through hard cases like Dodoma, Gitega, Yamoussoukro,
Naypyidaw, Astana). The derived attribute is "continent", resolved via the
public rule table capital-city -> continent (mirrors Stage A's city -> region).

The fact table is the scoring ground truth; entries were chosen for
unambiguity (single official capital, stable for years). Ambiguous cases
(Bolivia, Sri Lanka, Nauru) are deliberately excluded.
"""

from __future__ import annotations

import numpy as np

from . import protocol
from .world import PublicRules

# (country, capital, continent-of-capital)
FACTS = [
    ("France", "Paris", "Europe"), ("Japan", "Tokyo", "Asia"),
    ("Italy", "Rome", "Europe"), ("Spain", "Madrid", "Europe"),
    ("Germany", "Berlin", "Europe"), ("Russia", "Moscow", "Europe"),
    ("China", "Beijing", "Asia"), ("Egypt", "Cairo", "Africa"),
    ("Greece", "Athens", "Europe"), ("Portugal", "Lisbon", "Europe"),
    ("Austria", "Vienna", "Europe"), ("Norway", "Oslo", "Europe"),
    ("Sweden", "Stockholm", "Europe"), ("Finland", "Helsinki", "Europe"),
    ("Ireland", "Dublin", "Europe"), ("Canada", "Ottawa", "North America"),
    ("Australia", "Canberra", "Oceania"), ("Turkey", "Ankara", "Asia"),
    ("Switzerland", "Bern", "Europe"), ("Brazil", "Brasilia", "South America"),
    ("Morocco", "Rabat", "Africa"), ("Nigeria", "Abuja", "Africa"),
    ("Kenya", "Nairobi", "Africa"), ("Ethiopia", "Addis Ababa", "Africa"),
    ("Vietnam", "Hanoi", "Asia"), ("Thailand", "Bangkok", "Asia"),
    ("Indonesia", "Jakarta", "Asia"), ("Philippines", "Manila", "Asia"),
    ("Pakistan", "Islamabad", "Asia"), ("Iran", "Tehran", "Asia"),
    ("Iraq", "Baghdad", "Asia"), ("Syria", "Damascus", "Asia"),
    ("Jordan", "Amman", "Asia"), ("Lebanon", "Beirut", "Asia"),
    ("Cuba", "Havana", "North America"), ("Peru", "Lima", "South America"),
    ("Chile", "Santiago", "South America"), ("Colombia", "Bogota", "South America"),
    ("Venezuela", "Caracas", "South America"), ("Uruguay", "Montevideo", "South America"),
    ("Paraguay", "Asuncion", "South America"), ("Ecuador", "Quito", "South America"),
    ("Ghana", "Accra", "Africa"), ("Senegal", "Dakar", "Africa"),
    ("Tunisia", "Tunis", "Africa"), ("Myanmar", "Naypyidaw", "Asia"),
    ("Tanzania", "Dodoma", "Africa"), ("Ivory Coast", "Yamoussoukro", "Africa"),
    ("Kazakhstan", "Astana", "Asia"), ("Burundi", "Gitega", "Africa"),
    ("Bhutan", "Thimphu", "Asia"), ("Comoros", "Moroni", "Africa"),
    ("Palau", "Ngerulmud", "Oceania"), ("Tuvalu", "Funafuti", "Oceania"),
    ("Vanuatu", "Port Vila", "Oceania"), ("Lesotho", "Maseru", "Africa"),
    ("Eswatini", "Mbabane", "Africa"), ("Djibouti", "Djibouti", "Africa"),
    ("Moldova", "Chisinau", "Europe"), ("Mongolia", "Ulaanbaatar", "Asia"),
]

CAPITALS = [c for _, c, _ in FACTS]
CONTINENTS = sorted({k for _, _, k in FACTS})
CONTINENT_OF_CAPITAL = {c: k for _, c, k in FACTS}


class WorldB:
    tellable_attrs = ("capital",)
    askable_attrs = ("capital", "continent")
    values = {"capital": CAPITALS, "continent": CONTINENTS}

    def __init__(self) -> None:
        self.subjects = [s for s, _, _ in FACTS]
        self.facts = {(s, "capital"): c for s, c, _ in FACTS}

    def truth(self, subject: str, attribute: str) -> str | None:
        if attribute == "continent":
            cap = self.facts.get((subject, "capital"))
            return CONTINENT_OF_CAPITAL[cap] if cap else None
        return self.facts.get((subject, attribute))

    def public_rules(self) -> PublicRules:
        return PublicRules("continent", "capital", dict(CONTINENT_OF_CAPITAL))

    def retrieval_docs(
        self,
        coverage: float = protocol.RETRIEVAL_COVERAGE,
        stale_rate: float = protocol.RETRIEVAL_STALE,
        seed: int = protocol.SEED_WORLD,
    ) -> dict[tuple[str, str], str]:
        rng = np.random.default_rng(seed + 7)
        docs: dict[tuple[str, str], str] = {}
        for s in self.subjects:
            if rng.random() < coverage:
                true_cap = self.facts[(s, "capital")]
                if rng.random() < stale_rate:
                    others = [c for c in CAPITALS if c != true_cap]
                    docs[(s, "capital")] = others[rng.integers(len(others))]
                else:
                    docs[(s, "capital")] = true_cap
        return docs
