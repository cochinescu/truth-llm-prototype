"""Two independent confidence extractors (PROTOCOL: FROZEN set).

signal      — reads the stub's latent reliability + noise; the Stage-A analogue
              of a logit-based extractor (replaced by a real one in Stage B).
consistency — agreement fraction over k resamples; mechanism-faithful and ports
              to Stage B unchanged.
"""

from __future__ import annotations

import numpy as np

from . import protocol
from .basemodel import Answer, SyntheticBaseModel


def signal_confidence(
    answer: Answer, rng: np.random.Generator, noise: float = protocol.SIGNAL_NOISE
) -> float:
    return float(np.clip(answer.latent_r + rng.normal(0.0, noise), 0.0, 1.0))


def consistency_confidence(
    model: SyntheticBaseModel,
    subject: str,
    attribute: str,
    answer: Answer,
    rng: np.random.Generator,
    k: int = protocol.K_SAMPLES,
) -> float:
    agree = sum(model.sample(subject, attribute, rng) == answer.value for _ in range(k))
    return agree / k


def extract_all(
    model: SyntheticBaseModel,
    subject: str,
    attribute: str,
    answer: Answer,
    rng: np.random.Generator,
) -> dict[str, float]:
    return {
        "signal": signal_confidence(answer, rng),
        "consistency": consistency_confidence(model, subject, attribute, answer, rng),
    }
