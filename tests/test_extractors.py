import numpy as np

from truthllm.basemodel import SyntheticBaseModel
from truthllm.extractors import consistency_confidence, signal_confidence
from truthllm.world import World


def test_signal_correlates_with_latent_reliability():
    w = World(seed=21)
    m = SyntheticBaseModel(w, seed=21)
    rng = np.random.default_rng(1)
    rs, cs = [], []
    for k, ans in list(m.held.items())[:80]:
        rs.append(ans.latent_r)
        cs.append(signal_confidence(ans, rng))
    # Spearman via rank correlation
    def ranks(x):
        return np.argsort(np.argsort(x))
    rho = np.corrcoef(ranks(np.array(rs)), ranks(np.array(cs)))[0, 1]
    assert rho >= 0.8, rho


def test_consistency_confidence_near_reliability():
    w = World(seed=22)
    m = SyntheticBaseModel(w, seed=22)
    rng = np.random.default_rng(2)
    errs = [abs(consistency_confidence(m, k[0], k[1], ans, rng) - ans.latent_r)
            for k, ans in list(m.held.items())[:60]]
    assert float(np.median(errs)) <= 0.15


def test_extractors_seeded_stable():
    w = World(seed=23)
    m = SyntheticBaseModel(w, seed=23)
    k, ans = next(iter(m.held.items()))
    a = consistency_confidence(m, k[0], k[1], ans, np.random.default_rng(9))
    b = consistency_confidence(m, k[0], k[1], ans, np.random.default_rng(9))
    assert a == b
