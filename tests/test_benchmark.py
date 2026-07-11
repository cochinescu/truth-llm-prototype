import json

from benchmark.build_benchmark import generate
from truthllm import protocol
from truthllm.world import World


def _gen(seed=31, n=12):
    w = World(seed=seed)
    return w, *generate(w, n_cases=n, seed=seed)


def test_seeded_identity():
    w = World(seed=31)
    a = generate(w, n_cases=6, seed=31)
    b = generate(w, n_cases=6, seed=31)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_targets_valid_and_labels_complete():
    w, instances, labels = _gen()
    for inst, lab in zip(instances, labels):
        turns, tl = inst["turns"], lab["turn_labels"]
        assert len(turns) == len(tl) == protocol.N_SESSIONS * protocol.TURNS_PER_SESSION
        asked = set()
        for turn, l in zip(turns, tl):
            if turn["type"] in ("ASK", "REASK"):
                asked.add(turn["t"])
                assert "truth" in l
            if turn["type"] == "REASK":
                assert turn["reask_of"] in asked and turn["reask_of"] < turn["t"]
            if turn["type"] == "CORRECT":
                assert "correct_true" in l and turn["targets"] < turn["t"]
                # label value consistency: true corrections carry world truth
                if l["correct_true"]:
                    assert turn["value"] == w.truth(turn["subject"], turn["attribute"])
                else:
                    assert turn["value"] != w.truth(turn["subject"], turn["attribute"])
            if turn["type"] == "TELL":
                assert "tell_true" in l


def test_cross_session_reasks_exist():
    _, instances, _ = _gen(n=20)
    n_cross = 0
    for inst in instances:
        session_of = {t["t"]: t["session"] for t in inst["turns"]}
        for t in inst["turns"]:
            if t["type"] == "REASK" and session_of[t["reask_of"]] < t["session"]:
                n_cross += 1
                break
    assert n_cross >= 15  # nearly every case; forced on the last turn when possible


def test_false_corrections_present():
    _, instances, labels = _gen(n=20)
    flags = [l.get("correct_true") for lab in labels for l in lab["turn_labels"]
             if "correct_true" in l]
    assert any(flags) and not all(flags)
