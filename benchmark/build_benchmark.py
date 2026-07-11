"""Seeded generator for the multi-session contradiction benchmark.

Labels are the injection record — world truth, TELL/CORRECT truthfulness, and
REASK back-references — written at generation time, never derived from any
layer's outputs (anti-circularity, PROTOCOL FROZEN).

`--freeze` writes benchmark/v1.0/: instances.jsonl + labels.jsonl +
MANIFEST.json (sha256 of both files, generation params, seed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truthllm import protocol  # noqa: E402
from truthllm.world import World  # noqa: E402


def generate(world: World, n_cases: int = protocol.N_CASES, seed: int = protocol.SEED_BENCHMARK):
    rng = np.random.default_rng(seed)
    tellable, askable = world.tellable_attrs, world.askable_attrs
    base_attr = world.public_rules().base_attr
    turn_types = list(protocol.TURN_MIX)
    probs = np.array([protocol.TURN_MIX[t] for t in turn_types])
    instances, labels = [], []

    for c in range(n_cases):
        turns, turn_labels = [], []
        asked: list[tuple[int, int, str, str]] = []  # (t, session, subject, attribute)
        cross_session_reask_done = False
        n_turns = protocol.N_SESSIONS * protocol.TURNS_PER_SESSION

        for t in range(n_turns):
            session = t // protocol.TURNS_PER_SESSION
            ttype = turn_types[rng.choice(len(turn_types), p=probs)]
            # targeted types degrade to ASK when no target exists yet
            if ttype in ("CORRECT", "REASK") and not asked:
                ttype = "ASK"
            # guarantee >=1 cross-session REASK: force it on the last turn if needed
            if (t == n_turns - 1 and not cross_session_reask_done
                    and any(sess < session for _, sess, _, _ in asked)):
                ttype = "REASK"

            if ttype == "ASK":
                s = world.subjects[rng.integers(len(world.subjects))]
                a = askable[rng.integers(len(askable))]
                turns.append({"t": t, "session": session, "type": "ASK", "subject": s, "attribute": a})
                turn_labels.append({"truth": world.truth(s, a)})
                asked.append((t, session, s, a))

            elif ttype == "TELL":
                s = world.subjects[rng.integers(len(world.subjects))]
                a = tellable[rng.integers(len(tellable))]
                truth = world.truth(s, a)
                tell_true = bool(rng.random() < protocol.TELL_TRUE_RATE)
                value = truth if tell_true else _wrong(truth, a, rng, world)
                turns.append({"t": t, "session": session, "type": "TELL",
                              "subject": s, "attribute": a, "value": value})
                turn_labels.append({"truth": truth, "tell_true": tell_true})

            elif ttype == "CORRECT":
                t0, _, s, a = asked[rng.integers(len(asked))]
                if a not in tellable:  # corrections target base facts
                    a = base_attr
                truth = world.truth(s, a)
                correct_true = bool(rng.random() < protocol.CORRECT_TRUE_RATE)
                value = truth if correct_true else _wrong(truth, a, rng, world)
                turns.append({"t": t, "session": session, "type": "CORRECT",
                              "subject": s, "attribute": a, "value": value, "targets": t0})
                turn_labels.append({"truth": truth, "correct_true": correct_true, "targets": t0})

            else:  # REASK
                earlier = [(tt, ss, s, a) for tt, ss, s, a in asked if ss < session]
                pool = earlier if (earlier and (not cross_session_reask_done or rng.random() < 0.5)) else asked
                t0, s0_sess, s, a = pool[rng.integers(len(pool))]
                if s0_sess < session:
                    cross_session_reask_done = True
                turns.append({"t": t, "session": session, "type": "REASK",
                              "subject": s, "attribute": a, "reask_of": t0})
                turn_labels.append({"truth": world.truth(s, a), "reask_of": t0})
                asked.append((t, session, s, a))

        cid = f"case-{c:04d}"
        instances.append({"case_id": cid, "turns": turns})
        labels.append({"case_id": cid, "turn_labels": turn_labels})
    return instances, labels


def _wrong(truth: str, attribute: str, rng: np.random.Generator, world: World) -> str:
    others = [v for v in world.values[attribute] if v != truth]
    return others[rng.integers(len(others))]


def freeze(version: str = "v1.0", world: World | None = None, seed: int = protocol.SEED_BENCHMARK) -> Path:
    out = Path(__file__).resolve().parent / version
    out.mkdir(parents=True, exist_ok=True)
    world = world if world is not None else World()
    instances, labels = generate(world, seed=seed)
    files = {"instances.jsonl": instances, "labels.jsonl": labels}
    sha = {}
    for name, rows in files.items():
        p = out / name
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        sha[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "benchmarkVersion": version.lstrip("v"),
        "schemaVersion": "1.0",
        "seed": seed,
        "world": type(world).__name__,
        "worldSeed": protocol.SEED_WORLD,
        "nCases": len(instances),
        "params": {
            "n_sessions": protocol.N_SESSIONS,
            "turns_per_session": protocol.TURNS_PER_SESSION,
            "turn_mix": protocol.TURN_MIX,
            "tell_true_rate": protocol.TELL_TRUE_RATE,
            "correct_true_rate": protocol.CORRECT_TRUE_RATE,
            "n_subjects": protocol.N_SUBJECTS,
        },
        "sha256": sha,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def load(version: str = "v1.0", verify: bool = True):
    out = Path(__file__).resolve().parent / version
    manifest = json.loads((out / "MANIFEST.json").read_text())
    if verify:
        for name, want in manifest["sha256"].items():
            got = hashlib.sha256((out / name).read_bytes()).hexdigest()
            if got != want:
                raise RuntimeError(
                    f"hash-freeze violation: {name} sha256 {got} != manifest {want}")
    instances = [json.loads(l) for l in (out / "instances.jsonl").read_text().splitlines()]
    labels = [json.loads(l) for l in (out / "labels.jsonl").read_text().splitlines()]
    return instances, labels, manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true", help="write benchmark/v1.0/")
    args = ap.parse_args()
    if args.freeze:
        path = freeze()
        print(f"frozen -> {path}")
    else:
        inst, lab = generate(World())
        print(f"generated {len(inst)} cases (in-memory; use --freeze to version)")
