"""One seeded command -> benchmark -> all arms -> CSVs + figures + RESULTS.md.

Usage:
    python scripts/reproduce.py [--seed 20260711] [--quick]

--quick: 20 cases, 200 bootstrap resamples, 50 overhead iters (smoke run).
If benchmark/v1.0/ is frozen, it is loaded and hash-verified (error on
mismatch — the PROTOCOL hash-freeze rule); --quick uses its first 20 cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.build_benchmark import generate, load  # noqa: E402
from truthllm import protocol  # noqa: E402
from truthllm.arms import ARM_ORDER, ARMS  # noqa: E402
from truthllm.basemodel import SyntheticBaseModel  # noqa: E402
from truthllm.metrics import (  # noqa: E402
    ack_audit, ack_completeness, assertion_rates, cluster_bootstrap,
    contradiction_rate, correction_scores, expression_auc, expression_ece,
    overhead_ms,
)
from truthllm.pipeline import run_conversation  # noqa: E402
from truthllm.world import World  # noqa: E402

FIDELITY_ARMS = ["full", "uniform", "always_hedged", "threshold_only"]
EXTRACTOR_MODES = ["combined", "signal", "consistency"]


def run_all(seed: int, quick: bool):
    world = World(seed=seed)
    model = SyntheticBaseModel(world, seed=seed)
    docs = world.retrieval_docs(seed=seed)

    frozen = (ROOT / "benchmark" / "v1.0" / "MANIFEST.json").exists()
    if frozen:
        instances, labels_rows, manifest = load(verify=True)
        benchmark_src = f"benchmark/v1.0 (frozen, sha-verified, seed {manifest['seed']})"
    else:
        instances, labels_rows = generate(world, seed=seed + 1)
        manifest = None
        benchmark_src = f"in-memory (seed {seed + 1}, NOT frozen)"
    if quick:
        instances, labels_rows = instances[:20], labels_rows[:20]
    labels = {row["case_id"]: row["turn_labels"] for row in labels_rows}

    runs: dict[tuple[str, str], list] = {}
    for mode in EXTRACTOR_MODES:
        for i, arm_name in enumerate(ARM_ORDER):
            if mode != "combined" and arm_name not in FIDELITY_ARMS:
                continue
            rng = np.random.default_rng(protocol.SEED_ARM_BASE + i + 1000 * EXTRACTOR_MODES.index(mode))
            case_events = [(inst["case_id"],
                            run_conversation(inst, ARMS[arm_name], model, docs, rng, extractor_mode=mode))
                           for inst in instances]
            runs[(arm_name, mode)] = case_events
    return world, model, docs, instances, labels, runs, benchmark_src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=protocol.MASTER_SEED)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    boot_n = 200 if args.quick else protocol.BOOTSTRAP_N
    overhead_iters = 50 if args.quick else protocol.OVERHEAD_ITERS

    t_start = time.perf_counter()
    world, model, docs, instances, labels, runs, benchmark_src = run_all(args.seed, args.quick)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "events").mkdir(exist_ok=True)

    # --- fidelity.csv + fidelity_bins.csv (C2) --------------------------------
    fid_rows, bin_rows = [], []
    for (arm, mode), case_events in runs.items():
        if arm not in FIDELITY_ARMS:
            continue
        est, lo, hi = cluster_bootstrap(case_events, lambda ce: expression_ece(ce, world)[0],
                                        n=boot_n, seed=protocol.SEED_BOOTSTRAP)
        fid_rows.append([arm, mode, f"{est:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
        for cat, (acc, nominal, n) in expression_ece(case_events, world)[1].items():
            bin_rows.append([arm, mode, cat, f"{acc:.4f}", nominal, n])
    _write_csv(results / "fidelity.csv", ["arm", "extractor_mode", "ece", "ci_lo", "ci_hi"], fid_rows)
    _write_csv(results / "fidelity_bins.csv",
               ["arm", "extractor_mode", "category", "empirical_acc", "nominal", "n"], bin_rows)

    # --- discrimination.csv (Amendment 1: expression-AUC) ---------------------
    auc_rows = []
    for (arm, mode), case_events in runs.items():
        if arm not in FIDELITY_ARMS:
            continue
        est, lo, hi = cluster_bootstrap(case_events, lambda ce: expression_auc(ce, world),
                                        n=boot_n, seed=protocol.SEED_BOOTSTRAP)
        auc_rows.append([arm, mode, f"{est:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    _write_csv(results / "discrimination.csv",
               ["arm", "extractor_mode", "auc", "ci_lo", "ci_hi"], auc_rows)

    # --- consistency.csv (C3) --------------------------------------------------
    con_rows = []
    for arm in ARM_ORDER:
        ce = runs[(arm, "combined")]
        est, lo, hi = cluster_bootstrap(ce, contradiction_rate, n=boot_n, seed=protocol.SEED_BOOTSTRAP)
        n_acks, n_traced = ack_audit(ce)
        n_accepted, _n_ackt = ack_completeness(ce)
        t_acc, f_acc = correction_scores(ce, labels)
        con_rows.append([arm, f"{est:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                         n_acks, n_traced, n_accepted, _fmt(t_acc), _fmt(f_acc)])
    _write_csv(results / "consistency.csv",
               ["arm", "contradiction_rate", "ci_lo", "ci_hi", "n_acks", "n_traced",
                "n_accepted_revisions", "true_accept_rate", "false_accept_rate"], con_rows)

    # --- assertions.csv + capability.csv (C4) ----------------------------------
    asr_rows, cap_rows = [], []
    for arm in ARM_ORDER:
        ce = runs[(arm, "combined")]
        est, lo, hi = cluster_bootstrap(
            ce, lambda c: assertion_rates(c, world)["confidently_false_rate"],
            n=boot_n, seed=protocol.SEED_BOOTSTRAP)
        rates = assertion_rates(ce, world)
        asr_rows.append([arm, f"{est:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                         f"{rates['coverage']:.4f}", f"{rates['answer_accuracy']:.4f}"])
        acc, acc_lo, acc_hi = cluster_bootstrap(
            ce, lambda c: assertion_rates(c, world)["answer_accuracy"],
            n=boot_n, seed=protocol.SEED_BOOTSTRAP)
        cov, cov_lo, cov_hi = cluster_bootstrap(
            ce, lambda c: assertion_rates(c, world)["coverage"],
            n=boot_n, seed=protocol.SEED_BOOTSTRAP)
        cap_rows.append([arm, f"{acc:.4f}", f"{acc_lo:.4f}", f"{acc_hi:.4f}",
                         f"{cov:.4f}", f"{cov_lo:.4f}", f"{cov_hi:.4f}"])
    _write_csv(results / "assertions.csv",
               ["arm", "confidently_false_rate", "ci_lo", "ci_hi", "coverage", "answer_accuracy"],
               asr_rows)
    _write_csv(results / "capability.csv",
               ["arm", "answer_accuracy", "acc_ci_lo", "acc_ci_hi",
                "coverage", "cov_ci_lo", "cov_ci_hi"], cap_rows)

    # --- overhead.csv -----------------------------------------------------------
    case0 = instances[0]
    rng_oh = np.random.default_rng(args.seed + 42)
    def run_full():
        run_conversation(case0, ARMS["full"], model, docs, rng_oh)
    def run_bare():
        for turn in case0["turns"]:
            if turn["type"] in ("ASK", "REASK"):
                model.query(turn["subject"], turn["attribute"])
    oh_rows = []
    for name, fn in [("full_layer", run_full), ("bare_baseline", run_bare)]:
        med, lo, hi = overhead_ms(fn, iters=overhead_iters)
        oh_rows.append([name, f"{med:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    added = float(oh_rows[0][1]) - float(oh_rows[1][1])
    oh_rows.append(["added_per_conversation", f"{added:.4f}", "", ""])
    _write_csv(results / "overhead.csv", ["config", "median_ms", "ci_lo", "ci_hi"], oh_rows)

    # --- event logs for the language-agnostic scorer -----------------------------
    for arm in ARM_ORDER:
        _write_events(results / "events" / f"{arm}.jsonl", runs[(arm, "combined")])

    # --- run_meta.json ------------------------------------------------------------
    meta = {
        "master_seed": args.seed,
        "quick": args.quick,
        "n_cases": len(instances),
        "benchmark": benchmark_src,
        "bootstrap_n": boot_n,
        "overhead_iters": overhead_iters,
        "extractor_modes": EXTRACTOR_MODES,
        "arms": ARM_ORDER,
        "protocol": {k: v for k, v in vars(protocol).items()
                     if k.isupper() and isinstance(v, (int, float, str, dict, tuple, list))},
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_seconds": round(time.perf_counter() - t_start, 2),
    }
    (results / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")

    # --- figures + RESULTS.md -------------------------------------------------------
    from plot import make_figures  # scripts/ sibling
    make_figures(results)
    _write_results_md(results, meta, fid_rows, con_rows, asr_rows, oh_rows, auc_rows)
    print(f"done: {len(instances)} cases, {len(runs)} arm-runs, "
          f"{meta['wall_seconds']}s -> {results}")
    return 0


def _fmt(x: float) -> str:
    return "" if x != x else f"{x:.4f}"  # NaN -> empty cell


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _write_events(path: Path, case_events) -> None:
    with open(path, "w") as f:
        for cid, events in case_events:
            for ev in events:
                accepted = [[r.t, r.subject, r.attribute]
                            for r in ev.revision_events if r.accepted]
                corr = None
                if ev.turn_type == "CORRECT" and ev.revision_events:
                    corr = any(r.accepted for r in ev.revision_events
                               if r.trigger.value == "USER_CORRECTION")
                f.write(json.dumps({
                    "case_id": cid, "t": ev.t, "turn_type": ev.turn_type,
                    "subject": ev.subject, "attribute": ev.attribute,
                    "category": ev.category, "expressed_value": ev.expressed_value,
                    "acknowledged": ev.acknowledged,
                    "ack_key": list(ev.ack_key) if ev.ack_key else None,
                    "accepted_revisions": accepted,
                    "correction_accepted": corr,
                }) + "\n")


def _write_results_md(results: Path, meta: dict, fid_rows, con_rows, asr_rows, oh_rows, auc_rows) -> None:
    fid = {(r[0], r[1]): float(r[2]) for r in fid_rows}
    con = {r[0]: float(r[1]) for r in con_rows}
    asr = {r[0]: float(r[1]) for r in asr_rows}
    checks = [
        ("C2 direction (signal): full ECE < uniform AND < always_hedged",
         fid[("full", "signal")] < fid[("uniform", "signal")]
         and fid[("full", "signal")] < fid[("always_hedged", "signal")]),
        ("C2 direction (consistency): full ECE < uniform AND < always_hedged",
         fid[("full", "consistency")] < fid[("uniform", "consistency")]
         and fid[("full", "consistency")] < fid[("always_hedged", "consistency")]),
        ("C3 direction: stateless contradiction rate > every store arm's",
         all(con["stateless"] > con[a] for a in con if a != "stateless")),
        ("C4 direction: full confidently-false rate < no_provenance's",
         asr["full"] < asr["no_provenance"]),
    ]
    ack_ok = all(int(r[4]) == int(r[5]) for r in con_rows)
    checks.append(("C3b audit: every acknowledgment traces to an accepted revision", ack_ok))

    lines = [
        "# Truth Prototype Results — Stage A (synthetic validation)",
        "",
        f"Generated by `scripts/reproduce.py` with master seed `{meta['master_seed']}`"
        f"{' (QUICK mode)' if meta['quick'] else ''}; benchmark: {meta['benchmark']}; "
        f"{meta['n_cases']} conversations per arm; cluster bootstrap n={meta['bootstrap_n']} "
        f"(unit = conversation).",
        "",
        "## Instrument-validation checks (pre-registered in the build plan)",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {'PASS' if ok else '**FAIL**'} |" for name, ok in checks]
    lines += [
        "",
        "## Expression fidelity (C2) — expression-ECE, lower is better",
        "",
        "| arm | combined | signal | consistency |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in FIDELITY_ARMS:
        lines.append(f"| {arm} | " + " | ".join(
            f"{fid[(arm, m)]:.3f}" for m in EXTRACTOR_MODES) + " |")
    auc = {(r[0], r[1]): float(r[2]) for r in auc_rows}
    lines += [
        "",
        "## Expression discrimination (Amendment 1) — AUC, 0.5 = chance",
        "",
        "| arm | combined | signal | consistency |",
        "| --- | ---: | ---: | ---: |",
    ]
    for arm in FIDELITY_ARMS:
        lines.append(f"| {arm} | " + " | ".join(
            f"{auc[(arm, m)]:.3f}" for m in EXTRACTOR_MODES) + " |")
    lines += [
        "",
        "## Consistency & revision (C3)",
        "",
        "| arm | contradiction rate | acks | traced | accepted revs | true-corr accept | false-corr accept |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in con_rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[4]} | {r[5]} | {r[6]} | {r[7] or '—'} | {r[8] or '—'} |")
    lines += [
        "",
        "## Assertions & capability (C4)",
        "",
        "| arm | confidently-false rate | coverage | answer accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in asr_rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[4]} | {r[5]} |")
    lines += [
        "",
        "## Overhead (per 24-turn conversation, median + 95% interval)",
        "",
        "| config | median (ms) | lo | hi |",
        "| --- | ---: | ---: | ---: |",
    ]
    for r in oh_rows:
        lines.append(f"| {r[0]} | {r[1]} | {r[2] or '—'} | {r[3] or '—'} |")
    lines += [
        "",
        "## Honesty notes (non-negotiable)",
        "",
        "- **Stage A validates the instrument, not the paper's claims.** The base",
        "  model is an LLM-free synthetic stub with knowable corruption; these",
        "  numbers demonstrate that the pipeline, metrics, controls, and benchmark",
        "  behave as designed. The manuscript's headline numbers come from the",
        "  Stage-B grid on a pinned open-weight model (PROTOCOL.md TBDs), never",
        "  from this table.",
        "- **Nothing here measures trust, believability, or perceived mind** —",
        "  that is Paper 5's human study. The layer produces calibrated *expression*,",
        "  not knowledge: over a wrong base model it politely asserts falsehoods",
        "  with well-ranked confidence.",
        "- Provenance tags are pipeline instrumentation, not epistemology; the",
        "  contradiction benchmark is constructed (injected, mechanically scored),",
        "  not open-domain; labels come from the generator's injection record only.",
        "- The coverage cost of gating (declines) is reported beside the",
        "  confidently-false reduction — the trade is the result, not a nuisance.",
        "- Overhead is layer-over-stub cost; real-model overhead is a Stage-B number.",
        "- Determinism: logical outputs byte-identical under the recorded seeds on a",
        "  pinned environment; timing rows are wall-clock (median + interval only).",
        "",
        f"Full parameter register: `run_meta.json` · protocol: `../PROTOCOL.md` · "
        f"wall time {meta['wall_seconds']}s.",
    ]
    (results / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
