"""End-to-end: one-command seeded regeneration + the pre-specified
directional acceptance criteria (build plan §4) + the scorer parity check."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    outdir = tmp_path_factory.mktemp("results-quick")
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reproduce.py"), "--quick",
         "--outdir", str(outdir)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    return outdir


def _csv(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def test_all_outputs_exist(quick_run):
    for name in ["fidelity.csv", "fidelity_bins.csv", "consistency.csv",
                 "assertions.csv", "capability.csv", "overhead.csv",
                 "run_meta.json", "RESULTS.md",
                 "fig1_reliability.png", "fig2_consistency.png",
                 "fig3_assertions.png", "fig4_overhead.png"]:
        assert (quick_run / name).exists(), name


def test_c2_direction_full_beats_both_controls(quick_run):
    fid = {(r["arm"], r["extractor_mode"]): float(r["ece"]) for r in _csv(quick_run / "fidelity.csv")}
    for mode in ["signal", "consistency"]:  # directional agreement across extractors
        assert fid[("full", mode)] < fid[("uniform", mode)], mode
        assert fid[("full", mode)] < fid[("always_hedged", mode)], mode


def test_c3_direction_stateless_worst_and_audit_perfect(quick_run):
    rows = _csv(quick_run / "consistency.csv")
    con = {r["arm"]: float(r["contradiction_rate"]) for r in rows}
    assert all(con["stateless"] > v for a, v in con.items() if a != "stateless"), con
    for r in rows:
        assert r["n_acks"] == r["n_traced"], r["arm"]


def test_c3_acceptance_rule_discriminates(quick_run):
    rows = {r["arm"]: r for r in _csv(quick_run / "consistency.csv")}
    r = rows["full"]
    assert r["true_accept_rate"] and r["false_accept_rate"], "no correction decisions scored"
    assert float(r["false_accept_rate"]) < float(r["true_accept_rate"])


def test_c4_direction_gating_reduces_confident_falsehoods(quick_run):
    rows = {r["arm"]: r for r in _csv(quick_run / "assertions.csv")}
    assert float(rows["full"]["confidently_false_rate"]) \
        < float(rows["no_provenance"]["confidently_false_rate"])
    # the trade is reported: gating costs coverage and that is visible
    assert float(rows["full"]["coverage"]) <= float(rows["uniform"]["coverage"])


def test_rerun_byte_identical_csvs(quick_run):
    """Logical outputs are byte-identical under the same seed; overhead.csv is
    wall-clock timing and exempt (the protocol's two-regime determinism rule)."""
    logical = lambda: {p.name: p.read_bytes() for p in quick_run.glob("*.csv")
                       if p.name != "overhead.csv"}
    before = logical()
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reproduce.py"), "--quick",
         "--outdir", str(quick_run)],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert before == logical()


def test_scorer_parity_on_full_arm(quick_run):
    """benchmark/score.py (stdlib, language-agnostic) agrees with the package
    metrics on the serialized event log."""
    sys.path.insert(0, str(ROOT))
    from benchmark.score import score
    events = (quick_run / "events" / "full.jsonl").read_text().splitlines()
    if (ROOT / "benchmark" / "v1.0" / "labels.jsonl").exists():
        label_lines = (ROOT / "benchmark" / "v1.0" / "labels.jsonl").read_text().splitlines()
    else:
        pytest.skip("benchmark not frozen yet")
    labels = [json.loads(l) for l in label_lines]
    s = score(events, labels)
    rows = {r["arm"]: r for r in _csv(quick_run / "consistency.csv")}
    assert abs(s["contradiction_rate"] - float(rows["full"]["contradiction_rate"])) < 1e-9
    assert s["ack_total"] == int(rows["full"]["n_acks"])
    assert s["ack_traced"] == int(rows["full"]["n_traced"])
