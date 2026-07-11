"""CSVs -> the four headline figures. Rerunnable standalone:
python scripts/plot.py  (reads/writes results/)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARM_COLORS = {
    "full": "#2166ac", "uniform": "#b2182b", "always_hedged": "#ef8a62",
    "threshold_only": "#67a9cf", "store_no_ack": "#7b3294",
    "no_provenance": "#d6604d", "stateless": "#878787",
}


def _read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def make_figures(results: Path) -> None:
    sr = str(results)
    label = "Stage C" if "stagec" in sr else ("Stage B" if "stageb" in sr else "Stage A")
    _fig1_reliability(results, label)
    _fig2_consistency(results, label)
    _fig3_assertions(results, label)
    _fig4_overhead(results, label)


def _fig1_reliability(results: Path, label: str) -> None:
    rows = _read(results / "fidelity_bins.csv")
    modes = sorted({r["extractor_mode"] for r in rows} - {"combined"})[:2] or ["combined"]
    fig, axes = plt.subplots(1, max(len(modes), 1), figsize=(4.5 * max(len(modes), 1), 4),
                             sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect fidelity")
        for arm in ["full", "uniform", "always_hedged"]:
            pts = sorted(
                (float(r["nominal"]), float(r["empirical_acc"]))
                for r in rows if r["arm"] == arm and r["extractor_mode"] == mode)
            if pts:
                ax.plot(*zip(*pts), "o-", color=ARM_COLORS[arm], label=arm)
        ax.set_title(f"extractor: {mode}")
        ax.set_xlabel("nominal confidence of expression category")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("empirical accuracy")
    axes[0][0].legend(fontsize=8)
    title = (f"Expression fidelity (C2) — {label}" if len(modes) == 1 else
             f"Expression fidelity (C2): reliability over expression categories — {label}")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(results / "fig1_reliability.png", dpi=150)
    plt.close(fig)


def _fig2_consistency(results: Path, label: str) -> None:
    rows = _read(results / "consistency.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    arms = [r["arm"] for r in rows]
    vals = [float(r["contradiction_rate"]) for r in rows]
    err_lo = [max(0.0, v - float(r["ci_lo"])) for v, r in zip(vals, rows)]
    err_hi = [max(0.0, float(r["ci_hi"]) - v) for v, r in zip(vals, rows)]
    ax.bar(arms, vals, yerr=[err_lo, err_hi], capsize=3,
           color=[ARM_COLORS[a] for a in arms])
    ax.set_ylabel("self-contradiction rate (unacknowledged REASK flips)")
    ax.set_title(f"Revision & consistency (C3) — {label}; store arms vs stateless")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(results / "fig2_consistency.png", dpi=150)
    plt.close(fig)


def _fig3_assertions(results: Path, label: str) -> None:
    rows = _read(results / "assertions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    arms = [r["arm"] for r in rows]
    vals = [float(r["confidently_false_rate"]) for r in rows]
    err_lo = [max(0.0, v - float(r["ci_lo"])) for v, r in zip(vals, rows)]
    err_hi = [max(0.0, float(r["ci_hi"]) - v) for v, r in zip(vals, rows)]
    axes[0].bar(arms, vals, yerr=[err_lo, err_hi], capsize=3,
                color=[ARM_COLORS[a] for a in arms])
    axes[0].set_ylabel("confidently-asserted-false rate")
    axes[0].set_title("unsupported confident assertions (C4)")
    axes[0].tick_params(axis="x", rotation=30)
    x = range(len(arms))
    axes[1].bar([i - 0.2 for i in x], [float(r["coverage"]) for r in rows],
                width=0.4, label="coverage", color="#67a9cf")
    axes[1].bar([i + 0.2 for i in x], [float(r["answer_accuracy"]) for r in rows],
                width=0.4, label="answer-when-given accuracy", color="#2166ac")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(arms, rotation=30)
    axes[1].set_title("the trade is reported: coverage vs accuracy")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Provenance gating (C4) — {label}")
    fig.tight_layout()
    fig.savefig(results / "fig3_assertions.png", dpi=150)
    plt.close(fig)


def _fig4_overhead(results: Path, label: str) -> None:
    rows = [r for r in _read(results / "overhead.csv") if r["ci_lo"]]
    fig, ax = plt.subplots(figsize=(5, 4))
    names = [r["config"] for r in rows]
    vals = [float(r["median_ms"]) for r in rows]
    err_lo = [max(0.0, v - float(r["ci_lo"])) for v, r in zip(vals, rows)]
    err_hi = [max(0.0, float(r["ci_hi"]) - v) for v, r in zip(vals, rows)]
    ax.bar(names, vals, yerr=[err_lo, err_hi], capsize=4, color=["#2166ac", "#878787"])
    ax.set_ylabel("ms per 24-turn conversation (median, 95% interval)")
    ax.set_title(f"Layer machinery overhead — {label}\n(model inference excluded)")
    fig.tight_layout()
    fig.savefig(results / "fig4_overhead.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    make_figures(Path(__file__).resolve().parents[1] / "results")
    print("figures written")
