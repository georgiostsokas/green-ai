"""Regenerate the figures in docs/figures from the package itself.

python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from green_ai import Estimator
from green_ai import batch as pp_batch
from green_ai import report as pp_report
from green_ai.prefill import CENTRAL, LOW
from green_ai.ranges import mean
from green_ai.report import BLUE, INK_2, ORANGE, _style
from green_ai.tasks import load_profiles, rank_tasks

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def hero(path: Path) -> None:
    plt = _style()
    r = np.logspace(0, 3, 300)
    est = Estimator(zone="WOR")
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    ax.fill_between(
        r,
        np.array(
            [
                est.estimate("gpt-6-sol", input_tokens=float(x) * 500, output_tokens=500, prefill="low").prefill_share
                * 100
                for x in r
            ]
        ),
        np.array(
            [
                est.estimate(
                    "gpt-6-sol", input_tokens=float(x) * 500, output_tokens=500, prefill="central"
                ).prefill_share
                * 100
                for x in r
            ]
        ),
        color=ORANGE,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        r,
        np.array(
            [
                est.estimate(
                    "gpt-6-sol", input_tokens=float(x) * 500, output_tokens=500, prefill="central"
                ).prefill_share
                * 100
                for x in r
            ]
        ),
        color=ORANGE,
        lw=2,
        label=f"central (alpha={CENTRAL.alpha})",
    )
    ax.plot(
        r,
        np.array(
            [
                est.estimate("gpt-6-sol", input_tokens=float(x) * 500, output_tokens=500, prefill="low").prefill_share
                * 100
                for x in r
            ]
        ),
        color=BLUE,
        lw=2,
        label=f"low (alpha={LOW.alpha})",
    )
    for x, label in ((3, "short chat"), (50, "RAG, document Q&A"), (200, "agents, coding")):
        ax.axvline(x, color="#c9c8c3", lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.text(x * 1.07, 93, label, fontsize=8, color=INK_2, va="top")
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 3, 10, 30, 100, 300, 1000], ["1:1", "3:1", "10:1", "30:1", "100:1", "300:1", "1000:1"])
    ax.set_xlabel("Prompt length relative to answer length (input:output tokens)")
    ax.set_ylabel("Input correction share of\nestimated electricity (%)")
    ax.set_title("Longer prompts increase the input-processing scenario share")
    ax.legend(loc="center left")
    ax.grid(axis="x", visible=False)
    fig.text(
        0.01, 0.01, "GPU input scenarios for GPT-6 Sol. Workload positions are illustrative.", fontsize=7, color=INK_2
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def model_gap(path: Path) -> None:
    """Same 50:1 request through a few well known models."""
    plt = _style()
    est = Estimator(zone="WOR")
    models = [
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-opus-5-5"),
        ("google_genai", "gemini-3.8-flash"),
        ("mistralai", "mistral-large-4"),
        ("openai", "gpt-4o"),
        ("anthropic", "claude-sonnet-4-5"),
        ("openai", "gpt-6-sol"),
    ]
    rows = []
    for p, m in models:
        e = est.estimate(m, provider=p, input_tokens=25_000, output_tokens=500)
        rows.append((m, mean(e.baseline.energy_wh), mean(e.prefill.energy_wh)))
    rows.sort(key=lambda x: x[1] + x[2])
    names = [r[0] for r in rows]
    base = np.array([r[1] for r in rows])
    pref = np.array([r[2] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 3.9), dpi=160)
    ax.barh(names, base, color=BLUE, height=0.62, edgecolor="#fcfcfb", linewidth=1.5, label="Reference (decode)")
    ax.barh(
        names,
        pref,
        left=base,
        color=ORANGE,
        height=0.62,
        edgecolor="#fcfcfb",
        linewidth=1.5,
        label="Prefill correction (central)",
    )
    for i, (b, p) in enumerate(zip(base, pref, strict=True)):
        ax.text(b + p, i, f"  +{p / b:.0%}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Energy per request (Wh), world average grid")
    ax.set_title("One request, 25,000 tokens in, 500 tokens out")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def rank_shift(path: Path) -> None:
    """Slope chart: rank per 1,000 generated tokens versus rank per completed task."""
    plt = _style()
    ranking, _ = rank_tasks(load_profiles(ROOT / "examples" / "task_profiles.csv"), zone="WOR")
    n = len(ranking)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    for r in ranking:
        moved = r.shift > 0
        color = ORANGE if moved else BLUE
        ax.plot([0, 1], [r.token_rank, r.rank], color=color, lw=2, marker="o", ms=6, solid_capstyle="round")
        ax.text(-0.04, r.token_rank, r.footprint.profile.name, ha="right", va="center", fontsize=8.5, color=INK_2)
        wh = mean(r.footprint.energy_wh)
        ax.text(
            1.04, r.rank, f"{r.footprint.profile.name}  {wh:.3g} Wh", ha="left", va="center", fontsize=8.5, color=INK_2
        )
    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(n + 0.6, 0.4)
    ax.set_xticks([0, 1], ["Rank per 1,000\ngenerated tokens", "Rank per\ncompleted task"])
    ax.set_yticks([])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title("Rank per token vs rank per completed task")
    fig.text(
        0.01,
        0.01,
        "Synthetic example profiles (examples/task_profiles.csv), world average grid.",
        fontsize=7,
        color=INK_2,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    hero(FIG / "underestimation.png")
    model_gap(FIG / "model_gap.png")
    rank_shift(FIG / "rank_shift.png")
    est = Estimator()
    est.registry.load_custom(ROOT / "examples" / "custom_models.yaml")
    res = pp_batch.run(ROOT / "examples" / "sample_usage.csv", estimator=est)
    pp_report.chart_monthly(res, FIG / "example_monthly.png")
    pp_report.chart_models(res, FIG / "example_models.png")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
