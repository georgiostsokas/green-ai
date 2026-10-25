"""Reports from a :class:`~green_ai.batch.BatchResult`: Excel workbook, charts, Markdown summary."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from ._version import __version__
from .batch import BatchResult

# Palette: categorical slot 1 (blue) for the reference estimate, slot 2 (orange) for the
# prefill correction, recessive greys for ink and grid.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "font.size": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )
    return plt


def _fmt_energy(kwh: float) -> str:
    if kwh >= 1000:
        return f"{kwh / 1000:.2f} MWh"
    if kwh >= 1:
        return f"{kwh:.1f} kWh"
    return f"{kwh * 1000:.1f} Wh"


def _fmt_mass(kg: float) -> str:
    if kg >= 1000:
        return f"{kg / 1000:.2f} tCO2eq"
    if kg >= 1:
        return f"{kg:.1f} kgCO2eq"
    return f"{kg * 1000:.1f} gCO2eq"


def _fmt_water(l: float) -> str:  # noqa: E741
    return f"{l / 1000:.2f} m³" if l >= 1000 else f"{l:.1f} L"


def _energy_unit(series_max: float) -> tuple[float, str]:
    if series_max >= 2000:
        return 1 / 1000, "MWh"
    if series_max >= 2:
        return 1.0, "kWh"
    return 1000.0, "Wh"


# ---------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------


def chart_monthly(result: BatchResult, path: Path) -> Optional[Path]:
    if "month" not in result.rows:
        return None
    plt = _style()
    m = result.by_month()
    k, unit = _energy_unit(float(m["total_energy_kwh"].max()))
    base, pref = m["baseline_energy_kwh"] * k, m["prefill_energy_kwh"] * k
    x = range(len(m))
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    ax.bar(x, base, width=0.62, color=BLUE, edgecolor=SURFACE, linewidth=1.5, label="Reference estimate (decode)")
    ax.bar(
        x,
        pref,
        width=0.62,
        bottom=base,
        color=ORANGE,
        edgecolor=SURFACE,
        linewidth=1.5,
        label=f"Prefill correction (alpha={result.prefill_model.alpha:g})",
    )
    for i, (b, p) in enumerate(zip(base, pref, strict=True)):
        ax.text(i, b + p, f"+{p / b:.0%}" if b else "", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_xticks(list(x), m["month"].tolist())
    ax.set_ylabel(f"Energy ({unit})")
    ax.grid(axis="x", visible=False)
    ax.set_title("Monthly energy: reference estimate and prompt processing")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_models(result: BatchResult, path: Path, top: int = 10) -> Path:
    plt = _style()
    m = result.by_model().head(top).iloc[::-1]
    k, unit = _energy_unit(float(m["total_energy_kwh"].max()))
    labels = m["matched_model"].tolist()
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(m) + 1.4), dpi=160)
    ax.barh(
        labels,
        m["baseline_energy_kwh"] * k,
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=1.5,
        height=0.62,
        label="Reference estimate (decode)",
    )
    ax.barh(
        labels,
        m["prefill_energy_kwh"] * k,
        left=m["baseline_energy_kwh"] * k,
        color=ORANGE,
        edgecolor=SURFACE,
        linewidth=1.5,
        height=0.62,
        label="Prefill correction",
    )
    for i, (t, s) in enumerate(zip(m["total_energy_kwh"] * k, m["prefill_share"], strict=True)):
        ax.text(t, i, f"  {s:.0%} prefill", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel(f"Energy ({unit})")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Energy by model (top {len(m)})")
    ax.legend(loc="lower right")
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------------
# Markdown and Excel
# ---------------------------------------------------------------------------------


def summary_markdown(result: BatchResult) -> str:
    t = result.totals()
    sc = result.scenarios()
    lines = [
        "# LLM usage footprint report",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC with Green AI {__version__}.",
        "",
        "## Totals",
        "",
        "| Metric | Reference estimate | With prefill correction | Range (corrected) |",
        "|---|---|---|---|",
        f"| Energy | {_fmt_energy(t['baseline_energy_kwh'])} | {_fmt_energy(t['total_energy_kwh'])} | "
        f"{_fmt_energy(t['total_energy_kwh_min'])} to {_fmt_energy(t['total_energy_kwh_max'])} |",
        f"| GHG emissions | {_fmt_mass(t['baseline_gwp_kg'])} | {_fmt_mass(t['total_gwp_kg'])} | "
        f"{_fmt_mass(t['total_gwp_kg_min'])} to {_fmt_mass(t['total_gwp_kg_max'])} |",
        f"| Water consumption | {_fmt_water(t['baseline_water_l'])} | {_fmt_water(t['total_water_l'])} | "
        f"{_fmt_water(t['total_water_l_min'])} to {_fmt_water(t['total_water_l_max'])} |",
        "",
        f"Prompt processing accounts for **{t['prefill_share']:.0%}** of corrected energy: the correction adds "
        f"{t['correction_factor'] - 1:+.0%} to the reference estimate.",
        "",
        f"Tokens: {t['input_tokens']:,.0f} input ({t['cached_tokens']:,.0f} cached), "
        f"{t['output_tokens']:,.0f} output, overall input to output ratio "
        f"{t['input_tokens'] / max(t['output_tokens'], 1):.0f}:1.",
        "",
        "## Prefill scenarios",
        "",
        "| Scenario | alpha | Energy | vs reference |",
        "|---|---|---|---|",
    ]
    for _, r in sc.iterrows():
        lines.append(
            f"| {r['scenario']} | {r['alpha']:g} | {_fmt_energy(r['total_energy_kwh'])} | {r['vs_baseline']:+.0%} |"
        )
    lines += [
        "",
        "Low and central are GPU-energy sensitivity scenarios, not physical bounds or confidence intervals. "
        "The correction leaves server and embodied impacts unchanged. Use `--alpha` for other values.",
        "",
    ]

    um = result.unmatched
    lines += ["## Coverage", "", f"{t['rows_matched']:,} of {t['rows']:,} rows matched a registry model."]
    if not um.empty:
        share = t["unmatched_output_tokens"] / max(t["unmatched_output_tokens"] + t["output_tokens"], 1)
        lines += [
            f"Unmatched models carry {share:.1%} of output tokens and are excluded. Register them with "
            "`--custom models.yaml`:",
            "",
        ]
        for _, r in um.iterrows():
            sug = f" (did you mean {r['suggestions']}?)" if r["suggestions"] else ""
            lines.append(f"- `{r['model']}`: {int(r['rows'])} rows{sug}")
    lines += ["", "## Assumptions", ""]
    for k, v in result.settings.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def to_excel(result: BatchResult, path, *, include_original: bool = False):
    t = result.totals()
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame([{"metric": k, "value": v} for k, v in t.items()]).to_excel(xw, sheet_name="Summary", index=False)
        result.scenarios().to_excel(xw, sheet_name="Scenarios", index=False)
        if "month" in result.rows:
            result.by_month().to_excel(xw, sheet_name="Monthly", index=False)
        result.by_model().to_excel(xw, sheet_name="By model", index=False)
        result.by_provider().to_excel(xw, sheet_name="By provider", index=False)
        result.unmatched.to_excel(xw, sheet_name="Unmatched", index=False)
        result.export_rows(include_original=include_original).to_excel(xw, sheet_name="Rows", index=False)
        pd.DataFrame([{"setting": k, "value": str(v)} for k, v in result.settings.items()]).to_excel(
            xw, sheet_name="Assumptions", index=False
        )
        for ws in xw.book.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    if isinstance(cell.value, str) and cell.value.startswith(("=", "+", "-", "@")):
                        cell.value = "'" + cell.value
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 48)
            ws.freeze_panes = "A2"
    return path


def write_report(result: BatchResult, out_dir: Path, charts: bool = True) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    files["excel"] = to_excel(result, out_dir / "footprint.xlsx")
    if charts:
        mp = chart_monthly(result, out_dir / "monthly_energy.png")
        if mp:
            files["monthly_chart"] = mp
        files["model_chart"] = chart_models(result, out_dir / "energy_by_model.png")
    md = summary_markdown(result)
    if charts:
        imgs = [f"![{p.stem}]({p.name})" for k, p in files.items() if k.endswith("chart")]
        md = md.replace("## Prefill scenarios", "\n".join(imgs) + "\n\n## Prefill scenarios", 1)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    files["summary"] = out_dir / "summary.md"
    return files
