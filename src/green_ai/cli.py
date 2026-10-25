"""Command line interface: ``green-ai --help``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ._version import __version__
from .estimator import Estimator
from .ranges import Range, hi, lo, mean
from .registry import ModelNotFoundError, Registry, ZoneNotFoundError, default_registry

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Prefill aware energy, carbon and water estimates for LLM API usage.",
    rich_markup_mode="rich",
)
console = Console()

PREFILL_HELP = "Prefill correction: none, low, central, pricing, or a numeric alpha."


def _registry(custom: Optional[Path]) -> Registry:
    reg = default_registry()
    if custom:
        reg.load_custom(custom)
    return reg


def _parse_range(text: Optional[str]):
    if text is None:
        return None
    if ":" in text:
        a, b = text.split(":", 1)
        return Range(float(a), float(b))
    return float(text)


def _fmt(v, unit: str, k: float = 1.0) -> str:
    m, a, b = mean(v) * k, lo(v) * k, hi(v) * k
    if a == b:
        return f"{m:.4g} {unit}"
    return f"{m:.4g} {unit} [dim]({a:.3g} to {b:.3g})[/dim]"


def _estimator(custom, zone, pue, wue, ci, alpha, prefill, batch_size) -> Estimator:
    from .engine import DEFAULT_HARDWARE

    hw = DEFAULT_HARDWARE.with_overrides(batch_size=batch_size)
    pm = alpha if alpha is not None else prefill
    return Estimator(
        registry=_registry(custom),
        zone=zone,
        pue=_parse_range(pue),
        wue=_parse_range(wue),
        carbon_intensity=ci,
        hardware=hw,
        prefill=pm,
    )


@app.command()
def estimate(
    model: str = typer.Argument(..., help="Model name as logged, e.g. gpt-4o or azure/gpt-4o-2024-08-06."),
    output_tokens: int = typer.Option(..., "--out", "-o", help="Generated tokens (include reasoning tokens)."),
    input_tokens: int = typer.Option(0, "--in", "-i", help="Prompt tokens, including cached ones."),
    cached_tokens: int = typer.Option(0, "--cached", "-c", help="Prompt tokens served from the provider cache."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    requests: int = typer.Option(1, "--requests", "-n", help="Number of requests the token totals span."),
    zone: Optional[str] = typer.Option(None, "--zone", "-z", help="Electricity mix (ISO alpha-3, WOR = world)."),
    pue: Optional[str] = typer.Option(None, help="Override PUE, e.g. 1.2 or 1.1:1.4"),
    wue: Optional[str] = typer.Option(None, help="Override datacenter WUE in L/kWh."),
    carbon_intensity: Optional[float] = typer.Option(None, "--ci", help="Override grid carbon intensity kgCO2eq/kWh."),
    prefill: str = typer.Option("central", help=PREFILL_HELP),
    alpha: Optional[float] = typer.Option(None, help="Custom prefill coefficient, overrides --prefill."),
    batch_size: Optional[int] = typer.Option(None, help="Concurrent requests per server (default 64)."),
    custom: Optional[Path] = typer.Option(None, help="YAML/JSON file with custom models and providers."),
    as_json: bool = typer.Option(False, "--json", help="Print machine readable JSON."),
) -> None:
    """Estimate the footprint of one request (or of aggregated token totals)."""
    est = _estimator(custom, zone, pue, wue, carbon_intensity, alpha, prefill, batch_size)
    try:
        kw = dict(
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            requests=requests,
        )
        e = est.estimate(model, **kw)
    except (ModelNotFoundError, ZoneNotFoundError, ValueError, LookupError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None

    if as_json:
        import json

        typer.echo(json.dumps(e.to_dict(), indent=2))
        return

    console.print(
        f"[bold]{e.model.provider}/{e.model.name}[/bold]  [dim]({e.match.status.value} match, "
        f"{e.model.architecture}, zone {e.mix.zone}, PUE {e.pue})[/dim]"
    )
    t = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    t.add_column("")
    t.add_column("Reference (decode)", justify="right")
    t.add_column(f"Prefill ({e.prefill_model.name})", justify="right")
    t.add_column("Total", justify="right")
    for label, attr, unit, k in (
        ("Energy", "energy_kwh", "Wh", 1000),
        ("GHG", "gwp", "gCO2eq", 1000),
        ("Water", "water_l", "mL", 1000),
        ("Primary energy", "pe", "kJ", 1000),
    ):
        t.add_row(
            label,
            _fmt(getattr(e.baseline, attr), unit, k),
            _fmt(getattr(e.prefill, attr), unit, k),
            _fmt(getattr(e.total, attr), unit, k),
        )
    console.print(t)
    console.print(
        f"Prompt processing share of energy: [bold]{e.prefill_share:.0%}[/bold] (correction x{e.correction_factor:.2f})"
    )

    sc = Table(title="Prefill scenarios", title_justify="left", box=None, header_style="bold")
    for c in ("scenario", "alpha", "energy", "vs reference"):
        sc.add_column(c, justify="right" if c != "scenario" else "left")
    for name in ("none", "low", "central"):
        s = est.estimate(model, prefill=name, **kw)
        sc.add_row(
            name, f"{s.prefill_model.alpha:g}", _fmt(s.total.energy_kwh, "Wh", 1000), f"{s.correction_factor - 1:+.0%}"
        )
    console.print(sc)
    for w in e.warnings:
        console.print(f"[yellow]note[/yellow] {w}")


@app.command()
def batch(
    path: Path = typer.Argument(..., exists=True, help="CSV, Excel, Parquet or JSON usage log."),
    out: Path = typer.Option(Path("green_ai_report"), "--out", "-o", help="Output folder."),
    mapping: list[str] = typer.Option([], "--map", "-m", help="Column mapping, e.g. --map output_tokens=gen_tok"),
    zone: Optional[str] = typer.Option(None, "--zone", "-z"),
    pue: Optional[str] = typer.Option(None),
    wue: Optional[str] = typer.Option(None),
    carbon_intensity: Optional[float] = typer.Option(None, "--ci"),
    prefill: str = typer.Option("central", help=PREFILL_HELP),
    alpha: Optional[float] = typer.Option(None),
    batch_size: Optional[int] = typer.Option(None),
    provider: Optional[str] = typer.Option(None, help="Provider to assume when the log has no provider column."),
    custom: Optional[Path] = typer.Option(None, help="YAML/JSON file with custom models and providers."),
    no_charts: bool = typer.Option(False, "--no-charts"),
) -> None:
    """Estimate a whole usage log and write an Excel workbook, charts and a Markdown summary."""
    from . import batch as _batch
    from . import report

    est = _estimator(custom, zone, pue, wue, carbon_intensity, alpha, prefill, batch_size)
    overrides = {}
    for item in mapping:
        if "=" not in item:
            console.print(f"[red]Invalid --map '{item}', expected field=column[/red]")
            raise typer.Exit(1)
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()
    try:
        res = _batch.run(path, estimator=est, default_provider=provider, **overrides)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None
    files = report.write_report(res, out, charts=not no_charts)
    t = res.totals()
    console.print(
        f"[bold]{t['rows_matched']:,}/{t['rows']:,} rows matched[/bold]  "
        f"input {t['input_tokens']:,.0f}  output {t['output_tokens']:,.0f} tokens"
    )
    console.print(
        f"Energy  {t['baseline_energy_kwh']:,.1f} kWh reference -> "
        f"[bold]{t['total_energy_kwh']:,.1f} kWh[/bold] with prefill ({t['correction_factor'] - 1:+.0%})"
    )
    console.print(f"GHG     {t['total_gwp_kg']:,.1f} kgCO2eq   water {t['total_water_l']:,.0f} L")
    if not res.unmatched.empty:
        console.print(
            f"[yellow]{len(res.unmatched)} unmatched model(s)[/yellow]: "
            + ", ".join(res.unmatched["model"].astype(str).head(8))
        )
    for k, p in files.items():
        console.print(f"[dim]{k:>14}[/dim] {p}")


@app.command("models")
def list_models(
    search: str = typer.Argument("", help="Substring to filter by."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    custom: Optional[Path] = typer.Option(None),
) -> None:
    """List registered models and their parameters."""
    reg = _registry(custom)
    t = Table(box=None, header_style="bold")
    for c in ("provider", "model", "arch", "total (B)", "active (B)", "tok/s", "TTFT (s)"):
        t.add_column(c, justify="right" if "(" in c or c == "tok/s" else "left")
    rows = reg.search(search, provider)
    for m in rows:
        t.add_row(
            m.provider,
            m.name,
            m.architecture,
            f"{m.total_params_b:g}",
            f"{m.active_params_b:g}",
            f"{m.tps:g}" if m.tps else "",
            f"{m.ttft:g}" if m.ttft else "",
        )
    console.print(t)
    console.print(f"[dim]{len(rows)} models[/dim]")


@app.command()
def match(
    names: list[str] = typer.Argument(..., help="Raw model names to resolve."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
) -> None:
    """Show how raw model names from logs resolve to registry entries."""
    reg = default_registry()
    t = Table(box=None, header_style="bold")
    for c in ("query", "status", "resolved to", "suggestions"):
        t.add_column(c)
    for n in names:
        m = reg.match(n, provider)
        t.add_row(n, m.status.value, f"{m.model.provider}/{m.model.name}" if m.model else "", ", ".join(m.suggestions))
    console.print(t)


@app.command()
def zones(search: str = typer.Argument("", help="Filter by ISO code prefix.")) -> None:
    """List electricity mix zones and their factors."""
    reg = default_registry()
    t = Table(box=None, header_style="bold")
    for c in ("zone", "gCO2eq/kWh", "water L/kWh", "PE MJ/kWh"):
        t.add_column(c, justify="left" if c == "zone" else "right")
    for code, z in sorted(reg.zones.items()):
        if code.startswith(search.upper()):
            t.add_row(code, f"{z.gwp * 1000:.0f}", f"{z.wue:.2f}", f"{z.pe:.2f}")
    console.print(t)


@app.command()
def rank(
    input_tokens: int = typer.Option(1000, "--in", "-i"),
    output_tokens: int = typer.Option(1000, "--out", "-o"),
    provider: list[str] = typer.Option([], "--provider", "-p", help="Repeat to include several providers."),
    zone: Optional[str] = typer.Option("WOR", "--zone", "-z", help="Common zone so only models are compared."),
    prefill: str = typer.Option("central", help=PREFILL_HELP),
    top: int = typer.Option(25, help="Rows to show (0 = all)."),
    worst: bool = typer.Option(False, help="Show the most energy intensive first."),
) -> None:
    """Grade models A to E on a standard workload."""
    from .rating import rate_models

    rated = rate_models(
        input_tokens=input_tokens, output_tokens=output_tokens, providers=provider or None, zone=zone, prefill=prefill
    )
    if worst:
        rated = rated[::-1]
    shown = rated if top == 0 else rated[:top]
    t = Table(
        box=None,
        header_style="bold",
        title=f"{input_tokens:,} input + {output_tokens:,} output tokens, zone {zone}, prefill {prefill}",
        title_justify="left",
    )
    for c in ("#", "grade", "provider", "model", "energy Wh", "gCO2eq", "prefill share"):
        t.add_column(c, justify="right" if c not in ("provider", "model", "grade") else "left")
    colors = dict(zip("ABCDE", ("green", "green", "yellow", "orange3", "red"), strict=True))
    for r in shown:
        e = r.estimate
        t.add_row(
            str(r.rank),
            f"[{colors[r.grade]}]{r.grade}[/]",
            e.model.provider,
            e.model.name,
            f"{r.energy_wh:.3g}",
            f"{r.gwp_g:.3g}",
            f"{e.prefill_share:.0%}",
        )
    console.print(t)


@app.command()
def sensitivity(
    model: str = typer.Argument("gpt-4o"),
    output_tokens: int = typer.Option(500, "--out", "-o"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    zone: Optional[str] = typer.Option(None, "--zone", "-z"),
) -> None:
    """How much the reference estimate misses as prompts grow longer than answers."""
    est = Estimator(zone=zone)
    t = Table(box=None, header_style="bold", title=f"{model}, {output_tokens} output tokens", title_justify="left")
    for c in ("input:output", "reference Wh", "low Wh", "central Wh", "missed (central)"):
        t.add_column(c, justify="right")
    for ratio in (1, 5, 10, 25, 50, 100, 200, 500):
        kw = dict(provider=provider, output_tokens=output_tokens, input_tokens=ratio * output_tokens)
        b = est.estimate(model, prefill="none", **kw)
        lo_ = est.estimate(model, prefill="low", **kw)
        c = est.estimate(model, prefill="central", **kw)
        t.add_row(
            f"{ratio}:1",
            f"{mean(b.total.energy_wh):.3g}",
            f"{mean(lo_.total.energy_wh):.3g}",
            f"{mean(c.total.energy_wh):.3g}",
            f"{c.prefill_share:.0%}",
        )
    console.print(t)


@app.command()
def tasks(
    use_aa: bool = typer.Option(
        False, "--aa", help="Use Artificial Analysis Intelligence Index data (needs AA_API_KEY)."
    ),
    csv_path: Optional[Path] = typer.Option(None, "--csv", exists=True, help="CSV of task profiles (your own traces)."),
    key: Optional[str] = typer.Option(None, "--key", help="Artificial Analysis API key (default: AA_API_KEY)."),
    refresh: bool = typer.Option(False, "--refresh", help="Ignore the 24 h cache and fetch again."),
    tier: str = typer.Option("auto", help="auto, pro or free."),
    output_share: str = typer.Option("0.5:0.9", help="Free tier only: share of cost due to output tokens."),
    mapping: Optional[Path] = typer.Option(None, "--map", exists=True, help="YAML/JSON mapping of AA slugs to models."),
    zone: str = typer.Option("WOR", "--zone", "-z", help="Common electricity mix for a fair comparison."),
    by: str = typer.Option("energy", help="Rank energy, carbon, water per task, or efficiency (score per Wh)."),
    prefill: str = typer.Option("central", help=PREFILL_HELP),
    top: int = typer.Option(30, help="Rows to show (0 = all)."),
    export: Optional[Path] = typer.Option(None, help="Write the full ranking to a CSV file."),
    custom: Optional[Path] = typer.Option(None, help="YAML/JSON file with custom models and providers."),
    min_score: Optional[float] = typer.Option(None, help="Only compare models meeting this quality score."),
    aa_export: bool = typer.Option(
        False, help="CSV contains actual per-task values exported from Artificial Analysis."
    ),
    input_ratio: Optional[str] = typer.Option(
        None, help="For output-only CSV: assumed input/output ratio, e.g. 10:50."
    ),
    benchmark_plan: Optional[Path] = typer.Option(
        None, exists=True, help="Versioned AA evaluation weights/task denominators (JSON)."
    ),
    aa_model: list[str] = typer.Option(
        [], "--aa-model", help="Selected AA slug for a benchmark plan; repeat up to 20 times."
    ),
) -> None:
    """Footprint per completed task and the ranking it implies (reasoning tokens included)."""
    from . import tasks as _tasks

    reg = _registry(custom)
    est = Estimator(registry=reg)
    profiles: list = []
    footer = []
    if csv_path:
        profiles += _tasks.load_profiles(
            csv_path,
            registry=reg,
            source="artificial_analysis:export" if aa_export else None,
            input_ratio=_parse_range(input_ratio)
            if input_ratio and ":" in input_ratio
            else (Range(float(input_ratio), float(input_ratio)) if input_ratio else None),
        )
        footer.append(f"Profiles from {csv_path.name}.")
    if use_aa:
        from . import aa as _aa

        try:
            payload = _aa.fetch(key, refresh=refresh, tier=tier)
        except _aa.AAError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from None
        except OSError as exc:
            console.print(f"[red]Could not reach Artificial Analysis:[/red] {exc}")
            raise typer.Exit(1) from None
        share = _parse_range(output_share)
        plan = None
        if benchmark_plan:
            import json

            plan = json.loads(benchmark_plan.read_text(encoding="utf-8"))
            if not aa_model:
                raise typer.BadParameter("Use --aa-model to select model slugs for the benchmark plan")
            payload = _aa.fetch_details(aa_model, key, index_version=payload.get("intelligence_index_version"))
        res = _aa.to_profiles(
            payload,
            registry=reg,
            output_share=share if isinstance(share, Range) else Range(share, share),
            overrides=_aa.load_mapping(str(mapping) if mapping else None),
            benchmark_plan=plan,
        )
        profiles += res.profiles
        footer.append(res.attribution)
        if res.unmatched:
            footer.append(
                f"{len(res.unmatched)} benchmarked models have no registry match and were skipped "
                "(add them with --map or --custom): "
                + ", ".join(n for n, _ in res.unmatched[:8])
                + (" ..." if len(res.unmatched) > 8 else "")
            )
    if not profiles:
        console.print("Nothing to rank. Use --aa (with AA_API_KEY) and/or --csv profiles.csv.")
        raise typer.Exit(1)
    if any(p.token_basis == "synthetic" for p in profiles):
        footer.append(
            "Synthetic demonstration: tokens, prices and quality scores are illustrative, not benchmark results."
        )
    if any(p.source.startswith("artificial_analysis") for p in profiles):
        from .aa import ATTRIBUTION

        footer.append(ATTRIBUTION)

    ranking, skipped = _tasks.rank_tasks(
        profiles, estimator=est, zone=zone, prefill=prefill, by=by, min_score=min_score
    )
    shown = ranking if top == 0 else ranking[:top]
    t = Table(
        box=None,
        header_style="bold",
        title_justify="left",
        title=f"Footprint per task, zone {zone}, prefill {prefill}, sorted by {by}",
    )
    for c in (
        "#",
        "grade",
        "model",
        "score",
        "out tok/task",
        "Wh/task",
        "gCO2eq/task",
        "$/task",
        "pts/Wh",
        "per-token #",
    ):
        t.add_column(c, justify="left" if c in ("grade", "model") else "right")
    colors = dict(zip("ABCDE", ("green", "green", "yellow", "orange3", "red"), strict=True))
    for r in shown:
        f, p = r.footprint, r.footprint.profile
        est_mark = "~" if p.is_estimated else ""
        shift = f"{r.token_rank}" + (f" [dim]({'+' if r.shift > 0 else ''}{r.shift})[/dim]" if r.shift else "")
        t.add_row(
            str(r.rank),
            f"[{colors[r.grade]}]{r.grade}[/]",
            p.name,
            f"{p.score:.1f}" if p.score is not None else "",
            f"{est_mark}{mean(p.output_tokens):,.0f}",
            f"{mean(f.energy_wh):.3g}",
            f"{mean(f.gwp_g):.3g}",
            f"{p.cost_per_task:.3g}" if p.cost_per_task is not None else "",
            f"{f.efficiency:.3g}" if f.efficiency is not None else "",
            shift,
        )
    console.print(t)
    if any(r.footprint.profile.is_estimated for r in shown):
        footer.append("~ token usage includes assumptions (cost split or input/output ratio); see CSV provenance.")
    footer.append("Wh and gCO2eq are midpoints of the uncertainty range, --export writes both bounds.")
    footer.append(
        "per-token # ranks energy per 1,000 output tokens. Rank shift is shown only when ranking task energy."
    )
    for p, why in skipped:
        footer.append(f"skipped {p.name}: {why}")
    for line in footer:
        console.print(f"[dim]{line}[/dim]")
    if export:
        _tasks.save_ranking(ranking, export)
        console.print(f"Ranking written to {export}")


@app.command("app")
def launch_app(
    port: int = typer.Option(8501, min=1024, max=65535),
    no_browser: bool = typer.Option(False, help="Run locally without opening a browser."),
) -> None:
    """Open the dashboard on your computer (http://127.0.0.1:8501)."""
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("streamlit") is None:
        console.print('Install the app first: python -m pip install -e ".[app]"')
        raise typer.Exit(1)
    script = Path(__file__).with_name("dashboard.py")
    args = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
        f"--server.headless={'true' if no_browser else 'false'}",
    ]
    raise typer.Exit(subprocess.call(args))


@app.command()
def version() -> None:
    """Print the version and the registry data in use."""
    from .update import active_version

    data, source = active_version()
    typer.echo(f"Green AI {__version__} (model registry: EcoLogits data {data}, {source})")


@app.command()
def update(
    check: bool = typer.Option(False, "--check", help="Only report whether newer data exists."),
    clear: bool = typer.Option(False, "--clear", help="Remove downloaded data and use the bundled copy."),
    to_version: Optional[str] = typer.Option(None, "--version", help="Install a specific EcoLogits data release."),
) -> None:
    """Download the latest model registry (new models, updated electricity data)."""
    from . import update as _update

    if clear:
        console.print("Cache removed." if _update.clear_cache() else "No cache to remove.")
        return
    current, source = _update.active_version()
    try:
        if check:
            latest = _update.latest_version()
            newer = _update.parse_version(latest) > _update.parse_version(current)
            console.print(
                f"In use: {current} ({source}). Latest: {latest}. "
                + ("Run [bold]green-ai update[/bold] to install it." if newer else "Up to date.")
            )
            return
        res = _update.update(version=to_version, force=to_version is not None)
    except OSError as exc:
        console.print(f"[red]Could not reach PyPI or GitHub:[/red] {exc}")
        raise typer.Exit(1) from None
    except ValueError as exc:
        console.print(f"[red]Update rejected:[/red] {exc}")
        raise typer.Exit(1) from None
    console.print(str(res))


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
