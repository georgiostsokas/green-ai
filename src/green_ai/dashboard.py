"""Local dashboard. Run with `green-ai app` or the repository launcher."""

from __future__ import annotations

import hashlib
import io
import json
import time
from html import escape
from importlib.resources import files

import altair as alt
import pandas as pd
import streamlit as st

from green_ai import DEFAULT_HARDWARE, Estimator, PrefillModel, Range, Registry, __version__, aa, batch, report
from green_ai.prefill import resolve
from green_ai.ranges import hi, lo, mean
from green_ai.tasks import CSV_ALIASES, load_profiles, normalize_column, rank_tasks, ranking_rows

GREEN, BLUE, GOLD = "#19745D", "#4D6F99", "#B98037"
CSS = """
<style>
html, body, [class*="css"] {font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;}
.stApp {background:#F6F7F4;color:#21332F;}
.block-container {max-width:1350px;padding-top:3.6rem;padding-bottom:3rem;}
header[data-testid="stHeader"] {background:rgba(246,247,244,.94);}
section[data-testid="stSidebar"] {background:#EAEFE9;border-right:1px solid #DAE2D9;}
section[data-testid="stSidebar"] .block-container {padding-top:1.7rem;}
h1,h2,h3 {font-weight:650!important;letter-spacing:-.045em;}
h1 {font-size:3rem!important;line-height:1.08!important;}
h2 {font-size:1.55rem!important;}
h3 {font-size:1.12rem!important;letter-spacing:-.025em;}
.brand {display:flex;align-items:center;gap:.65rem;font-size:1.25rem;font-weight:750;letter-spacing:-.055em;}
.brand-mark {display:inline-flex;align-items:center;justify-content:center;background:#19745D;color:white;
 width:31px;height:35px;border-radius:10px 10px 10px 2px;letter-spacing:-.08em;}
.eyebrow {color:#697D73;text-transform:uppercase;letter-spacing:.15em;font-size:.69rem;font-weight:700;}
.hero-note {color:#64776D;font-size:.98rem;line-height:1.6;max-width:710px;}
.tag {display:inline-block;padding:5px 10px;border:1px solid #CFDCD0;border-radius:20px;
 font-size:.69rem;color:#52705D;background:#F0F5EE;margin:0 5px 6px 0;}
.hero {margin:1.3rem 0 1.65rem;}
div[data-testid="stMetric"] {background:#FFF;border:1px solid #DDE5DB;border-radius:15px;padding:1.15rem 1.25rem;}
div[data-testid="stMetricValue"] {font-size:1.9rem;font-weight:650;letter-spacing:-.055em;}
div[data-testid="stMetricLabel"] {color:#6B7B71;font-size:.8rem;}
div[data-testid="stMetricDelta"] {font-size:.75rem;}
div[data-testid="stRadio"] > div {gap:.25rem;}
div[data-testid="stRadio"] label {background:#EDF1EA;border-radius:10px;padding:.35rem .7rem!important;}
.stButton > button {border-radius:9px;font-weight:600;border-color:#C8D8CA;}
.stDownloadButton > button {border-radius:9px;}
div[data-testid="stDataFrame"] {border-radius:13px;border:1px solid #DEE6DA;overflow:hidden;}
.note-box {padding:1rem 1.2rem;background:#ECF2E9;border:1px solid #D8E4D2;border-radius:13px;
 font-size:.83rem;line-height:1.6;color:#4F6959;margin:1rem 0;}
.footer {border-top:1px solid #DCE4D9;padding-top:1.1rem;margin-top:2rem;font-size:.73rem;color:#748276;}
</style>
"""


def _asset(name: str) -> bytes:
    return files("green_ai").joinpath("data", "examples", name).read_bytes()


def _fmt(v, unit="", scale=1):
    return f"{mean(v) * scale:,.3g} {unit}".strip()


def _compact(x: float) -> str:
    """Readable large numbers for metric cards: 2,450 / 12.4k / 3.1M."""
    if x >= 1e6:
        return f"{x / 1e6:.3g}M"
    if x >= 1e5:
        return f"{x / 1e3:.3g}k"
    if x >= 100:
        return f"{x:,.0f}"
    return f"{x:.3g}"


def _interval(v, scale=1):
    return f"{lo(v) * scale:,.3g}–{hi(v) * scale:,.3g}"


def _hero(kicker, title, note):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{escape(kicker)}</div>'
        f'<h1>{escape(title)}</h1><p class="hero-note">{escape(note)}</p></div>',
        unsafe_allow_html=True,
    )


def _assumptions():
    # Fresh registry and estimator per rerun/session: uploads cannot mutate another visitor's data.
    reg = Registry.default()
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">G</span>Green AI</div>', unsafe_allow_html=True)
        st.caption(f"Carbon footprint estimator for LLMs · v{__version__}")
        st.divider()
        st.markdown("### Your assumptions")
        scenario = st.selectbox("Input correction", ["central", "low", "none", "custom"], key="prefill")
        alpha = (
            st.number_input(
                "GPU input/decode ratio α",
                min_value=0.0,
                max_value=1.0,
                value=0.008,
                step=0.001,
                format="%.3f",
                key="alpha",
            )
            if scenario == "custom"
            else resolve(scenario).alpha
        )
        cache = st.slider(
            "Cache energy factor",
            0.0,
            1.0,
            0.1,
            0.05,
            key="cache_factor",
            help="Scenario for the residual cost of cache reads; not measured from API prices.",
        )
        pm = PrefillModel(scenario, alpha, cache)
        zone = st.selectbox(
            "Electricity mix",
            ["WOR", "Provider default"] + [z for z in sorted(reg.zones) if z != "WOR"],
            key="zone",
            help="A common mix makes grid assumptions comparable. PUE remains provider-specific.",
        )
        with st.expander("Deployment settings"):
            override_ci = st.checkbox("Override grid carbon intensity", key="override_ci")
            ci = (
                st.number_input("gCO₂eq / kWh", min_value=0.0, value=400.0, step=10.0, key="ci")
                if override_ci
                else None
            )
            override_pue = st.checkbox("Use a common PUE", key="override_pue")
            pue = st.number_input("PUE", min_value=1.0, value=1.2, step=0.05, key="pue") if override_pue else None
            size = st.select_slider(
                "Concurrent requests / server", options=[8, 16, 32, 64, 128, 256], value=64, key="batch_size"
            )
            custom = st.file_uploader("Custom model configuration", type=["yaml", "yml", "json"], key="custom_models")
            if custom:
                import yaml

                try:
                    reg.load_custom_data(yaml.safe_load(custom.getvalue()))
                    st.caption("Custom configuration applied to this session.")
                except (ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
                    st.error(f"Invalid model configuration: {exc}")
                    st.stop()
        st.divider()
        st.caption(
            "Estimates for text inference. Input correction is experimental. "
            "Ranges express assumptions, not confidence intervals."
        )
        st.caption(
            "Uploads are processed in server memory. When hosted elsewhere, the host receives the uploaded bytes."
        )
    est = Estimator(
        registry=reg,
        prefill=pm,
        zone=None if zone == "Provider default" else zone,
        pue=pue,
        carbon_intensity=ci / 1000 if ci is not None else None,
        hardware=DEFAULT_HARDWARE.with_overrides(batch_size=size),
    )
    return reg, est


def _explore(reg, est):
    _hero(
        "Request explorer",
        "Every prompt has a footprint.",
        "Explore electricity, greenhouse gases and water, and see how prompt processing changes the estimate.",
    )
    st.markdown(
        f'<span class="tag">{len(reg.models)} models + {len(reg.identifiers) - len(reg.models)} aliases</span>'
        '<span class="tag">Request-level estimates</span><span class="tag">Explicit assumptions</span>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 2.1], gap="large")
    with left, st.container(border=True):
        st.markdown("### Build a request")
        ids = reg.identifiers
        choice = st.selectbox("Model", ids, index=ids.index("openai/gpt-6-astra"), key="model")
        provider, model = choice.split("/", 1)
        inp = st.number_input(
            "Input tokens · includes cache",
            min_value=0,
            max_value=10_000_000,
            value=24_000,
            step=1000,
            key="input_tokens",
        )
        if st.session_state.get("cached_tokens", 8000) > inp:
            st.session_state.cached_tokens = inp
        cached = st.number_input(
            "Cached input tokens",
            min_value=0,
            max_value=int(inp),
            value=min(8000, int(inp)),
            step=1000,
            key="cached_tokens",
        )
        out = st.number_input(
            "Output tokens · includes reasoning",
            min_value=0,
            max_value=1_000_000,
            value=600,
            step=100,
            key="output_tokens",
        )
    e = est.estimate(model, provider=provider, input_tokens=inp, cached_tokens=cached, output_tokens=out)
    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Electricity / request",
            _fmt(e.total.energy_wh, "Wh"),
            f"{e.correction_factor - 1:+.1%} with input",
            delta_color="off",
        )
        c2.metric("GHG / request · gCO₂eq", _fmt(e.total.gwp_g))
        c3.metric("Water / request", _fmt(e.total.water_l, "mL", 1000))
        st.caption(
            f"Assumption ranges: {_interval(e.total.energy_wh)} Wh · {_interval(e.total.gwp_g)} gCO₂eq · "
            f"{_interval(e.total.water_l, 1000)} mL"
        )
        chart = []
        cache_share = cached / inp if inp else 0
        for n in (0, 1000, 4000, 8000, 16000, 24000, 32000, 48000, 64000, 96000, 128000):
            for name, pm in (("Reference", PrefillModel("none", 0)), ("Your scenario", est.prefill)):
                x = est.estimate(
                    model,
                    provider=provider,
                    input_tokens=n,
                    cached_tokens=n * cache_share,
                    output_tokens=out,
                    prefill=pm,
                )
                chart.append(
                    {
                        "Input tokens": n,
                        "Scenario": name,
                        "Wh": mean(x.total.energy_wh),
                        "Low": lo(x.total.energy_wh),
                        "High": hi(x.total.energy_wh),
                    }
                )
        st.markdown("### The cost of a longer prompt")
        data = alt.Chart(pd.DataFrame(chart)).encode(
            x=alt.X("Input tokens:Q", title="Input tokens"),
            color=alt.Color(
                "Scenario:N",
                scale=alt.Scale(domain=["Reference", "Your scenario"], range=[BLUE, GREEN]),
                legend=alt.Legend(orient="top", title=None, symbolType="stroke", symbolStrokeWidth=3, symbolOpacity=1),
            ),
        )
        st.altair_chart(
            (
                data.mark_area(opacity=0.08).encode(y="Low:Q", y2="High:Q")
                + data.mark_line(strokeWidth=2.5).encode(
                    y=alt.Y("Wh:Q", title="Electricity (Wh)"),
                    tooltip=["Input tokens", "Scenario", alt.Tooltip("Wh:Q", format=".4g")],
                )
            ).properties(height=240),
            width="stretch",
        )
    breakdown = []
    for label, impacts in (("Reference", e.baseline), ("Input correction", e.prefill), ("Total", e.total)):
        breakdown.append(
            {
                "Component": label,
                "Electricity (Wh)": _fmt(impacts.energy_wh),
                "GHG (gCO₂eq)": _fmt(impacts.gwp_g),
                "Water (mL)": _fmt(impacts.water_l, scale=1000),
            }
        )
    st.dataframe(pd.DataFrame(breakdown), hide_index=True, width="stretch")
    st.markdown(
        '<div class="note-box">Input correction scales GPU electricity only. Baseline server electricity and '
        "embodied impacts stay unchanged. The same assumptions apply across every view.</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Sources, assumptions & export"):
        for warning in e.warnings:
            st.caption(warning)
        st.write(f"Model match: {e.match.status.value} · zone {e.mix.zone} · PUE {e.pue} · WUE {e.wue}")
        st.download_button(
            "Download request JSON",
            json.dumps(e.to_dict(), indent=2),
            "green_ai_request.json",
            mime="application/json",
        )


def _task_upload(reg, aa_export):
    up = st.file_uploader("Per-task data · CSV", type="csv", key="task_upload")
    st.caption("Use actual per-task averages, not total benchmark token usage. Output tokens include reasoning.")
    template = _asset("task_profiles.csv")
    st.download_button("Download input template", template, "task_profiles_template.csv", mime="text/csv")
    if not up:
        return []
    df = pd.read_csv(io.BytesIO(up.getvalue()), comment="#")
    cols = list(df.columns)
    normalized = {normalize_column(c): c for c in cols}
    mapping = {}
    with st.expander("Map your columns", expanded=True):
        for key in (
            "model",
            "provider",
            "input_tokens",
            "output_tokens",
            "answer_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "score",
            "cost_per_task",
            "requests_per_task",
            "index_version",
        ):
            detected = next((normalized[n] for n in CSV_ALIASES[key] if n in normalized), None)
            options = ["Not provided"] + cols
            selected = st.selectbox(
                key.replace("_", " "), options, index=options.index(detected) if detected else 0, key=f"map_{key}"
            )
            mapping[key] = None if selected == "Not provided" else selected
    ratio = None
    if not mapping["input_tokens"]:
        st.info("Output-only data: choose an input/output ratio scenario. Input tokens will be marked as assumed.")
        rlo, rhi = st.slider("Assumed input tokens / output token", 0.0, 300.0, (10.0, 50.0), 1.0, key="input_ratio")
        ratio = Range(rlo, rhi)
    includes_reasoning = st.checkbox("Output column already includes reasoning", value=True, key="includes_reasoning")
    return load_profiles(
        io.BytesIO(up.getvalue()),
        registry=reg,
        column_map=mapping,
        source="artificial_analysis:export" if aa_export else "user:csv",
        input_ratio=ratio,
        output_includes_reasoning=includes_reasoning,
    )


def _deployment_aa_key():
    """Key set by the host (Streamlit secrets), so a public demo can load AA data without asking visitors."""
    try:
        return st.secrets.get("AA_API_KEY") or None
    except Exception:  # noqa: BLE001 - no secrets file is the normal local case
        return None


def _aa_profiles(reg):
    st.caption(
        "Live data uses your own account. API keys and responses remain in this app session; no disk cache is used."
    )
    deployment_key = _deployment_aa_key()
    if deployment_key:
        st.caption("Using the Artificial Analysis key configured for this deployment.")
        key = deployment_key
    else:
        key = st.text_input("Artificial Analysis API key", type="password", key="aa_key")
    tier = st.selectbox("API access", ["auto", "free", "pro"], key="aa_tier")
    identity = hashlib.sha256(f"{tier}:{key}".encode()).hexdigest()
    if st.session_state.get("aa_identity") != identity:
        for field in ("aa_payload", "aa_fetched", "aa_detail_payload"):
            st.session_state.pop(field, None)
        st.session_state.aa_identity = identity
    if st.button("Load AA model data", type="primary", disabled=not key):
        with st.spinner("Fetching Artificial Analysis data…"):
            st.session_state.aa_payload = aa.fetch(key, tier=tier, use_cache=False)
            st.session_state.aa_fetched = time.time()
            st.session_state.pop("aa_detail_payload", None)
    payload = st.session_state.get("aa_payload")
    if not payload:
        st.info("Connect your API key, or choose AA token export to use data you have already downloaded.")
        return []
    if time.time() - st.session_state.get("aa_fetched", 0) > 24 * 3600:
        st.warning("This snapshot is older than 24 hours. Reload it before comparing current models.")
        return []
    st.caption(
        f"Artificial Analysis · Intelligence Index {payload.get('intelligence_index_version')} "
        f"· {payload.get('tier')} tier"
    )
    plan_file = st.file_uploader(
        "Optional benchmark plan · JSON",
        type="json",
        key="aa_plan",
        help="For Pro evaluation counts: supply explicit versioned weights and task denominators.",
    )
    plan = json.loads(plan_file.getvalue()) if plan_file else None
    if plan is not None:
        candidates = {i["slug"]: i.get("name", i["slug"]) for i in payload["data"] if i.get("slug")}
        selected = st.multiselect(
            "Fetch evaluation details for these models",
            list(candidates),
            format_func=lambda s: candidates[s],
            max_selections=20,
            key="aa_details",
        )
        if st.button("Load selected evaluation counts", disabled=not selected or not key):
            st.session_state.aa_detail_payload = aa.fetch_details(
                selected, key, index_version=payload.get("intelligence_index_version")
            )
        details = st.session_state.get("aa_detail_payload")
        if not details or {i.get("slug") for i in details.get("data", [])} != set(selected):
            st.caption("Load evaluation counts for the currently selected models.")
            return []
        converted = aa.to_profiles(details, registry=reg, benchmark_plan=plan)
    else:
        share = st.slider("Assumed share of task cost due to output tokens", 0.0, 1.0, (0.5, 0.9), 0.05, key="aa_share")
        converted = aa.to_profiles(payload, registry=reg, output_share=Range(*share))
        st.info(
            "API totals are not tokens per task. Without a benchmark plan, these are cost-derived scenarios, "
            "assuming fresh input and one call per task. For observed tokens, use the AA export option."
        )
    if converted.unmatched:
        with st.expander(f"{len(converted.unmatched)} excluded models & reasons"):
            st.dataframe(pd.DataFrame(converted.unmatched, columns=["Model", "Reason"]), hide_index=True)
    st.caption(converted.attribution)
    return converted.profiles


def _tasks(reg, est):
    _hero(
        "Task intelligence",
        "Compare the work. Count the impact.",
        "Rank estimated CO₂, electricity and water per task, with quality and token provenance alongside every result.",
    )
    src = st.radio(
        "Task data",
        ["Example workload", "AA token export", "AA API", "Your workload"],
        horizontal=True,
        key="task_source",
    )
    try:
        if src == "Example workload":
            profiles = load_profiles(io.BytesIO(_asset("task_profiles.csv")), registry=reg)
            st.info("Synthetic demonstration: token counts and quality scores are illustrative, not benchmark results.")
        elif src == "AA API":
            profiles = _aa_profiles(reg)
        else:
            profiles = _task_upload(reg, src == "AA token export")
    except (ValueError, KeyError, TypeError, aa.AAError, OSError) as exc:
        st.error(str(exc))
        return
    if not profiles:
        return
    suites = sorted({(p.task_set, p.index_version or "unspecified") for p in profiles})
    if len(suites) > 1:
        chosen = st.selectbox("Compare one task set / version", suites, format_func=lambda s: f"{s[0]} · {s[1]}")
        profiles = [p for p in profiles if (p.task_set, p.index_version or "unspecified") == chosen]
    c1, c2, c3 = st.columns([1.4, 1, 1])
    sorts = {
        "CO₂ / task": "carbon",
        "Electricity / task": "energy",
        "Water / task": "water",
        "Quality / Wh": "efficiency",
    }
    sort = c1.selectbox("Rank by", list(sorts), key="task_sort")
    threshold = c2.number_input("Minimum quality score", min_value=0.0, value=0.0, step=5.0, key="min_score")
    pareto_only = c3.checkbox("Show Pareto choices only", key="pareto_only")
    ranking, skipped = rank_tasks(profiles, estimator=est, zone=est.zone, by=sorts[sort], min_score=threshold or None)
    shown = [r for r in ranking if r.pareto] if pareto_only else ranking
    if not shown:
        st.warning("No matched models meet the selected criteria.")
        return
    rows = pd.DataFrame(ranking_rows(shown))
    best = shown[0]
    a, b, c = st.columns(3)
    a.metric("Models compared", len(shown))
    b.metric("First result · gCO₂eq / task", _fmt(best.footprint.gwp_g))
    c.metric("First result · electricity / task", _fmt(best.footprint.energy_wh, "Wh"))
    if any(p.is_estimated for p in profiles):
        st.warning(
            "Some token profiles include assumptions. Compare their ranges and provenance, not only the midpoint rank."
        )
    st.caption(
        "Ranked by assumption midpoints. Overlapping ranges do not establish a reliable order. "
        "Grades A–E are relative to this set. Pareto choices use score and the selected impact (energy for quality/Wh)."
    )
    columns = [
        "rank",
        "model",
        "score",
        "gco2eq_per_task",
        "gco2eq_min",
        "gco2eq_max",
        "energy_wh_per_task",
        "output_tokens_per_task",
        "points_per_wh",
        "token_basis",
        "pareto",
    ]
    st.dataframe(
        rows[columns],
        hide_index=True,
        width="stretch",
        column_config={
            "rank": st.column_config.NumberColumn("#", format="%d"),
            "gco2eq_per_task": st.column_config.NumberColumn("gCO₂eq / task", format="%.3f"),
            "gco2eq_min": st.column_config.NumberColumn("GHG low", format="%.3f"),
            "gco2eq_max": st.column_config.NumberColumn("GHG high", format="%.3f"),
            "energy_wh_per_task": st.column_config.NumberColumn("Wh / task", format="%.3f"),
            "points_per_wh": st.column_config.NumberColumn("Quality / Wh", format="%.3f"),
            "output_tokens_per_task": st.column_config.NumberColumn("Output / task", format="%.0f"),
        },
    )
    if rows.score.notna().any():
        x, title = {
            "carbon": ("gco2eq_per_task", "GHG (gCO₂eq / task)"),
            "water": ("water_ml_per_task", "Water (mL / task)"),
        }.get(sorts[sort], ("energy_wh_per_task", "Electricity (Wh / task)"))
        chart = (
            alt.Chart(rows)
            .mark_circle(size=130, opacity=0.8)
            .encode(
                x=alt.X(f"{x}:Q", title=title),
                y=alt.Y("score:Q", title="Quality score", scale=alt.Scale(zero=False)),
                color=alt.Color("token_basis:N", scale=alt.Scale(range=[GREEN, BLUE, GOLD])),
                tooltip=["model", "score", alt.Tooltip(f"{x}:Q", format=".4g"), "token_basis", "pareto"],
            )
        )
        st.altair_chart(chart.properties(height=290), width="stretch")
    st.download_button("Download full ranking · CSV", _csv_bytes(rows), "green_ai_task_ranking.csv", mime="text/csv")
    with st.expander("Token profiles & assumptions"):
        st.dataframe(
            rows[
                [
                    "model",
                    "input_tokens_per_task",
                    "output_tokens_per_task",
                    "requests_per_task",
                    "electricity_zone",
                    "prefill_alpha",
                    "token_basis",
                    "source",
                    "note",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Water covers cooling and electricity generation. GHG includes baseline usage and embodied allocation. "
            "Input correction changes operational impacts only. "
            "Source: Artificial Analysis where selected; own data otherwise."
        )
        if skipped:
            st.dataframe(
                pd.DataFrame([(p.name, why) for p, why in skipped], columns=["Excluded", "Reason"]), hide_index=True
            )


def _csv_bytes(df):
    safe = df.copy()
    for col in safe:
        safe[col] = safe[col].map(lambda v: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v)
    return safe.to_csv(index=False).encode()


def _logs(reg, est):
    _hero(
        "Usage analytics",
        "Find where your footprint concentrates.",
        "Inspect usage by month, model and provider. Excluded models are reported; invalid counts stop the analysis.",
    )
    up = st.file_uploader("Upload usage data", type=["csv", "xlsx", "parquet", "json", "jsonl"], key="usage_upload")
    demo = st.toggle("Use synthetic demonstration", value=up is None, key="usage_demo")
    if up is not None and not demo:
        data = batch.read_table(io.BytesIO(up.getvalue()), filename=up.name)
    elif demo:
        reg.load_custom_data(
            {
                "providers": {"onprem": {"location": "FRA", "pue": {"min": 1.2, "max": 1.5}, "wue": 0}},
                "models": [{"name": "acme-internal-qwen", "provider": "onprem", "total_params": 32}],
            }
        )
        data = batch.read_table(io.BytesIO(_asset("sample_usage.csv")), filename="sample_usage.csv")
        st.caption("All demonstration usage is generated synthetically.")
    else:
        return
    mapping = {}
    with st.expander("Column mapping"):
        try:
            detected = batch.ColumnMap.detect(list(data.columns))
        except ValueError:
            detected = batch.ColumnMap(input_tokens=None)
        for key in ("model", "input_tokens", "output_tokens", "cached_tokens", "requests", "provider", "date", "zone"):
            options = ["Not provided"] + list(data.columns)
            current = getattr(detected, key, None)
            col = st.selectbox(
                key.replace("_", " "),
                options,
                index=options.index(current) if current in options else 0,
                key=f"log_map_{key}",
            )
            mapping[key] = None if col == "Not provided" else col
    if not mapping["model"] or not mapping["output_tokens"]:
        st.info("Choose a model and output-token column to analyze this log.")
        return
    result = batch.run(data, estimator=est, columns=batch.ColumnMap(**mapping))
    t = result.totals()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Electricity · kWh", _compact(t["total_energy_kwh"]))
    c2.metric("GHG · kgCO₂eq", _compact(t["total_gwp_kg"]))
    c3.metric("Water · litres", _compact(t["total_water_l"]))
    denom = t["output_tokens"] + t["unmatched_output_tokens"]
    c4.metric("Output-token coverage", f"{t['output_tokens'] / denom:.1%}" if denom else "No output tokens")
    if not result.unmatched.empty:
        st.warning("Some models are excluded. Totals cover matched usage only.")
        st.dataframe(result.unmatched, hide_index=True, width="stretch")
    if "month" in result.rows:
        monthly = (
            result.by_month()[["month", "baseline_energy_kwh", "prefill_energy_kwh"]]
            .rename(columns={"baseline_energy_kwh": "Reference", "prefill_energy_kwh": "Input correction"})
            .melt("month", var_name="Component", value_name="kWh")
        )
        chart = (
            alt.Chart(monthly)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("month:N", title=None),
                y=alt.Y("kWh:Q", title="Electricity (kWh)"),
                color=alt.Color(
                    "Component:N",
                    scale=alt.Scale(domain=["Reference", "Input correction"], range=[BLUE, GREEN]),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=["month", "Component", "kWh"],
            )
        )
        st.altair_chart(chart.properties(height=260), width="stretch")
    st.markdown("### Impact by model")
    st.dataframe(
        result.by_model()[
            [
                "matched_provider",
                "matched_model",
                "requests",
                "input_tokens",
                "output_tokens",
                "total_energy_kwh",
                "total_gwp_kg",
                "total_water_l",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "matched_provider": st.column_config.TextColumn("Provider"),
            "matched_model": st.column_config.TextColumn("Model"),
            "requests": st.column_config.NumberColumn("Requests", format="localized"),
            "input_tokens": st.column_config.NumberColumn("Input tokens", format="compact"),
            "output_tokens": st.column_config.NumberColumn("Output tokens", format="compact"),
            "total_energy_kwh": st.column_config.NumberColumn("Electricity (kWh)", format="%.1f"),
            "total_gwp_kg": st.column_config.NumberColumn("GHG (kgCO₂eq)", format="%.1f"),
            "total_water_l": st.column_config.NumberColumn("Water (L)", format="%.0f"),
        },
    )
    st.caption("Excel includes normalized token counts and impact results. Unrelated original log columns are omitted.")
    buf = io.BytesIO()
    report.to_excel(result, buf)
    st.download_button(
        "Download analysis · Excel",
        buf.getvalue(),
        "green_ai_footprint.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Download normalized rows · CSV", _csv_bytes(result.export_rows()), "green_ai_rows.csv", mime="text/csv"
    )


def _library(reg):
    _hero(
        "Model registry",
        "A transparent starting point.",
        "Inspect architecture and deployment assumptions. Undisclosed model sizes remain ranges.",
    )
    c1, c2 = st.columns([2, 1])
    query = c1.text_input("Search model", placeholder="Astra, Sol, Opus, Gemini…", key="library_search")
    provider = c2.selectbox("Provider", ["All"] + sorted(reg.providers), key="library_provider")
    entries = []
    for m in reg.search(query, None if provider == "All" else provider):
        entries.append(
            {
                "Provider": m.provider,
                "Model": m.name,
                "Architecture": m.architecture,
                "Total parameters (B)": str(m.total_params_b),
                "Active parameters (B)": str(m.active_params_b),
                "Tokens/s": m.tps,
                "TTFT (s)": m.ttft,
                "Origin": m.origin,
            }
        )
    st.caption(
        f"Registry {reg.data_version} · {len(reg.models)} canonical models · "
        f"{len(reg.identifiers) - len(reg.models)} aliases · {len(reg.zones)} electricity mixes"
    )
    st.dataframe(pd.DataFrame(entries), hide_index=True, width="stretch", height=520)
    st.download_button(
        "Download model assumptions", _csv_bytes(pd.DataFrame(entries)), "green_ai_models.csv", mime="text/csv"
    )
    st.caption("Source: EcoLogits public model and electricity dataset (MPL-2.0). Custom models are session-specific.")


def _methodology():
    _hero(
        "Method & evidence",
        "Make every assumption visible.",
        "A reproducible reference model with an explicit, experimental input correction.",
    )
    text = files("green_ai").joinpath("data/methodology.md").read_text(encoding="utf-8")
    text = text.replace("[AA integration](AA_INTEGRATION.md)", "the AA integration guide below")
    st.markdown(text)
    with st.expander("AA integration guide"):
        st.markdown(files("green_ai").joinpath("data/aa_integration.md").read_text(encoding="utf-8"))


def render():
    st.set_page_config(page_title="Green AI · LLM carbon footprint estimator", page_icon="🌿", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    reg, est = _assumptions()
    st.markdown('<div class="eyebrow">Green AI · LLM carbon footprint estimator</div>', unsafe_allow_html=True)
    page = st.radio(
        "Workspace",
        ["Explore", "Task rankings", "Usage logs", "Model library", "Methodology"],
        horizontal=True,
        label_visibility="collapsed",
        key="page",
    )
    try:
        {
            "Explore": lambda: _explore(reg, est),
            "Task rankings": lambda: _tasks(reg, est),
            "Usage logs": lambda: _logs(reg, est),
            "Model library": lambda: _library(reg),
            "Methodology": _methodology,
        }[page]()
    except (ValueError, LookupError, KeyError, TypeError, ImportError, OSError) as exc:
        st.error(str(exc))
    st.markdown(
        '<div class="footer">Green AI · Energy, carbon & water estimates for informed comparisons. '
        "Text inference scope · Experimental input correction · Sources in Methodology.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    render()
