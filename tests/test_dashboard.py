"""Execute Streamlit pages and settings without opening ports or calling APIs."""

import io
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from green_ai import Estimator, Registry, aa
from green_ai.ranges import mean
from green_ai.tasks import load_profiles, rank_tasks, ranking_rows

ENTRY = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"


def assert_healthy(app):
    assert not app.exception, [e.message for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_every_page_executes_with_the_real_streamlit_renderer():
    app = AppTest.from_file(str(ENTRY), default_timeout=20).run()
    assert_healthy(app)
    assert len(app.selectbox(key="model").options) == len(Registry.default().identifiers)
    assert "openai/gpt-6-astra" in app.selectbox(key="model").options
    for page in ("Task rankings", "Usage logs", "Model library", "Methodology"):
        app.radio(key="page").set_value(page).run()
        assert_healthy(app)


def test_sidebar_none_and_custom_input_scenarios_reach_task_rankings():
    app = AppTest.from_file(str(ENTRY), default_timeout=20).run()
    app.selectbox(key="prefill").set_value("none").run()
    app.radio(key="page").set_value("Task rankings").run()
    assert_healthy(app)
    profiles = load_profiles(ENTRY.parents[1] / "examples/task_profiles.csv", registry=Registry.default())
    rows = pd.DataFrame(
        ranking_rows(rank_tasks(profiles, estimator=Estimator(prefill="none", zone="WOR"), zone="WOR", by="carbon")[0])
    )
    assert app.dataframe[0].value.gco2eq_per_task.tolist() == pytest.approx(rows.gco2eq_per_task.tolist())
    app.selectbox(key="prefill").set_value("custom").run()
    app.number_input(key="alpha").set_value(0.02).run()
    assert_healthy(app)
    assert app.dataframe[0].value.gco2eq_per_task.sum() > rows.gco2eq_per_task.sum()


def test_request_zero_grid_override_is_respected():
    app = AppTest.from_file(str(ENTRY), default_timeout=20).run()
    app.checkbox(key="override_ci").check().run()
    app.number_input(key="ci").set_value(0.0).run()
    assert_healthy(app)
    expected = Estimator(zone="WOR", carbon_intensity=0).estimate(
        "gpt-6-astra", input_tokens=24000, cached_tokens=8000, output_tokens=600
    )
    assert app.metric[1].value.startswith(f"{mean(expected.total.gwp_g):,.3g}")


def test_aa_api_view_keeps_cost_derived_profiles_labelled(monkeypatch):
    item = {
        "slug": "gpt-6-sol",
        "name": "GPT-6 Sol",
        "model_creator": {"name": "OpenAI"},
        "evaluations": {"artificial_analysis_intelligence_index": 60},
        "pricing": {"price_1m_input_tokens": 2, "price_1m_output_tokens": 10},
        "artificial_analysis_intelligence_index_cost": {"cost_per_task": {"total_cost": 0.1}},
    }
    calls = []

    def fake_fetch(key, **kw):
        calls.append(kw)
        return {"tier": "free", "intelligence_index_version": "4.3", "data": [item]}

    monkeypatch.setattr(aa, "fetch", fake_fetch)
    app = AppTest.from_file(str(ENTRY), default_timeout=20).run()
    app.radio(key="page").set_value("Task rankings").run()
    app.radio(key="task_source").set_value("AA API").run()
    app.text_input(key="aa_key").set_value("not-a-real-key").run()
    next(b for b in app.button if b.label == "Load AA model data").click().run()
    assert_healthy(app)
    assert app.dataframe[0].value.token_basis.tolist() == ["cost-inferred"]
    assert calls == [{"tier": "auto", "use_cache": False}]


def test_hosted_demo_uses_the_deployment_key(monkeypatch):
    seen = []

    def fake_fetch(key, **kw):
        seen.append(key)
        return {"tier": "free", "intelligence_index_version": "4.3", "data": []}

    monkeypatch.setattr(aa, "fetch", fake_fetch)
    app = AppTest.from_file(str(ENTRY), default_timeout=20)
    app.secrets["AA_API_KEY"] = "deployment-key"
    app.run()
    app.radio(key="page").set_value("Task rankings").run()
    app.radio(key="task_source").set_value("AA API").run()
    assert not [t for t in app.text_input if t.key == "aa_key"]
    next(b for b in app.button if b.label == "Load AA model data").click().run()
    assert seen == ["deployment-key"]


def test_ui_csv_export_preserves_provenance_and_escapes_formulas():
    from green_ai.dashboard import _csv_bytes

    frame = pd.DataFrame([{"model": "=1+1", "token_basis": "observed", "gco2eq_min": 0.1, "gco2eq_max": 0.2}])
    row = pd.read_csv(io.BytesIO(_csv_bytes(frame))).iloc[0]
    assert row.model == "'=1+1" and row.token_basis == "observed"
