"""Regression cases discovered in review; tests exercise externally visible failures."""

import io
import math

import pandas as pd
import pytest

from green_ai import Estimator, Range, Registry, Tracker, aa, batch, report
from green_ai.ranges import mean
from green_ai.rating import rate_models
from green_ai.tasks import TaskProfile, load_profiles, rank_tasks, ranking_rows, task_footprint


@pytest.mark.parametrize(
    "column,value",
    [
        ("output_tokens", "oops"),
        ("input_tokens", None),
        ("cached_tokens", -1),
        ("cached_tokens", 2000),
        ("requests", -1),
        ("requests", 0),
        ("output_tokens", math.inf),
        ("output_tokens", math.nan),
    ],
)
def test_invalid_log_counts_cannot_silently_enter_totals(column, value):
    row = {"model": "gpt-4o", "input_tokens": 1000, "output_tokens": 100, "cached_tokens": 0, "requests": 1}
    row[column] = value
    with pytest.raises(ValueError):
        batch.run(pd.DataFrame([row]))


def test_concatenated_logs_with_duplicate_index_work():
    d = pd.DataFrame([{"model": "gpt-4o", "output_tokens": 100}] * 2, index=[0, 0])
    result = batch.run(d)
    assert result.totals()["rows_matched"] == 2
    assert list(result.rows.source_row) == [1, 2]


def test_original_log_columns_are_not_exported_by_default():
    r = batch.run(pd.DataFrame([{"model": "gpt-4o", "output_tokens": 100, "employee": "PRIVATE"}]))
    assert "employee" not in r.export_rows()
    assert "employee" in r.export_rows(include_original=True)
    b = io.BytesIO()
    report.to_excel(r, b)
    b.seek(0)
    assert "employee" not in pd.read_excel(b, sheet_name="Rows")


def test_formula_text_is_not_executable_in_excel():
    r = batch.run(pd.DataFrame([{"model": "=1+1", "output_tokens": 100}]))
    b = io.BytesIO()
    report.to_excel(r, b)
    from openpyxl import load_workbook

    b.seek(0)
    workbook = load_workbook(b)
    assert all(c.data_type != "f" for ws in workbook for row in ws for c in row)


@pytest.mark.parametrize(
    "model,usage",
    [
        ("mistral-large-latest", {"prompt_tokens": 1000, "completion_tokens": 100}),
        ("claude-opus-5-5", {"input_tokens": 1000, "output_tokens": 100}),
    ],
)
def test_tracker_infers_provider_without_cache_fields(model, usage):
    assert Tracker().add({"model": model, "usage": usage}).output_tokens == 100


@pytest.mark.parametrize("model", ["gpt-4o", "claude-opus-5-5"])
def test_tracker_rejects_missing_usage(model):
    with pytest.raises(ValueError):
        Tracker().add({"model": model})


def test_custom_model_override_invalidates_cached_estimates():
    reg = Registry.default()
    reg.register("toy", total_params=7, provider="lab")
    est = Estimator(registry=reg)
    before = mean(est.estimate("toy", provider="lab", output_tokens=1000).total.energy_wh)
    reg.register("toy", total_params=700, provider="lab", replace=True)
    after = mean(est.estimate("toy", provider="lab", output_tokens=1000).total.energy_wh)
    assert after > before * 10
    assert after == mean(Estimator(registry=reg).estimate("toy", provider="lab", output_tokens=1000).total.energy_wh)


def test_input_scenario_is_shared_by_task_and_model_comparison():
    est = Estimator(prefill="none")
    p = TaskProfile.exact("gpt-4o", 50000, 500)
    expected = mean(est.estimate("gpt-4o", input_tokens=50000, output_tokens=500).total.energy_wh)
    assert mean(task_footprint(p, estimator=est).energy_wh) == expected
    assert rate_models(estimator=est, models=["gpt-4o"], input_tokens=50000, output_tokens=500)[0].energy_wh == expected


def test_agent_task_counts_all_calls():
    p = TaskProfile.exact("gpt-4o", 40000, 1000, requests_per_task=12)
    est = Estimator()
    f = task_footprint(p, estimator=est)
    assert mean(f.energy_wh) == mean(
        est.estimate("gpt-4o", input_tokens=40000, output_tokens=1000, requests=12).total.energy_wh
    )
    assert mean(f.energy_wh) > mean(task_footprint(TaskProfile.exact("gpt-4o", 40000, 1000), estimator=est).energy_wh)


def test_carbon_ranking_quality_filter_and_all_bounds_export():
    profiles = [
        TaskProfile.exact("gpt-4o-mini", 1000, 100, score=10),
        TaskProfile.exact("gpt-4o", 1000, 100, score=60),
        TaskProfile.exact("gpt-5", 1000, 100, score=70),
    ]
    ranks, skipped = rank_tasks(profiles, by="carbon", min_score=50)
    assert len(ranks) == 2 and len(skipped) == 1
    assert [mean(r.footprint.gwp_g) for r in ranks] == sorted(mean(r.footprint.gwp_g) for r in ranks)
    assert all(r.shift is None for r in ranks)
    assert {"gco2eq_min", "gco2eq_max", "token_basis", "source"} <= set(ranking_rows(ranks)[0])


def test_equal_estimates_do_not_get_different_grades():
    ranks, _ = rank_tasks(
        [TaskProfile.exact("gpt-4o", 1000, 100, label="A"), TaskProfile.exact("gpt-4o", 1000, 100, label="B")]
    )
    assert ranks[0].rank == ranks[1].rank
    assert ranks[0].grade == ranks[1].grade


def test_different_task_sets_cannot_be_ranked_together():
    with pytest.raises(ValueError, match="task set"):
        rank_tasks(
            [
                TaskProfile.exact("gpt-4o", 100, 100, task_set="coding"),
                TaskProfile.exact("gpt-5", 100, 100, task_set="writing"),
            ]
        )


def test_aa_evaluation_weights_do_not_infer_tasks_from_cost():
    # Two equal-weight evaluations, with 100 tasks in A and one task in B.
    item = {
        "slug": "gpt-4o",
        "name": "GPT-4o",
        "model_creator": {"name": "OpenAI"},
        "evaluations": {"artificial_analysis_intelligence_index": 50},
        "artificial_analysis_intelligence_index_cost": {"total_cost": 1.0201, "cost_per_task": {"total_cost": 0.0101}},
        "evaluation_token_counts": {
            "A": {"input_tokens": 1000000, "output_tokens": 10000},
            "B": {"input_tokens": 100, "output_tokens": 10000},
        },
    }
    plan = {
        "index_version": "4.3",
        "benchmarks": {"A": {"weight": 0.5, "tasks": 100}, "B": {"weight": 0.5, "tasks": 1}},
    }
    p = aa.to_profiles(
        {"tier": "pro", "intelligence_index_version": "4.3", "data": [item]}, benchmark_plan=plan
    ).profiles[0]
    assert p.input_tokens == 5050 and p.output_tokens == 5050
    assert p.token_basis == "benchmark-weighted" and not p.is_estimated
    plan["index_version"] = "old"
    result = aa.to_profiles({"intelligence_index_version": "4.3", "data": [item]}, benchmark_plan=plan)
    assert not result.profiles and "versions differ" in result.unmatched[0][1]


def test_actual_aa_per_task_export_includes_reasoning_once():
    data = io.StringIO(
        "Model,Input Tokens Per Task,Answer Tokens Per Task,Reasoning Tokens Per Task,Intelligence Index\n"
        "GPT-6 Sol (high),40000,500,1500,60\n"
    )
    p = load_profiles(data, source="artificial_analysis:export", registry=Registry.default())[0]
    assert p.model == "gpt-6-sol" and p.output_tokens == 2000 and not p.is_estimated
    assert p.score == 60


def test_output_only_aa_export_needs_an_explicit_input_scenario():
    text = "model,output_tokens_per_task,score\ngpt-6-astra,2000,70\n"
    with pytest.raises(ValueError):
        load_profiles(io.StringIO(text))
    p = load_profiles(io.StringIO(text), input_ratio=Range(10, 50))[0]
    assert p.scenarios == ((20000, 2000), (100000, 2000))
    assert p.token_basis == "output-observed/input-assumed" and p.is_estimated


def test_aa_no_disk_cache_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("GREEN_AI_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(aa, "_fetch_all", lambda path, key: {"tier": "pro", "data": []})
    aa.fetch("test-key", use_cache=False)
    assert not list(tmp_path.iterdir())


def test_aa_cache_is_isolated_by_key_and_tier(monkeypatch, tmp_path):
    monkeypatch.setenv("GREEN_AI_CACHE_DIR", str(tmp_path))
    calls = []

    def fake(path, key):
        calls.append((path, key))
        return {"tier": "pro", "data": []}

    monkeypatch.setattr(aa, "_fetch_all", fake)
    aa.fetch("one")
    aa.fetch("one")
    aa.fetch("two")
    aa.fetch("two", tier="free")
    assert len(calls) == 3


@pytest.mark.parametrize(
    "params",
    [
        {"total_params": -1},
        {"total_params": 10, "active_params": 20},
        {"total_params": math.nan},
        {"total_params": 10, "tps": 0},
    ],
)
def test_invalid_custom_architectures_are_rejected(params):
    with pytest.raises(ValueError):
        Registry.default().register("bad", **params)


def test_reference_update_stops_when_equations_change(monkeypatch, tmp_path):
    from green_ai import update

    monkeypatch.setenv("GREEN_AI_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(update, "_get", lambda url: b"changed upstream source")
    with pytest.raises(ValueError, match="equations or provider assumptions changed"):
        update.update(version="99.0.0")
    assert not update.registry_cache_dir().exists()


def test_gpu_prefill_applies_pue_once_and_does_not_scale_embodied():
    a = Estimator(pue=1).estimate("gpt-4o", input_tokens=10000, output_tokens=200)
    b = Estimator(pue=2).estimate("gpt-4o", input_tokens=10000, output_tokens=200)
    assert mean(b.prefill.energy_kwh) == pytest.approx(2 * mean(a.prefill.energy_kwh))
    assert mean(a.prefill.gwp_embodied) == mean(b.prefill.gwp_embodied) == 0


@pytest.mark.parametrize(
    "plan,counts",
    [
        (
            {"A": {"weight": 0.5, "tasks": 10}, "B": {"weight": 0.5, "tasks": 10}},
            {"A": {"input_tokens": 100, "output_tokens": 100}},
        ),
        ({"A": {"weight": 0.8, "tasks": 10}}, {"A": {"input_tokens": 100, "output_tokens": 100}}),
        ({"A": {"weight": 1, "tasks": 0}}, {"A": {"input_tokens": 100, "output_tokens": 100}}),
    ],
)
def test_incomplete_aa_suites_cannot_be_reweighted_silently(plan, counts):
    with pytest.raises(ValueError):
        aa.weighted_tokens({"evaluation_token_counts": counts}, {"index_version": "4.3", "benchmarks": plan}, "4.3")
