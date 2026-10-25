from pathlib import Path

import pandas as pd
import pytest

from green_ai import Estimator, Registry
from green_ai import batch as pp_batch
from green_ai import report as pp_report
from green_ai.ranges import hi, lo, mean

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "sample_usage.csv"


@pytest.fixture()
def est():
    return Estimator(registry=Registry.default())


@pytest.fixture()
def df():
    return pd.DataFrame(
        {
            "date": ["2026-01-03", "2026-01-20", "2026-02-02", "2026-02-11", "2026-02-12"],
            "model": ["gpt-4o", "azure/gpt-4o-mini", "claude-sonnet-4-5", "gemini-2.5-flash", "not-a-model"],
            "provider": ["openai", "azure", "anthropic", "google", "x"],
            "prompt_tokens": [12_000, 40_000, 90_000, 5_000, 1_000],
            "cached_tokens": [2_000, 0, 60_000, 0, 0],
            "completion_tokens": [400, 300, 900, 250, 100],
            "requests": [3, 2, 4, 1, 1],
        }
    )


def test_vectorised_rows_equal_scalar_estimates(est, df):
    res = pp_batch.run(df, estimator=est)
    for _, row in res.matched.iterrows():
        e = est.estimate(
            row["model"],
            provider=row["provider"],
            input_tokens=row["prompt_tokens"],
            cached_tokens=row["cached_tokens"],
            output_tokens=row["completion_tokens"],
            requests=row["requests"],
        )
        assert row["total_energy_kwh"] == pytest.approx(mean(e.total.energy_kwh), rel=1e-12)
        assert row["total_energy_kwh_min"] == pytest.approx(lo(e.total.energy_kwh), rel=1e-12)
        assert row["total_energy_kwh_max"] == pytest.approx(hi(e.total.energy_kwh), rel=1e-12)
        assert row["baseline_gwp_kg"] == pytest.approx(mean(e.baseline.gwp), rel=1e-12)
        assert row["prefill_water_l"] == pytest.approx(mean(e.prefill.water_l), rel=1e-12)


def test_column_detection(df):
    cmap = pp_batch.ColumnMap.detect(list(df.columns))
    assert cmap.input_tokens == "prompt_tokens"
    assert cmap.output_tokens == "completion_tokens"
    assert cmap.cached_tokens == "cached_tokens"
    assert cmap.requests == "requests"
    assert cmap.date == "date"


def test_column_override(est, df):
    d = df.rename(columns={"completion_tokens": "gen"})
    with pytest.raises(ValueError):
        pp_batch.run(d, estimator=est)
    res = pp_batch.run(d, estimator=est, output_tokens="gen")
    assert res.totals()["output_tokens"] > 0


def test_unmatched_are_reported_not_dropped(est, df):
    res = pp_batch.run(df, estimator=est)
    assert len(res.rows) == len(df)
    assert list(res.unmatched["model"]) == ["not-a-model"]
    assert res.totals()["rows_matched"] == 4


def test_aggregations_are_consistent(est, df):
    res = pp_batch.run(df, estimator=est)
    t = res.totals()
    assert res.by_month()["total_energy_kwh"].sum() == pytest.approx(t["total_energy_kwh"])
    assert res.by_model()["total_energy_kwh"].sum() == pytest.approx(t["total_energy_kwh"])
    assert res.by_provider()["total_gwp_kg"].sum() == pytest.approx(t["total_gwp_kg"])
    assert list(res.by_month()["month"]) == ["2026-01", "2026-02"]


def test_scenarios_match_reruns(est, df):
    res = pp_batch.run(df, estimator=est, prefill="central")
    sc = res.scenarios().set_index("scenario")
    for name in ("none", "low", "central"):
        rerun = pp_batch.run(df, estimator=est, prefill=name).totals()
        assert sc.loc[name, "total_energy_kwh"] == pytest.approx(rerun["total_energy_kwh"])
        assert sc.loc[name, "total_gwp_kg"] == pytest.approx(rerun["total_gwp_kg"])


def test_zone_override_changes_carbon_not_energy(df):
    a = pp_batch.run(df, estimator=Estimator(registry=Registry.default())).totals()
    b = pp_batch.run(df, estimator=Estimator(registry=Registry.default(), zone="FRA")).totals()
    assert a["total_energy_kwh"] == pytest.approx(b["total_energy_kwh"])
    assert b["total_gwp_kg"] < a["total_gwp_kg"]


def test_report_files(est, tmp_path):
    res = pp_batch.run(EXAMPLE, estimator=est)
    files = pp_report.write_report(res, tmp_path)
    for p in files.values():
        assert p.exists() and p.stat().st_size > 0
    xl = pd.ExcelFile(files["excel"])
    assert {"Summary", "Monthly", "By model", "Rows", "Assumptions", "Scenarios"} <= set(xl.sheet_names)
    md = files["summary"].read_text(encoding="utf-8")
    assert "Prefill scenarios" in md and "acme-internal-qwen" in md
