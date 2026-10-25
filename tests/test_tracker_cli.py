import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from green_ai import Tracker
from green_ai.cli import app
from green_ai.ranges import Range, mean
from green_ai.tracker import parse_usage

runner = CliRunner()


def test_parse_openai_chat_object():
    resp = SimpleNamespace(
        model="gpt-4o-2024-08-06",
        usage=SimpleNamespace(
            prompt_tokens=1200, completion_tokens=300, prompt_tokens_details=SimpleNamespace(cached_tokens=1024)
        ),
    )
    u = parse_usage(resp)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens, u.provider) == (1200, 300, 1024, "openai")


def test_parse_openai_responses_dict():
    u = parse_usage(
        {
            "model": "gpt-5",
            "usage": {"input_tokens": 900, "output_tokens": 2000, "input_tokens_details": {"cached_tokens": 100}},
        }
    )
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (900, 2000, 100)


def test_parse_anthropic_adds_cache_back():
    resp = {
        "model": "claude-sonnet-4-5",
        "usage": {
            "input_tokens": 50,
            "output_tokens": 400,
            "cache_read_input_tokens": 9000,
            "cache_creation_input_tokens": 1000,
        },
    }
    u = parse_usage(resp)
    assert (u.input_tokens, u.cached_tokens, u.provider) == (10_050, 9000, "anthropic")


def test_parse_gemini():
    resp = {
        "model_version": "gemini-2.5-flash",
        "usage_metadata": {
            "prompt_token_count": 3000,
            "candidates_token_count": 200,
            "thoughts_token_count": 800,
            "cached_content_token_count": 1000,
        },
    }
    u = parse_usage(resp)
    assert (u.input_tokens, u.output_tokens, u.cached_tokens, u.provider) == (3000, 1000, 1000, "google_genai")


def test_tracker_accumulates():
    t = Tracker(zone="FRA")
    t.add({"model": "gpt-4o", "usage": {"prompt_tokens": 5000, "completion_tokens": 200}})
    t.add({"model": "gpt-4o", "usage": {"prompt_tokens": 5000, "completion_tokens": 200}})
    assert len(t.records) == 2
    assert mean(t.total.energy_kwh) == pytest.approx(2 * mean(t.records[0].total.energy_kwh))
    assert "2 calls" in t.summary()


def test_missing_model_name():
    with pytest.raises(ValueError):
        parse_usage({"usage": {"prompt_tokens": 1}})


def test_cli_estimate_json():
    r = runner.invoke(app, ["estimate", "gpt-4o", "--in", "10000", "--out", "300", "--json"])
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["model"] == "gpt-4o" and d["prefill_alpha"] == 0.008
    assert d["total_energy_kwh"] > d["baseline_energy_kwh"]


def test_cli_estimate_table_and_overrides():
    r = runner.invoke(
        app,
        [
            "estimate",
            "claude-sonnet-4-5",
            "--in",
            "40000",
            "--cached",
            "20000",
            "--out",
            "500",
            "--zone",
            "FRA",
            "--pue",
            "1.1:1.3",
            "--alpha",
            "0.01",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "Prefill scenarios" in r.output


def test_cli_unknown_model_exit_code():
    r = runner.invoke(app, ["estimate", "gpt4o", "--out", "10"])
    assert r.exit_code == 1
    assert "Did you mean" in r.output


@pytest.mark.parametrize(
    "args",
    [
        ["models", "gpt-4o"],
        ["zones", "FR"],
        ["match", "azure/gpt-4o"],
        ["rank", "-p", "mistral", "--top", "3"],
        ["sensitivity", "gpt-4o"],
        ["version"],
    ],
)
def test_cli_commands(args):
    r = runner.invoke(app, args)
    assert r.exit_code == 0, r.output


def test_cli_batch(tmp_path):
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples"
    r = runner.invoke(
        app,
        [
            "batch",
            str(example / "sample_usage.csv"),
            "-o",
            str(tmp_path),
            "--custom",
            str(example / "custom_models.yaml"),
            "--zone",
            "FRA",
        ],
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "footprint.xlsx").exists()
    assert (tmp_path / "summary.md").exists()


def test_range_arithmetic():
    a = Range(1, 3)
    assert (a + 1) == Range(2, 4)
    assert (a * Range(2, 3)) == Range(2, 9)
    assert a.mean == 2
    with pytest.raises(ValueError):
        Range(3, 1)
    assert f"{Range(1, 1):.1f}" == "1.0"
