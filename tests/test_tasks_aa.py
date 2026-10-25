import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from green_ai import Estimator, Range, Registry
from green_ai import aa as pp_aa
from green_ai import update as pp_update
from green_ai.cli import app
from green_ai.ranges import hi, lo, mean
from green_ai.tasks import TaskProfile, load_profiles, rank_tasks, save_ranking, task_footprint

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
runner = CliRunner()


def _item(slug, name, creator, score, total_cost, per_task, p_in=1.0, p_out=4.0, counts=None, **extra):
    d = {
        "id": slug,
        "slug": slug,
        "name": name,
        "model_creator": {"id": creator.lower(), "name": creator},
        "evaluations": {"artificial_analysis_intelligence_index": score},
        "artificial_analysis_intelligence_index_cost": {
            "total_cost": total_cost,
            "cost_per_task": {"total_cost": per_task},
        },
        "pricing": {"price_1m_input_tokens": p_in, "price_1m_output_tokens": p_out},
    }
    if counts:
        d["artificial_analysis_intelligence_index_token_counts"] = counts
    d.update(extra)
    return d


PRO = {
    "tier": "pro",
    "intelligence_index_version": "4.1",
    "data": [
        _item(
            "gpt-5",
            "GPT-5 (high)",
            "OpenAI",
            68.0,
            900.0,
            0.9,
            counts={"input_tokens": 30e6, "answer_tokens": 2e6, "reasoning_tokens": 80e6, "output_tokens": 82e6},
        ),
        _item(
            "gpt-4o",
            "GPT-4o",
            "OpenAI",
            30.0,
            200.0,
            0.2,
            counts={"input_tokens": 25e6, "answer_tokens": 4e6, "reasoning_tokens": 0, "output_tokens": 4e6},
        ),
        _item(
            "claude-4-5-sonnet-thinking",
            "Claude 4.5 Sonnet (Reasoning)",
            "Anthropic",
            63.0,
            1000.0,
            1.0,
            counts={"input_tokens": 28e6, "answer_tokens": 3e6, "reasoning_tokens": 30e6},
        ),
        _item(
            "some-open-model",
            "Some Open Model",
            "Acme AI",
            40.0,
            50.0,
            0.05,
            counts={"input_tokens": 26e6, "output_tokens": 20e6},
            parameters={"total": 120, "active": 12},
        ),
        _item("mystery", "Mystery Model", "Nobody", 20.0, 10.0, 0.01, counts={"input_tokens": 1, "output_tokens": 1}),
        {"slug": "no-index", "name": "Not evaluated", "model_creator": {"name": "OpenAI"}, "evaluations": {}},
    ],
}

FREE = {
    "tier": "free",
    "intelligence_index_version": "4.1",
    "data": [
        _item("gpt-4o", "GPT-4o", "OpenAI", 30.0, 200.0, 0.2, p_in=2.5, p_out=10.0),
        _item("gemini-2-5-flash", "Gemini 2.5 Flash (Reasoning)", "Google", 50.0, 40.0, 0.04, p_in=0.3, p_out=2.5),
    ],
}


@pytest.fixture()
def reg():
    return Registry.default()


def test_pro_totals_are_never_misrepresented_as_exact_per_task(reg):
    res = pp_aa.to_profiles(PRO, registry=reg)
    by = {p.label: p for p in res.profiles}
    gpt5 = by["GPT-5 (high)"]
    assert gpt5.model == "gpt-5" and gpt5.is_estimated
    assert gpt5.token_basis == "cost-inferred"
    sonnet = by["Claude 4.5 Sonnet (Reasoning)"]
    assert sonnet.model.startswith("claude-sonnet-4-5")
    assert sonnet.is_estimated
    assert by["Some Open Model"].model == "aa/some-open-model"  # registered from disclosed parameters
    assert [n for n, _ in res.unmatched] == ["Mystery Model"]
    assert "Artificial Analysis" in res.attribution and "pro" in res.attribution


def test_free_profiles_inferred_from_cost(reg):
    res = pp_aa.to_profiles(FREE, registry=reg, output_share=Range(0.5, 0.9))
    p = next(x for x in res.profiles if x.label == "GPT-4o")
    assert p.is_estimated
    # cost 0.2 $/task, output share 50% -> 0.1 $ / 10 $ per 1M = 10,000 output tokens
    assert lo(p.output_tokens) == pytest.approx(10_000)
    assert hi(p.output_tokens) == pytest.approx(18_000)
    # every scenario reproduces the cost exactly
    for i, o in p.scenarios:
        assert i * 2.5e-6 + o * 10e-6 == pytest.approx(0.2)
    f = task_footprint(p, estimator=Estimator(registry=reg))
    assert lo(f.energy_wh) < hi(f.energy_wh)


def test_name_candidates():
    c = pp_aa.name_candidates({"slug": "claude-4-5-sonnet-thinking", "name": "Claude 4.5 Sonnet (Reasoning)"})
    assert "claude-sonnet-4-5" in c
    c = pp_aa.name_candidates({"slug": "gemini-2-5-pro", "name": "Gemini 2.5 Pro"})
    assert "gemini-2.5-pro" in c
    assert "gpt-5" in pp_aa.name_candidates({"slug": "gpt-5-low", "name": "GPT-5 (low)"})


def test_mapping_override(reg):
    res = pp_aa.to_profiles(PRO, registry=reg, overrides={"mystery": "openai/gpt-4o-mini"})
    assert any(p.label == "Mystery Model" and p.model == "gpt-4o-mini" for p in res.profiles)


def test_task_footprint_equals_estimate(reg):
    est = Estimator(registry=reg)
    p = TaskProfile.exact("gpt-4o", 20_000, 800, provider="openai")
    f = task_footprint(p, estimator=est, zone="FRA")
    e = est.estimate("gpt-4o", provider="openai", input_tokens=20_000, output_tokens=800, zone="FRA")
    assert mean(f.energy_wh) == pytest.approx(mean(e.total.energy_wh))
    assert mean(f.water_ml) == pytest.approx(mean(e.total.water_l) * 1000)


def test_ranking_and_shift(reg):
    est = Estimator(registry=reg)
    profiles = [
        TaskProfile.exact("gpt-4o-mini", 10_000, 500, provider="openai", score=40),
        TaskProfile.exact("gpt-4o", 10_000, 500, provider="openai", score=55),
        # same architecture as gpt-4o but ten times the output tokens per task
        TaskProfile.exact("gpt-4o", 10_000, 5_000, provider="openai", label="gpt-4o verbose", score=60),
        TaskProfile.exact("does-not-exist", 1, 1),
    ]
    ranking, skipped = rank_tasks(profiles, estimator=est)
    assert [r.footprint.profile.name for r in ranking] == ["gpt-4o-mini", "gpt-4o", "gpt-4o verbose"]
    assert len(skipped) == 1
    assert ranking[0].grade == "A" and ranking[-1].grade == "E"
    eff, _ = rank_tasks(profiles[:3], estimator=est, by="efficiency")
    assert eff[0].footprint.efficiency >= eff[-1].footprint.efficiency
    with pytest.raises(ValueError):
        rank_tasks(profiles[:1], estimator=est, by="bogus")


def test_csv_roundtrip(reg, tmp_path):
    profiles = load_profiles(EXAMPLES / "task_profiles.csv")
    assert len(profiles) == 6 and profiles[0].score == 38
    ranking, _ = rank_tasks(profiles, estimator=Estimator(registry=reg))
    out = save_ranking(ranking, tmp_path / "r.csv")
    assert "energy_wh_per_task" in out.read_text(encoding="utf-8").splitlines()[0]


def test_example_profiles_show_a_rank_shift(reg):
    """The README figure depends on the example: per task order must differ from per token order."""
    ranking, _ = rank_tasks(load_profiles(EXAMPLES / "task_profiles.csv"), estimator=Estimator(registry=reg))
    shifts = [r.shift for r in ranking]
    assert any(s > 0 for s in shifts) and any(s < 0 for s in shifts)


def test_aa_fetch_uses_cache_and_falls_back_to_free(monkeypatch, tmp_path):
    monkeypatch.setenv("GREEN_AI_CACHE_DIR", str(tmp_path))
    calls = []

    def fake_request(path, key, page):
        calls.append(path)
        if path.endswith("/language/models"):
            raise PermissionError("tier")
        return {
            "tier": "free",
            "intelligence_index_version": "4.1",
            "pagination": {"has_more": False},
            "data": FREE["data"],
        }

    monkeypatch.setattr(pp_aa, "_request", fake_request)
    data = pp_aa.fetch("k")
    assert data["tier"] == "free" and len(data["data"]) == 2
    assert calls == ["/language/models", "/language/models/free"]
    pp_aa.fetch("k")  # served from cache
    assert len(calls) == 2
    with pytest.raises(pp_aa.AAError):
        monkeypatch.delenv("AA_API_KEY", raising=False)
        pp_aa.fetch(None, refresh=True)


def test_cli_tasks_with_mocked_aa(monkeypatch):
    monkeypatch.setattr(pp_aa, "fetch", lambda *a, **k: PRO)
    r = runner.invoke(app, ["tasks", "--aa", "--top", "5"])
    assert r.exit_code == 0, r.output
    assert "Artificial Analysis" in r.output
    r = runner.invoke(app, ["tasks"])
    assert r.exit_code == 1


def test_update_prefers_newer_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(pp_update, "_check_reference_compatibility", lambda version: None)
    monkeypatch.setenv("GREEN_AI_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("GREEN_AI_USE_BUNDLED", raising=False)
    bundled = pp_update.bundled_dir()
    models = json.loads((bundled / "models.json").read_text(encoding="utf-8"))
    models["models"].append(
        {
            "type": "model",
            "provider": "newprov",
            "name": "brand-new-model",
            "architecture": {"type": "dense", "parameters": 70},
            "warnings": None,
            "sources": [],
            "deployment": {"tps": None, "ttft": None},
        }
    )
    files = {
        "models.json": json.dumps(models).encode(),
        "electricity_mixes.json": (bundled / "electricity_mixes.json").read_bytes(),
    }

    def fake_get(url):
        if "pypi" in url:
            return json.dumps({"info": {"version": "99.0.0"}}).encode()
        return files[url.rsplit("/", 1)[-1]]

    monkeypatch.setattr(pp_update, "_get", fake_get)
    res = pp_update.update()
    assert res.changed and res.current == "99.0.0"
    assert "newprov/brand-new-model" in res.added
    assert pp_update.active_version() == ("99.0.0", "cache")
    reg = Registry.default()
    assert reg.data_source == "cache"
    assert reg.get("brand-new-model").total_params_b == 70
    assert reg.provider("newprov").location == "WOR"  # neutral default for unknown providers
    assert not pp_update.update().changed  # already current
    monkeypatch.setenv("GREEN_AI_USE_BUNDLED", "1")
    assert Registry.default().data_source == "bundled"
    assert pp_update.clear_cache()


def test_update_rejects_broken_data(monkeypatch, tmp_path):
    monkeypatch.setattr(pp_update, "_check_reference_compatibility", lambda version: None)
    monkeypatch.setenv("GREEN_AI_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("GREEN_AI_USE_BUNDLED", raising=False)

    def fake_get(url):
        if "pypi" in url:
            return json.dumps({"info": {"version": "99.0.0"}}).encode()
        if url.endswith("models.json"):
            return json.dumps({"models": []}).encode()
        return json.dumps({"electricity_mixes": []}).encode()

    monkeypatch.setattr(pp_update, "_get", fake_get)
    with pytest.raises(ValueError):
        pp_update.update()
    assert pp_update.active_version()[1] == "bundled"
