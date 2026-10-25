import pytest

from green_ai import MatchStatus, ModelNotFoundError, Range, Registry
from green_ai.registry import canonical_provider, strip_date, strip_prefix


@pytest.fixture()
def reg():
    return Registry.default()


@pytest.mark.parametrize(
    "query,provider,expected",
    [
        ("gpt-4o", None, "gpt-4o"),
        ("openai/gpt-4o", None, "gpt-4o"),
        ("azure/gpt-4o-2024-08-06", None, "gpt-4o-2024-08-06"),
        ("GPT-4o", None, "gpt-4o"),
        ("gpt-4o-mini:latest", None, "gpt-4o-mini"),
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", None, "claude-sonnet-4-5-20250929"),
        ("us.anthropic.claude-opus-4-5-20251101-v1:0", None, "claude-opus-4-5-20251101"),
        ("claude-sonnet-4-5@20250929", None, "claude-sonnet-4-5-20250929"),
        ("models/gemini-2.5-flash", None, "gemini-2.5-flash"),
        ("gpt-4o", "azure", "gpt-4o"),
    ],
)
def test_matching(reg, query, provider, expected):
    m = reg.match(query, provider)
    assert m.ok, m
    assert m.model.name == expected


def test_alias_status(reg):
    m = reg.match("claude-sonnet-4-5", "anthropic")
    assert m.status is MatchStatus.ALIAS
    assert m.model.name == "claude-sonnet-4-5-20250929"


def test_unmatched_has_suggestions(reg):
    m = reg.match("gpt4o")
    assert m.status is MatchStatus.UNMATCHED
    assert "gpt-4o" in m.suggestions
    with pytest.raises(ModelNotFoundError, match="Did you mean"):
        reg.get("gpt4o")


def test_provider_filter_is_respected(reg):
    assert not reg.match("gpt-4o", "anthropic").ok


def test_register_custom_moe(reg):
    spec = reg.register("my-moe", total_params=284, active_params=13, provider="onprem")
    assert spec.architecture == "moe"
    assert reg.get("my-moe").active_params_b == 13
    assert reg.provider("onprem").location == "WOR"


def test_register_ranges(reg):
    spec = reg.register("mystery", total_params=Range(100, 300), provider="custom")
    assert isinstance(spec.active_params_b, Range)


def test_duplicate_requires_replace(reg):
    reg.register("dup", total_params=7)
    with pytest.raises(ValueError):
        reg.register("dup", total_params=8)
    reg.register("dup", total_params=8, replace=True)
    assert reg.get("dup").total_params_b == 8


def test_load_custom_yaml(reg, tmp_path):
    f = tmp_path / "m.yaml"
    f.write_text(
        "providers:\n  lab:\n    location: fra\n    pue: {min: 1.2, max: 1.5}\n    wue: 0\n"
        "models:\n  - name: lab-model\n    provider: lab\n    total_params: 32\n"
        "  - name: lab-moe\n    provider: lab\n    total_params: {min: 200, max: 300}\n    active_params: 20\n",
        encoding="utf-8",
    )
    added = reg.load_custom(f)
    assert len(added) == 2
    assert reg.provider("lab").location == "FRA"
    assert reg.provider("lab").pue == Range(1.2, 1.5)
    reg.load_custom(f)  # loading twice is idempotent


def test_zone_lookup_and_override(reg):
    assert reg.zone("fra").gwp < reg.zone("WOR").gwp
    mix = reg.add_zone("XYZ", gwp=0.05)
    assert mix.wue == reg.zone("WOR").wue
    with pytest.raises(LookupError):
        reg.zone("NOPE")


def test_helpers():
    assert canonical_provider("Azure OpenAI") == "openai"
    assert canonical_provider("bedrock") == "anthropic"
    assert strip_prefix("azure/openai/gpt-4o") == "gpt-4o"
    assert strip_date("gpt-4o-2024-08-06") == "gpt-4o"
    assert strip_date("claude-sonnet-4-5-20250929") == "claude-sonnet-4-5"
    assert strip_date("claude-x@20250929") == "claude-x"


def test_registry_is_complete(reg):
    assert len(reg.models) > 300
    assert {"openai", "anthropic", "mistralai", "google_genai", "cohere", "huggingface_hub"} <= set(reg.providers)
    assert len(reg.zones) > 200
