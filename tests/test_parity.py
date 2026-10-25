"""Numerical parity with the EcoLogits reference implementation.

With the prefill correction switched off, Green AI must return the same
numbers as EcoLogits for every registered model, every zone and any token count.
"""

import math

import pytest

ecologits = pytest.importorskip("ecologits")
from ecologits.model_repository import models as eco_models  # noqa: E402
from ecologits.tracers.utils import llm_impacts  # noqa: E402

from green_ai import Estimator  # noqa: E402
from green_ai.engine import compute_impacts  # noqa: E402
from green_ai.ranges import hi, lo  # noqa: E402

pytestmark = pytest.mark.parity
REL = 1e-9

ALL_MODELS = [(m.provider.value, m.name) for m in eco_models.list_models()]


def _bounds(v):
    v = v.value
    return (v.min, v.max) if hasattr(v, "min") else (v, v)


def _assert_close(ref, ours, label):
    rl, rh = _bounds(ref)
    assert lo(ours) == pytest.approx(rl, rel=REL), f"{label} lower bound"
    assert hi(ours) == pytest.approx(rh, rel=REL), f"{label} upper bound"


@pytest.fixture(scope="module")
def est():
    return Estimator(prefill="none")


@pytest.mark.parametrize("provider,name", ALL_MODELS)
@pytest.mark.parametrize("out_tokens", [1, 250, 4000])
def test_every_model_matches_ecologits(est, provider, name, out_tokens):
    ref = llm_impacts(provider, name, out_tokens, math.inf)
    e = est.estimate(name, provider=provider, output_tokens=out_tokens)
    b = e.baseline
    _assert_close(ref.energy, b.energy_kwh, "energy")
    _assert_close(ref.gwp, b.gwp, "gwp")
    _assert_close(ref.adpe, b.adpe, "adpe")
    _assert_close(ref.pe, b.pe, "pe")
    _assert_close(ref.wcf, b.water_l, "water")
    _assert_close(ref.usage.gwp, b.gwp_usage, "gwp usage")
    _assert_close(ref.embodied.gwp, b.gwp_embodied, "gwp embodied")


@pytest.mark.parametrize("zone", ["WOR", "FRA", "USA", "DEU", "SWE", "GRC", "CHN", "IND", "BRA"])
@pytest.mark.parametrize(
    "provider,name", [("openai", "gpt-4o"), ("anthropic", "claude-sonnet-4-5"), ("mistralai", "mistral-large-latest")]
)
def test_zones_match_ecologits(est, zone, provider, name):
    ref = llm_impacts(provider, name, 800, math.inf, zone)
    e = est.estimate(name, provider=provider, output_tokens=800, zone=zone)
    _assert_close(ref.gwp, e.baseline.gwp, "gwp")
    _assert_close(ref.wcf, e.baseline.water_l, "water")


@pytest.mark.parametrize("latency", [0.2, 1.0, 5.0])
def test_measured_latency_cap_matches_ecologits(est, latency):
    ref = llm_impacts("openai", "gpt-4o", 1500, latency)
    e = est.estimate("gpt-4o", provider="openai", output_tokens=1500, request_latency_s=latency)
    _assert_close(ref.energy, e.baseline.energy_kwh, "energy")
    _assert_close(ref.gwp, e.baseline.gwp, "gwp")


def test_engine_matches_ecologits_without_deployment_data():
    """Regression latency path (no tokens per second known)."""
    from ecologits.impacts.llm import compute_llm_impacts

    from green_ai.engine import ElectricityMix

    mix = ElectricityMix("X", gwp=0.3, adpe=1e-8, pe=10.0, wue=2.0)
    ref = compute_llm_impacts(
        model_active_parameter_count=70,
        model_total_parameter_count=70,
        output_token_count=512,
        if_electricity_mix_adpe=mix.adpe,
        if_electricity_mix_pe=mix.pe,
        if_electricity_mix_gwp=mix.gwp,
        if_electricity_mix_wue=mix.wue,
        datacenter_pue=1.2,
        datacenter_wue=0.5,
    )
    ours = compute_impacts(active_params_b=70, total_params_b=70, output_tokens=512, mix=mix, pue=1.2, wue=0.5)
    _assert_close(ref.energy, ours.energy_kwh, "energy")
    _assert_close(ref.gwp, ours.gwp, "gwp")
    _assert_close(ref.wcf, ours.water_l, "water")
