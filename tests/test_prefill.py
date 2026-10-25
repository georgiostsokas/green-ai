import pytest

from green_ai import CENTRAL, LOW, NONE, Estimator, PrefillModel
from green_ai.prefill import resolve, underestimation
from green_ai.ranges import mean


@pytest.fixture(scope="module")
def est():
    return Estimator()


def test_no_input_means_no_correction(est):
    e = est.estimate("gpt-4o", provider="openai", output_tokens=300)
    assert mean(e.prefill.energy_kwh) == 0
    assert e.correction_factor == 1


def test_none_preset_reproduces_baseline(est):
    e = est.estimate("gpt-4o", provider="openai", input_tokens=50_000, output_tokens=300, prefill="none")
    assert mean(e.total.energy_kwh) == mean(e.baseline.energy_kwh)


def test_prefill_is_linear_in_input_tokens(est):
    a = est.estimate("gpt-4o", provider="openai", input_tokens=10_000, output_tokens=200)
    b = est.estimate("gpt-4o", provider="openai", input_tokens=20_000, output_tokens=200)
    assert mean(b.prefill.energy_kwh) == pytest.approx(2 * mean(a.prefill.energy_kwh))
    assert mean(b.baseline.energy_kwh) == pytest.approx(mean(a.baseline.energy_kwh))


def test_prefill_changes_operational_gpu_impacts_only(est):
    """A phase GPU-energy ratio cannot scale baseline host time or manufacturing."""
    n_in = 50_000
    e = est.estimate("gpt-4o", provider="openai", input_tokens=n_in, output_tokens=100)
    assert mean(e.prefill.gwp_embodied) == 0
    assert mean(e.prefill.pe_embodied) == 0
    assert mean(e.prefill.adpe_embodied) == 0
    assert mean(e.prefill.gwp_usage) == pytest.approx(mean(e.prefill.energy_kwh) * e.mix.gwp)
    larger_host = Estimator(hardware=est.hardware.with_overrides(server_power_kw=12))
    x = larger_host.estimate("gpt-4o", input_tokens=n_in, output_tokens=100)
    assert mean(x.baseline.energy_kwh) > mean(e.baseline.energy_kwh)
    assert mean(x.prefill.energy_kwh) == pytest.approx(mean(e.prefill.energy_kwh))


def test_cached_tokens_are_discounted(est):
    fresh = est.estimate("gpt-4o", provider="openai", input_tokens=10_000, output_tokens=100)
    cached = est.estimate("gpt-4o", provider="openai", input_tokens=10_000, cached_tokens=10_000, output_tokens=100)
    ratio = mean(cached.prefill.energy_kwh) / mean(fresh.prefill.energy_kwh)
    assert ratio == pytest.approx(CENTRAL.cached_factor)


def test_scenarios_are_ordered(est):
    s = est.scenarios("claude-sonnet-4-5", input_tokens=30_000, output_tokens=300)
    energies = [mean(s[k].total.energy_kwh) for k in ("none", "low", "central")]
    assert energies == sorted(energies)
    assert energies[0] < energies[2]


def test_ranges_survive_correction(est):
    e = est.estimate("claude-sonnet-4-5", input_tokens=30_000, output_tokens=300)
    t = e.total.energy_kwh
    assert t.lo < t.mean < t.hi


def test_equivalent_tokens():
    m = PrefillModel("x", alpha=0.01, cached_factor=0.5)
    assert m.equivalent_tokens(1000) == pytest.approx(10)
    assert m.equivalent_tokens(1000, 1000) == pytest.approx(5)
    assert m.equivalent_tokens(1000, 5000) == pytest.approx(5)  # cached capped at input
    assert NONE.equivalent_tokens(1_000_000) == 0
    with pytest.raises(ValueError):
        m.equivalent_tokens(-1)


def test_resolve():
    assert resolve("central") is CENTRAL
    assert resolve("LOW") is LOW
    assert resolve(None) is NONE
    assert resolve(0.01).alpha == 0.01
    assert resolve("0.02").alpha == 0.02
    with pytest.raises(ValueError):
        resolve("bogus")
    with pytest.raises(ValueError):
        PrefillModel("bad", alpha=-1)


def test_underestimation_formula(est):
    assert underestimation(100, 0.008) == pytest.approx(0.8 / 1.8)
    # Consistency with the full model for a model without time to first token
    est.registry.register("toy-dense-70b", total_params=70, provider="toyprov", replace=True)
    e = est.estimate("toy-dense-70b", provider="toyprov", input_tokens=50_000, output_tokens=500)
    assert 0 < e.prefill_share < underestimation(100, 0.008)  # baseline also includes host energy


def test_input_validation(est):
    with pytest.raises(ValueError):
        est.estimate("gpt-4o", output_tokens=10, input_tokens=5, cached_tokens=10)
    with pytest.raises(ValueError):
        est.estimate("gpt-4o", output_tokens=-1)
