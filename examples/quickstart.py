"""Green AI in five minutes. Run: python examples/quickstart.py"""

from pathlib import Path

from green_ai import Estimator, Tracker, estimate, register_model
from green_ai.ranges import mean

HERE = Path(__file__).parent

# 1. One request ---------------------------------------------------------------
# A retrieval augmented answer: 24k prompt tokens (8k from the provider cache), 600 out.
e = estimate("gpt-4o", provider="openai", input_tokens=24_000, cached_tokens=8_000, output_tokens=600)
print(e)
print(f"reference estimate misses {e.prefill_share:.0%} of the energy\n")

# 2. Same request, different assumptions ---------------------------------------
est = Estimator(zone="FRA", pue=1.3)  # French grid, your own PUE
for name, s in est.scenarios("gpt-4o", input_tokens=24_000, output_tokens=600).items():
    print(f"{name:>8}: {mean(s.total.energy_wh):.3f} Wh  {mean(s.total.gwp_g):.4f} gCO2eq")
print()

# 3. A model missing from the registry -----------------------------------------
register_model("my-finetuned-llama", total_params=70, provider="onprem")
print(estimate("my-finetuned-llama", input_tokens=5_000, output_tokens=300, zone="DEU"), "\n")

# 4. Track live calls from the usage objects providers return ------------------
tracker = Tracker()
fake_openai_response = {
    "model": "gpt-4o-mini",
    "usage": {"prompt_tokens": 3200, "completion_tokens": 250, "prompt_tokens_details": {"cached_tokens": 2048}},
}
fake_anthropic_response = {
    "model": "claude-sonnet-4-5",
    "usage": {"input_tokens": 120, "output_tokens": 900, "cache_read_input_tokens": 40_000},
}
tracker.add(fake_openai_response)
tracker.add(fake_anthropic_response)
print(tracker.summary(), "\n")

# 5. A whole usage log (needs: pip install "green-ai[report]") --------------
try:
    from green_ai import batch, report

    est = Estimator()
    est.registry.load_custom(HERE / "custom_models.yaml")
    result = batch.run(HERE / "sample_usage.csv", estimator=est)
    print(result.by_month()[["month", "baseline_energy_kwh", "total_energy_kwh", "prefill_share"]])
    files = report.write_report(result, HERE / "report")
    print("report written to", files["summary"].parent)
except ImportError:
    print('Install the report extra for batch processing: pip install "green-ai[report]"')
