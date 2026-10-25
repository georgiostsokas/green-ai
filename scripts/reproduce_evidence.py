"""Reproduce published-data reanalysis from numerical aggregates (no private logs)."""

import json
import statistics
from importlib.resources import files

d = json.loads(files("green_ai").joinpath("data/prefill_evidence.json").read_text(encoding="utf-8"))
a = statistics.median(
    (g["prefill_kwh"] / g["input_tokens"]) / (g["decode_kwh"] / g["output_tokens"]) for g in d["h100"]["groups"]
)
b = statistics.mean(
    (r["total_energy_J"] * r["prefill_percent"] / 100 / r["input_tokens"]) / r["decode_J_per_output_token"]
    for r in d["solovyeva"]["rows"]
)
assert abs(a - d["h100"]["median_alpha"]) < 1e-12
assert abs(b - d["solovyeva"]["mean_alpha"]) < 1e-12
print(f"H100 dataset: median group ratio {100 * a:.10f}%")
print(f"Published tables: mean row ratio {100 * b:.10f}%")
print("Reanalysis of public research, not original measurements or universal production coefficients.")
