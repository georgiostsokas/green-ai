"""Measure agreement with EcoLogits over the full registry and print a Markdown table.

python scripts/validation_report.py > docs/VALIDATION_RESULTS.md
"""

from __future__ import annotations

import math
from importlib.metadata import version

from ecologits.model_repository import models as eco_models
from ecologits.tracers.utils import llm_impacts

from green_ai import Estimator
from green_ai.ranges import hi, lo

FIELDS = {
    "energy": ("energy", "energy_kwh"),
    "GWP total": ("gwp", "gwp"),
    "GWP embodied": (("embodied", "gwp"), "gwp_embodied"),
    "ADPe": ("adpe", "adpe"),
    "primary energy": ("pe", "pe"),
    "water": ("wcf", "water_l"),
}


def _ref(r, path):
    if isinstance(path, tuple):
        r = getattr(r, path[0])
        path = path[1]
    v = getattr(r, path).value
    return (v.min, v.max) if hasattr(v, "min") else (v, v)


def main() -> None:
    import logging

    logging.disable(logging.WARNING)
    est = Estimator(prefill="none")
    tokens = [1, 10, 100, 500, 1000, 5000, 20000]
    zones = [None, "WOR", "FRA", "USA", "DEU", "CHN"]
    worst = dict.fromkeys(FIELDS, 0.0)
    n = 0
    models = eco_models.list_models()
    for m in models:
        for t in tokens:
            for z in zones:
                r = llm_impacts(m.provider.value, m.name, t, math.inf, z)
                e = est.estimate(m.name, provider=m.provider.value, output_tokens=t, zone=z).baseline
                for label, (rp, ours) in FIELDS.items():
                    rl, rh = _ref(r, rp)
                    v = getattr(e, ours)
                    for a, b in ((rl, lo(v)), (rh, hi(v))):
                        worst[label] = max(worst[label], abs(a - b) / max(abs(a), 1e-300))
                        n += 1
    print(f"Reference: EcoLogits {version('ecologits')}, Green AI {version('green-ai')}\n")
    print(
        f"{len(models)} models x {len(tokens)} output token counts x {len(zones)} electricity zones, "
        f"lower and upper bounds: **{n:,} comparisons**.\n"
    )
    print("| Quantity | Worst relative difference |")
    print("|---|---|")
    for label, w in worst.items():
        print(f"| {label} | {w:.1e} |")


if __name__ == "__main__":
    main()
