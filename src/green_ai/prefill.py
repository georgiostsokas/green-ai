"""Experimental input correction based on GPU phase-energy ratios.

    equivalents = alpha * (fresh_input + cached_factor * cached_input)
    extra_GPU_IT_kWh = equivalents * GPU_decode_kWh_per_output_token

PUE is applied once. Operational GHG, water, PE and ADPe follow the same mix.
Server electricity, request latency and embodied allocation stay in the baseline.
Alpha 0.005 and 0.008 are sensitivity scenarios informed by public measurements,
not universal coefficients or statistical bounds. Cache factor 0.1 is a pricing-
informed scenario, not a measured energy discount. See bundled evidence and
scripts/reproduce_evidence.py for the numerical derivation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class PrefillModel:
    """How many decode token equivalents one prompt token costs."""

    name: str
    alpha: float
    cached_factor: float = 0.1
    description: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha < 0 or not 0 <= self.cached_factor <= 1:
            raise ValueError("alpha must be >= 0 and cached_factor must be within [0, 1]")

    def equivalent_tokens(self, input_tokens: float, cached_tokens: float = 0.0) -> float:
        """Decode token equivalents of a prompt.

        ``input_tokens`` is the full prompt length *including* cached tokens
        (OpenAI convention). ``cached_tokens`` is the part read from cache.
        """
        if any(not math.isfinite(v) or v < 0 for v in (input_tokens, cached_tokens)):
            raise ValueError("token counts must be non-negative")
        cached = min(cached_tokens, input_tokens)
        fresh = input_tokens - cached
        return self.alpha * (fresh + self.cached_factor * cached)

    @classmethod
    def custom(cls, alpha: float, cached_factor: float = 0.1) -> PrefillModel:
        return cls(f"alpha={alpha:g}", alpha, cached_factor, "User defined coefficient")


NONE = PrefillModel("none", 0.0, 0.0, "No correction, reference methodology only")
LOW = PrefillModel("low", 0.005, 0.1, "Low GPU-energy sensitivity scenario")
CENTRAL = PrefillModel("central", 0.008, 0.1, "Central GPU-energy sensitivity scenario")
PRICING = PrefillModel(
    "pricing",
    0.30,
    0.1,
    "Legacy pricing sensitivity. Neither a physical estimate nor an upper bound.",
)

PRESETS: dict[str, PrefillModel] = {m.name: m for m in (NONE, LOW, CENTRAL, PRICING)}

PrefillLike = Union[str, float, PrefillModel, None]


def resolve(prefill: PrefillLike) -> PrefillModel:
    """Accept a preset name, a numeric alpha, a :class:`PrefillModel` or ``None``."""
    if prefill is None:
        return NONE
    if isinstance(prefill, PrefillModel):
        return prefill
    if isinstance(prefill, (int, float)):
        return PrefillModel.custom(float(prefill))
    key = str(prefill).strip().lower()
    if key in PRESETS:
        return PRESETS[key]
    try:
        return PrefillModel.custom(float(key))
    except ValueError:
        raise ValueError(f"Unknown prefill model '{prefill}'. Use one of {sorted(PRESETS)} or a number.") from None


def underestimation(io_ratio: float, alpha: float) -> float:
    """Missing share in a GPU-only linear toy model; not the full server/TTFT model.

    For a prompt ``r`` times longer than the answer the corrected energy is
    ``(1 + alpha * r)`` times the baseline, so the baseline misses
    ``alpha * r / (1 + alpha * r)`` of that toy-model total.
    """
    x = alpha * io_ratio
    return x / (1 + x)
