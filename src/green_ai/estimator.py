"""High level estimation API."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from . import prefill as _prefill
from .engine import DEFAULT_HARDWARE, ZERO, ElectricityMix, Hardware, Impacts, compute_impacts, compute_prefill_unit
from .prefill import PrefillLike, PrefillModel
from .ranges import Value, hi, lo, mean
from .registry import Match, MatchStatus, ModelNotFoundError, ModelSpec, ProviderSpec, Registry, default_registry

WARNING_TEXT = {
    "model-arch-not-released": "Architecture not disclosed by the provider, parameter count is an estimate range.",
    "model-arch-multimodal": "Multimodal model, text only methodology may be less precise.",
    "electricity-mix-adpe-world": "ADPe factor for this zone falls back to the world average.",
    "electricity-mix-pe-world": "Primary energy factor for this zone falls back to the world average.",
    "electricity-mix-wue-world": "Water factor for this zone falls back to the world average.",
}


@dataclass(frozen=True)
class Estimate:
    """Result of an estimation. ``baseline`` follows the reference methodology,
    ``prefill`` is the input token correction, ``total`` is their sum."""

    model: ModelSpec
    match: Match
    provider: ProviderSpec
    mix: ElectricityMix
    pue: Value
    wue: Value
    input_tokens: float
    cached_tokens: float
    output_tokens: float
    requests: float
    prefill_model: PrefillModel
    baseline: Impacts
    prefill: Impacts
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total(self) -> Impacts:
        return self.baseline + self.prefill

    @property
    def prefill_share(self) -> float:
        """Share of total energy attributable to prompt processing (mean values)."""
        t = mean(self.total.energy_kwh)
        return mean(self.prefill.energy_kwh) / t if t else 0.0

    @property
    def correction_factor(self) -> float:
        """Corrected over baseline energy (mean values). 1.0 means no correction."""
        b = mean(self.baseline.energy_kwh)
        return mean(self.total.energy_kwh) / b if b else 1.0

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "model": self.model.name,
            "provider": self.model.provider,
            "match_status": self.match.status.value,
            "zone": self.mix.zone,
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "requests": self.requests,
            "prefill_model": self.prefill_model.name,
            "prefill_alpha": self.prefill_model.alpha,
            "cached_factor": self.prefill_model.cached_factor,
            "pue_min": lo(self.pue),
            "pue_max": hi(self.pue),
            "wue_min": lo(self.wue),
            "wue_max": hi(self.wue),
            "grid_kgco2eq_per_kwh": self.mix.gwp,
            "prefill_scope": "GPU operational impacts only; server and embodied allocation unchanged",
        }
        row.update(self.baseline.flat("baseline_"))
        row.update(self.prefill.flat("prefill_"))
        row.update(self.total.flat("total_"))
        row["prefill_share"] = self.prefill_share
        row["warnings"] = "; ".join(self.warnings)
        return row

    def __str__(self) -> str:
        t, b = self.total, self.baseline
        return (
            f"{self.model.provider}/{self.model.name} | in={self.input_tokens:,.0f} "
            f"(cached {self.cached_tokens:,.0f}) out={self.output_tokens:,.0f} | zone={self.mix.zone}\n"
            f"  energy  {t.energy_wh:.4g} Wh   (baseline {b.energy_wh:.4g} Wh, "
            f"prefill share {self.prefill_share:.0%})\n"
            f"  GHG     {t.gwp_g:.4g} gCO2eq\n"
            f"  water   {t.water_l:.4g} L"
        )


@dataclass
class Estimator:
    """Reusable estimator holding the registry and deployment assumptions.

    The marginal impact of one output token and the fixed cost of one request
    (time to first token) are cached per model and deployment, which makes
    estimating millions of log rows cheap: impacts are linear in token counts.
    """

    registry: Registry = field(default_factory=default_registry)
    zone: Optional[str] = None
    pue: Optional[Value] = None
    wue: Optional[Value] = None
    carbon_intensity: Optional[float] = None
    hardware: Hardware = DEFAULT_HARDWARE
    prefill: PrefillLike = "central"
    _units: dict = field(default_factory=dict, init=False, repr=False)

    # resolution ---------------------------------------------------------------

    def resolve_model(self, model: str, provider: Optional[str] = None) -> tuple[ModelSpec, Match]:
        m = self.registry.match(model, provider)
        if not m.ok:
            raise ModelNotFoundError(m)
        return m.model, m  # type: ignore[return-value]

    def deployment(
        self,
        spec: ModelSpec,
        zone: Optional[str] = None,
        pue: Optional[Value] = None,
        wue: Optional[Value] = None,
        carbon_intensity: Optional[float] = None,
    ) -> tuple[ProviderSpec, ElectricityMix, Value, Value]:
        prov = self.registry.provider(spec.provider)
        mix = self.registry.zone(zone or self.zone or prov.location)
        ci = carbon_intensity if carbon_intensity is not None else self.carbon_intensity
        if ci is not None:
            if not math.isfinite(ci) or ci < 0:
                raise ValueError("carbon_intensity must be finite and non-negative")
            mix = replace(mix, zone=f"{mix.zone}*", gwp=float(ci), warnings=mix.warnings)
        p = pue if pue is not None else (self.pue if self.pue is not None else prov.pue)
        w = wue if wue is not None else (self.wue if self.wue is not None else prov.wue)
        return prov, mix, p, w

    def units(self, spec: ModelSpec, mix: ElectricityMix, pue: Value, wue: Value) -> tuple[Impacts, Impacts]:
        """(impact of one output token, fixed impact of one request) for a deployment."""
        key = (spec, mix, pue, wue, self.hardware)
        if key not in self._units:
            common = dict(
                active_params_b=spec.active_params_b,
                total_params_b=spec.total_params_b,
                mix=mix,
                pue=pue,
                wue=wue,
                hardware=self.hardware,
                tps=spec.tps,
            )
            per_token = compute_impacts(output_tokens=1.0, ttft=None, **common)
            per_request = compute_impacts(output_tokens=0.0, ttft=spec.ttft, **common) if spec.ttft else ZERO
            self._units[key] = (per_token, per_request)
        return self._units[key]

    def prefill_unit(self, spec: ModelSpec, mix: ElectricityMix, pue: Value, wue: Value) -> Impacts:
        key = ("prefill", spec, mix, pue, wue, self.hardware)
        if key not in self._units:
            self._units[key] = compute_prefill_unit(
                active_params_b=spec.active_params_b,
                total_params_b=spec.total_params_b,
                mix=mix,
                pue=pue,
                wue=wue,
                hardware=self.hardware,
            )
        return self._units[key]

    # estimation ---------------------------------------------------------------

    def estimate(
        self,
        model: str,
        *,
        output_tokens: float,
        input_tokens: float = 0,
        cached_tokens: float = 0,
        provider: Optional[str] = None,
        requests: float = 1,
        zone: Optional[str] = None,
        pue: Optional[Value] = None,
        wue: Optional[Value] = None,
        carbon_intensity: Optional[float] = None,
        prefill: PrefillLike = "default",
        request_latency_s: Optional[float] = None,
    ) -> Estimate:
        """Estimate the impacts of ``requests`` calls totalling the given token counts.

        ``input_tokens`` includes ``cached_tokens``. Reasoning or thinking tokens
        must be counted in ``output_tokens``, which is the case for the usage
        fields returned by OpenAI and Anthropic.
        """
        if any(not math.isfinite(v) or v < 0 for v in (output_tokens, input_tokens, cached_tokens, requests)):
            raise ValueError("token counts and requests must be finite and non-negative")
        if requests == 0 and (input_tokens > 0 or output_tokens > 0):
            raise ValueError("Non-zero token totals require a positive request count")
        if cached_tokens > input_tokens:
            raise ValueError("cached_tokens cannot exceed input_tokens (input_tokens includes cached tokens)")

        spec, match = self.resolve_model(model, provider)
        prov, mix, p, w = self.deployment(spec, zone, pue, wue, carbon_intensity)
        pm = _prefill.resolve(self.prefill if prefill == "default" else prefill)
        per_token, per_request = self.units(spec, mix, p, w)

        if request_latency_s is not None:
            if requests != 1:
                raise ValueError("request_latency_s can only be used for a single request")
            baseline = compute_impacts(
                active_params_b=spec.active_params_b,
                total_params_b=spec.total_params_b,
                output_tokens=output_tokens,
                mix=mix,
                pue=p,
                wue=w,
                hardware=self.hardware,
                tps=spec.tps,
                ttft=spec.ttft,
                request_latency_s=request_latency_s,
            )
        else:
            baseline = per_token.scale(output_tokens) + per_request.scale(requests)

        prefill_imp = self.prefill_unit(spec, mix, p, w).scale(pm.equivalent_tokens(input_tokens, cached_tokens))

        warnings = [WARNING_TEXT.get(c, c) for c in spec.warnings + mix.warnings]
        if match.status is MatchStatus.NORMALIZED:
            warnings.append(f"'{match.query}' matched to registry entry '{spec.name}'.")
        if spec.origin == "custom":
            warnings.append("Custom model: parameters supplied by the user.")
        if pm.alpha and input_tokens:
            warnings.append(
                "Input correction is a GPU-energy sensitivity scenario, not a calibrated prefill measurement."
            )
            if cached_tokens:
                warnings.append(
                    f"Cache factor {pm.cached_factor:g} is a scenario assumption, not an energy measurement."
                )

        return Estimate(
            model=spec,
            match=match,
            provider=prov,
            mix=mix,
            pue=p,
            wue=w,
            input_tokens=float(input_tokens),
            cached_tokens=float(cached_tokens),
            output_tokens=float(output_tokens),
            requests=float(requests),
            prefill_model=pm,
            baseline=baseline,
            prefill=prefill_imp,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def scenarios(
        self, model: str, scenarios: tuple[str, ...] = ("none", "low", "central"), **kwargs: Any
    ) -> dict[str, Estimate]:
        """Run the same request under several prefill assumptions."""
        return {s: self.estimate(model, prefill=s, **kwargs) for s in scenarios}


_DEFAULT_ESTIMATOR: Optional[Estimator] = None


def default_estimator() -> Estimator:
    global _DEFAULT_ESTIMATOR
    if _DEFAULT_ESTIMATOR is None:
        _DEFAULT_ESTIMATOR = Estimator()
    return _DEFAULT_ESTIMATOR


def estimate(model: str, **kwargs: Any) -> Estimate:
    """Estimate the footprint of LLM usage with the default registry.

    >>> from green_ai import estimate
    >>> e = estimate("gpt-4o", provider="openai", input_tokens=20_000, output_tokens=400)
    >>> print(e.total.energy_wh)  # doctest: +SKIP
    """
    return default_estimator().estimate(model, **kwargs)


def register_model(name: str, **kwargs: Any) -> ModelSpec:
    """Register a custom model in the default registry (see :meth:`Registry.register`)."""
    return default_registry().register(name, **kwargs)


__all__ = ["Estimate", "Estimator", "estimate", "register_model", "default_estimator", "PrefillModel"]
