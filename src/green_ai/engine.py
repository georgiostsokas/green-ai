"""Bottom-up inference impact model.

The model follows the open LLM inference methodology published by EcoLogits
(Rincé & Banse, JOSS 2025, https://doi.org/10.21105/joss.07471). The test suite
checks it numerically against that reference for every registered model.

Pipeline for one request that generates ``n_out`` tokens:

1. GPU energy per output token, from a regression on measured H100 data
   ``e_gpu = alpha * exp(beta * batch) * P_active + gamma``  (Wh per token).
2. Generation latency ``t = n_out / tps + ttft`` when the provider throughput is
   known, else ``n_out * (a * P_active + b * batch + c)``. Capped by the measured
   request latency when one is supplied.
3. Number of GPUs needed to hold the weights, rounded up to a power of two
   ``n_gpu = 2 ** ceil(log2(ceil(1.2 * P_total * bits / 8 / gpu_memory)))``.
4. Server (non GPU) energy ``t * P_server * n_gpu / gpus_per_server / batch``.
5. Request energy ``PUE * (server + n_gpu * e_gpu * n_out)``.
6. Usage impacts ``energy * electricity_mix_factor`` and water
   ``IT_energy * (WUE_dc + PUE * WUE_mix)``.
7. Embodied impacts allocated by occupied server time
   ``t * (server_share + n_gpu * gpu) / (lifetime * batch)``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Optional

from .ranges import Range, Value, add, hi, lo, mean, mul

# ---------------------------------------------------------------------------------
# Hardware and regression defaults (EcoLogits 0.11 reference values)
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Hardware:
    """Reference serving infrastructure. Override any field to model your own fleet."""

    gpu_energy_alpha: float = 1.1665273170451914e-06
    gpu_energy_beta: float = -0.011205921025579175
    gpu_energy_gamma: float = 4.052928146734005e-05
    latency_alpha: float = 0.0006785088094353663
    latency_beta: float = 0.0003119310311688259
    latency_gamma: float = 0.019473717579473387
    quantization_bits: int = 16
    gpu_memory_gb: float = 80.0
    gpu_embodied_gwp: float = 273.0  # kgCO2eq per GPU
    gpu_embodied_adpe: float = 0.00895  # kgSbeq per GPU
    gpu_embodied_pe: float = 3721.0  # MJ per GPU
    server_gpu_count: int = 8
    server_power_kw: float = 1.2  # excluding GPUs
    server_embodied_gwp: float = 5700.0
    server_embodied_adpe: float = 0.37
    server_embodied_pe: float = 70000.0
    lifetime_s: float = 3 * 365 * 24 * 60 * 60
    batch_size: float = 64

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if not math.isfinite(v):
                raise ValueError(f"{f.name} must be finite")
            if f.name != "gpu_energy_beta" and v <= 0:
                raise ValueError(f"{f.name} must be positive")

    def with_overrides(self, **kwargs: float) -> Hardware:
        return replace(self, **{k: v for k, v in kwargs.items() if v is not None})


DEFAULT_HARDWARE = Hardware()


@dataclass(frozen=True)
class ElectricityMix:
    """Impact factors of one kWh of electricity in a zone."""

    zone: str
    gwp: float  # kgCO2eq / kWh
    adpe: float  # kgSbeq / kWh
    pe: float  # MJ / kWh
    wue: float  # L / kWh (water consumed to produce the electricity)
    warnings: tuple[str, ...] = ()


# ---------------------------------------------------------------------------------
# Impact container
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Impacts:
    """Environmental impacts of one request or of a sum of requests.

    Units: energy in kWh, GWP in kgCO2eq, ADPe in kgSbeq, PE in MJ, water in litres.
    Values are floats or :class:`~green_ai.ranges.Range` intervals.
    """

    energy_kwh: Value = 0.0
    gwp_usage: Value = 0.0
    gwp_embodied: Value = 0.0
    adpe_usage: Value = 0.0
    adpe_embodied: Value = 0.0
    pe_usage: Value = 0.0
    pe_embodied: Value = 0.0
    water_l: Value = 0.0

    @property
    def gwp(self) -> Value:
        return add(self.gwp_usage, self.gwp_embodied)

    @property
    def adpe(self) -> Value:
        return add(self.adpe_usage, self.adpe_embodied)

    @property
    def pe(self) -> Value:
        return add(self.pe_usage, self.pe_embodied)

    @property
    def energy_wh(self) -> Value:
        return mul(self.energy_kwh, 1000.0)

    @property
    def gwp_g(self) -> Value:
        return mul(self.gwp, 1000.0)

    def __add__(self, other: Impacts) -> Impacts:
        return Impacts(**{f.name: add(getattr(self, f.name), getattr(other, f.name)) for f in fields(self)})

    def scale(self, k: float) -> Impacts:
        if k < 0:
            raise ValueError("Impacts can only be scaled by a non-negative factor")
        return Impacts(**{f.name: mul(getattr(self, f.name), k) for f in fields(self)})

    def flat(self, prefix: str = "") -> dict[str, float]:
        """Flatten to ``{name: mean, name_min: lo, name_max: hi}`` including totals."""
        out: dict[str, float] = {}
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values.update(gwp=self.gwp, adpe=self.adpe, pe=self.pe)
        for name, v in values.items():
            out[f"{prefix}{name}"] = mean(v)
            out[f"{prefix}{name}_min"] = lo(v)
            out[f"{prefix}{name}_max"] = hi(v)
        return out


ZERO = Impacts()


# ---------------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------------


def gpu_energy_per_token_kwh(active_params_b: float, hw: Hardware) -> float:
    wh = hw.gpu_energy_alpha * math.exp(hw.gpu_energy_beta * hw.batch_size) * active_params_b + hw.gpu_energy_gamma
    return wh / 1000.0


def latency_per_token_s(active_params_b: float, hw: Hardware, tps: Optional[float]) -> float:
    if tps is not None:
        return 1.0 / tps
    return hw.latency_alpha * active_params_b + hw.latency_beta * hw.batch_size + hw.latency_gamma


def gpu_count(total_params_b: float, hw: Hardware) -> int:
    memory_gb = 1.2 * total_params_b * hw.quantization_bits / 8
    n = math.ceil(memory_gb / hw.gpu_memory_gb)
    return 2 ** math.ceil(math.log2(n))


@dataclass
class _Trace:
    """Intermediate values of one evaluation, useful for debugging and teaching."""

    gpu_count: int = 0
    latency_s: float = 0.0
    gpu_energy_kwh: float = 0.0
    server_energy_kwh: float = 0.0
    it_energy_kwh: float = 0.0
    extra: dict = field(default_factory=dict)


def _evaluate(
    active_b: float,
    total_b: float,
    output_tokens: float,
    mix: ElectricityMix,
    pue: Value,
    wue: Value,
    hw: Hardware,
    tps: Optional[float],
    ttft: Optional[float],
    request_latency_s: float,
) -> tuple[Impacts, _Trace]:
    n_gpu = gpu_count(total_b, hw)
    e_gpu = output_tokens * gpu_energy_per_token_kwh(active_b, hw)

    latency = output_tokens * latency_per_token_s(active_b, hw, tps) + (ttft or 0.0)
    latency = min(latency, request_latency_s)

    e_server = (latency / 3600.0) * hw.server_power_kw * (n_gpu / hw.server_gpu_count) / hw.batch_size
    it_energy = e_server + n_gpu * e_gpu
    energy = mul(pue, it_energy)

    water = mul(it_energy, add(wue, mul(pue, mix.wue)))

    share = n_gpu / hw.server_gpu_count
    alloc = latency / (hw.lifetime_s * hw.batch_size)
    emb_gwp = alloc * (share * hw.server_embodied_gwp + n_gpu * hw.gpu_embodied_gwp)
    emb_adpe = alloc * (share * hw.server_embodied_adpe + n_gpu * hw.gpu_embodied_adpe)
    emb_pe = alloc * (share * hw.server_embodied_pe + n_gpu * hw.gpu_embodied_pe)

    impacts = Impacts(
        energy_kwh=energy,
        gwp_usage=mul(energy, mix.gwp),
        gwp_embodied=emb_gwp,
        adpe_usage=mul(energy, mix.adpe),
        adpe_embodied=emb_adpe,
        pe_usage=mul(energy, mix.pe),
        pe_embodied=emb_pe,
        water_l=water,
    )
    trace = _Trace(n_gpu, latency, n_gpu * e_gpu, e_server, it_energy)
    return impacts, trace


def _combine(a: Impacts, b: Impacts) -> Impacts:
    """Merge the low-parameter run ``a`` and the high-parameter run ``b`` into intervals."""
    return Impacts(**{f.name: Range(lo(getattr(a, f.name)), hi(getattr(b, f.name))) for f in fields(a)})


def compute_impacts(
    *,
    active_params_b: Value,
    total_params_b: Value,
    output_tokens: float,
    mix: ElectricityMix,
    pue: Value,
    wue: Value,
    hardware: Hardware = DEFAULT_HARDWARE,
    tps: Optional[float] = None,
    ttft: Optional[float] = None,
    request_latency_s: Optional[float] = None,
) -> Impacts:
    """Impacts of a single generation request (decode phase as modelled by EcoLogits).

    Parameter counts are in billions and can be intervals. When either is an
    interval the model is evaluated at the low and at the high end and the results
    are merged into an interval, like EcoLogits does.
    """
    if not math.isfinite(output_tokens) or output_tokens < 0:
        raise ValueError("output_tokens must be finite and non-negative")
    if tps is not None and (not math.isfinite(tps) or tps <= 0):
        raise ValueError("tps must be finite and positive")
    if ttft is not None and (not math.isfinite(ttft) or ttft < 0):
        raise ValueError("ttft must be finite and non-negative")
    if request_latency_s is not None and (not math.isfinite(request_latency_s) or request_latency_s < 0):
        raise ValueError("request_latency_s must be finite and non-negative")
    if lo(pue) < 1 or not math.isfinite(hi(pue)) or lo(wue) < 0 or not math.isfinite(hi(wue)):
        raise ValueError("PUE must be finite and >= 1; WUE must be finite and >= 0")
    cap = math.inf if request_latency_s is None else float(request_latency_s)
    common = dict(
        output_tokens=output_tokens, mix=mix, pue=pue, wue=wue, hw=hardware, tps=tps, ttft=ttft, request_latency_s=cap
    )

    if isinstance(active_params_b, Range) or isinstance(total_params_b, Range):
        a_lo, _ = _evaluate(lo(active_params_b), lo(total_params_b), **common)
        a_hi, _ = _evaluate(hi(active_params_b), hi(total_params_b), **common)
        return _combine(a_lo, a_hi)

    impacts, _ = _evaluate(float(active_params_b), float(total_params_b), **common)
    return impacts


def compute_prefill_unit(
    *,
    active_params_b: Value,
    total_params_b: Value,
    mix: ElectricityMix,
    pue: Value,
    wue: Value,
    hardware: Hardware = DEFAULT_HARDWARE,
) -> Impacts:
    """Impact of one GPU decode-token equivalent, for the experimental input extension.

    The coefficient is an energy ratio from GPU phase measurements. It therefore
    scales GPU IT energy only. Apply PUE once and derive operational impacts;
    leave server energy, latency and embodied allocation in the baseline.
    """

    def evaluate(active: float, total: float) -> Impacts:
        it = gpu_count(total, hardware) * gpu_energy_per_token_kwh(active, hardware)
        facility = mul(pue, it)
        return Impacts(
            energy_kwh=facility,
            gwp_usage=mul(facility, mix.gwp),
            adpe_usage=mul(facility, mix.adpe),
            pe_usage=mul(facility, mix.pe),
            water_l=mul(it, add(wue, mul(pue, mix.wue))),
        )

    a = evaluate(lo(active_params_b), lo(total_params_b))
    if isinstance(active_params_b, Range) or isinstance(total_params_b, Range):
        return _combine(a, evaluate(hi(active_params_b), hi(total_params_b)))
    return a


def trace_request(
    *,
    active_params_b: float,
    total_params_b: float,
    output_tokens: float,
    mix: ElectricityMix,
    pue: float,
    wue: float,
    hardware: Hardware = DEFAULT_HARDWARE,
    tps: Optional[float] = None,
    ttft: Optional[float] = None,
) -> dict:
    """Return every intermediate quantity for a point estimate (for docs and teaching)."""
    impacts, t = _evaluate(active_params_b, total_params_b, output_tokens, mix, pue, wue, hardware, tps, ttft, math.inf)
    return {**asdict(t), **impacts.flat()}
