"""Track the footprint of live API calls from the usage objects providers return.

No monkey patching: pass the response (or its ``usage``) after each call.

>>> from green_ai import Tracker
>>> tracker = Tracker(zone="FRA")
>>> resp = client.chat.completions.create(model="gpt-4o", messages=msgs)   # doctest: +SKIP
>>> tracker.add(resp)                                                       # doctest: +SKIP
>>> print(tracker.summary())                                                # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .engine import Impacts
from .estimator import Estimate, Estimator
from .ranges import mean


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Read an attribute or key, following dotted paths, on SDK objects or dicts."""
    for name in names:
        cur = obj
        ok = True
        for part in name.split("."):
            if cur is None:
                ok = False
                break
            if isinstance(cur, dict):
                if part not in cur:
                    ok = False
                    break
                cur = cur[part]
            elif hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default


@dataclass(frozen=True)
class Usage:
    """Provider agnostic token usage of one call. ``input_tokens`` includes cached tokens."""

    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0


def parse_usage(response: Any, model: Optional[str] = None, provider: Optional[str] = None) -> Usage:
    """Extract token usage from an OpenAI, Anthropic, Mistral or Gemini response (or a dict).

    Conventions handled:

    * OpenAI Chat Completions: ``prompt_tokens`` includes ``prompt_tokens_details.cached_tokens``,
      ``completion_tokens`` includes reasoning tokens.
    * OpenAI Responses API: ``input_tokens`` / ``output_tokens`` / ``input_tokens_details.cached_tokens``.
    * Anthropic Messages: ``input_tokens`` excludes ``cache_read_input_tokens`` and
      ``cache_creation_input_tokens``, which are added back here (cache writes count as fresh).
    * Google Gemini: ``usage_metadata.prompt_token_count``, ``candidates_token_count``,
      ``thoughts_token_count`` and ``cached_content_token_count``.
    """
    model = model or _get(response, "model", "model_version", default=None)
    if model is None:
        raise ValueError("Could not read the model name from the response, pass model=...")

    usage = _get(response, "usage", "usage_metadata", default=response)

    # Gemini
    if _get(usage, "prompt_token_count") is not None:
        inp = int(_get(usage, "prompt_token_count", default=0))
        out = int(_get(usage, "candidates_token_count", default=0)) + int(
            _get(usage, "thoughts_token_count", default=0)
        )
        cached = int(_get(usage, "cached_content_token_count", default=0))
        return Usage(str(model), provider or "google_genai", inp, out, cached)

    # Infer provider from the model identifier when usage shapes overlap.
    if provider is None:
        from .registry import default_registry

        match = default_registry().match(str(model))
        if match.ok:
            provider = match.model.provider
        elif str(model).lower().startswith("claude"):
            provider = "anthropic"

    # Anthropic
    if _get(usage, "cache_read_input_tokens", "cache_creation_input_tokens") is not None or (
        provider and provider.lower().startswith("anthropic")
    ):
        if _get(usage, "input_tokens") is None or _get(usage, "output_tokens") is None:
            raise ValueError("Response has no complete Anthropic token usage")
        read = int(_get(usage, "cache_read_input_tokens", default=0))
        write = int(_get(usage, "cache_creation_input_tokens", default=0))
        inp = int(_get(usage, "input_tokens", default=0)) + read + write
        out = int(_get(usage, "output_tokens", default=0))
        return Usage(str(model), provider or "anthropic", inp, out, read)

    # OpenAI style (chat completions, responses API, Mistral, most gateways)
    if (
        _get(usage, "prompt_tokens", "input_tokens") is None
        or _get(usage, "completion_tokens", "output_tokens") is None
    ):
        raise ValueError("Response has no complete token usage. Pass a final response with usage, not a stream chunk.")
    inp = int(_get(usage, "prompt_tokens", "input_tokens", default=0))
    out = int(_get(usage, "completion_tokens", "output_tokens", default=0))
    cached = int(_get(usage, "prompt_tokens_details.cached_tokens", "input_tokens_details.cached_tokens", default=0))
    return Usage(str(model), provider or "openai", inp, out, cached)


@dataclass
class Tracker:
    """Accumulates estimates over many calls."""

    estimator: Estimator = field(default_factory=Estimator)
    zone: Optional[str] = None
    prefill: Any = "default"
    records: list[Estimate] = field(default_factory=list)

    def add(self, response: Any, *, model: Optional[str] = None, provider: Optional[str] = None) -> Estimate:
        u = parse_usage(response, model=model, provider=provider)
        return self.add_usage(u)

    def add_usage(self, u: Usage) -> Estimate:
        e = self.estimator.estimate(
            u.model,
            provider=u.provider,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cached_tokens=u.cached_tokens,
            zone=self.zone,
            prefill=self.prefill,
        )
        self.records.append(e)
        return e

    @property
    def total(self) -> Impacts:
        acc = Impacts()
        for r in self.records:
            acc = acc + r.total
        return acc

    @property
    def baseline(self) -> Impacts:
        acc = Impacts()
        for r in self.records:
            acc = acc + r.baseline
        return acc

    def summary(self) -> str:
        t, b = self.total, self.baseline
        share = 1 - mean(b.energy_kwh) / mean(t.energy_kwh) if mean(t.energy_kwh) else 0.0
        return (
            f"{len(self.records)} calls | energy {t.energy_wh:.4g} Wh | GHG {t.gwp_g:.4g} gCO2eq | "
            f"water {t.water_l:.4g} L | prefill share {share:.0%}"
        )
