"""Compare models on a standard workload and grade them A (lowest impact) to E."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

from .estimator import Estimate, Estimator, default_estimator
from .ranges import mean
from .registry import canonical_provider, strip_date

GRADES = "ABCDE"


def _grade(i: int, n: int) -> str:
    """Spread grades evenly over ranks: the best is A, the worst is E."""
    return GRADES[0] if n <= 1 else GRADES[round(4 * i / (n - 1))]


@dataclass(frozen=True)
class Rated:
    grade: str
    rank: int
    estimate: Estimate

    @property
    def energy_wh(self) -> float:
        return mean(self.estimate.total.energy_wh)

    @property
    def gwp_g(self) -> float:
        return mean(self.estimate.total.gwp_g)


def rate_models(
    *,
    input_tokens: int = 1000,
    output_tokens: int = 1000,
    providers: Optional[Iterable[str]] = None,
    models: Optional[Iterable[str]] = None,
    zone: Optional[str] = None,
    prefill: str = "default",
    estimator: Optional[Estimator] = None,
    include_snapshots: bool = False,
) -> list[Rated]:
    """Run the same request through every model and grade by energy rank.

    Grades are relative to the set of models compared and run evenly from A (least
    energy) to E (most). Use a common ``zone`` to compare models rather than
    datacenter locations.
    """
    est = estimator or default_estimator()
    wanted = {canonical_provider(p) for p in providers} if providers else None
    if models:
        specs = [est.registry.get(m) for m in models]
    else:
        specs = [m for m in est.registry.models if wanted is None or m.provider in wanted]
        if not include_snapshots:
            # Hide dated snapshots (gpt-4o-2024-08-06) when the undated name is registered too.
            names = {m.key for m in specs}
            specs = [m for m in specs if strip_date(m.name) == m.name or (m.provider, strip_date(m.name)) not in names]
    results = [
        est.estimate(
            s.name,
            provider=s.provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            zone=zone,
            prefill=prefill,
        )
        for s in specs
    ]
    results.sort(key=lambda e: mean(e.total.energy_kwh))
    n = len(results)
    return [Rated(_grade(i, n), i + 1, e) for i, e in enumerate(results)]
