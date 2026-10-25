"""Task-level estimates, quality filters and transparent energy/carbon rankings."""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from .estimator import Estimate, Estimator, default_estimator
from .ranges import Range, Value, hi, lo, mean, mul
from .registry import ModelSpec, Registry

GRADES = "ABCDE"


def _grade(i: int, n: int) -> str:
    return GRADES[0] if n <= 1 else GRADES[round(4 * i / (n - 1))]


@dataclass(frozen=True)
class TaskProfile:
    """Average tokens and calls per task, with correlated input/output scenarios.

    `token_basis` describes the source of tokens; footprint ranges describe
    model/deployment assumptions, not statistical confidence in measurements.
    """

    model: str
    scenarios: tuple[tuple[float, float], ...]
    provider: Optional[str] = None
    cached_tokens: float = 0.0
    label: Optional[str] = None
    score: Optional[float] = None
    cost_per_task: Optional[float] = None
    source: str = "user"
    note: str = ""
    requests_per_task: float = 1.0
    task_set: str = "user workload"
    token_basis: str = "observed"
    index_version: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("A task profile needs at least one (input_tokens, output_tokens) pair")
        for inp, out in self.scenarios:
            if any(not math.isfinite(v) or v < 0 for v in (inp, out)):
                raise ValueError("Tokens per task must be finite and non-negative")
            if not math.isfinite(self.cached_tokens) or not 0 <= self.cached_tokens <= inp:
                raise ValueError("Cached tokens must be finite, non-negative and no greater than input")
        if not math.isfinite(self.requests_per_task) or self.requests_per_task <= 0:
            raise ValueError("Requests per task must be finite and positive")
        for name in ("score", "cost_per_task"):
            v = getattr(self, name)
            if v is not None and (not math.isfinite(v) or v < 0):
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def exact(cls, model: str, input_tokens: float, output_tokens: float, **kw) -> TaskProfile:
        return cls(model=model, scenarios=((float(input_tokens), float(output_tokens)),), **kw)

    @property
    def name(self) -> str:
        return self.label or self.model

    @property
    def is_estimated(self) -> bool:
        return len(self.scenarios) > 1 or self.token_basis in {"cost-inferred", "output-observed/input-assumed"}

    @property
    def input_tokens(self) -> Value:
        return _envelope([s[0] for s in self.scenarios])

    @property
    def output_tokens(self) -> Value:
        return _envelope([s[1] for s in self.scenarios])


@dataclass(frozen=True)
class TaskFootprint:
    profile: TaskProfile
    model: ModelSpec
    energy_wh: Value
    gwp_g: Value
    water_ml: Value
    prefill_share: float
    estimates: tuple[Estimate, ...] = field(repr=False, default=())

    @property
    def efficiency(self) -> Optional[float]:
        e = mean(self.energy_wh)
        return self.profile.score / e if self.profile.score is not None and e else None


def _envelope(values: list[Value]) -> Value:
    a, b = min(lo(v) for v in values), max(hi(v) for v in values)
    return a if a == b else Range(a, b)


def task_footprint(
    profile: TaskProfile, *, estimator: Optional[Estimator] = None, zone: Optional[str] = None, prefill="default"
) -> TaskFootprint:
    est = estimator or default_estimator()
    estimates = tuple(
        est.estimate(
            profile.model,
            provider=profile.provider,
            input_tokens=i,
            cached_tokens=profile.cached_tokens,
            output_tokens=o,
            requests=profile.requests_per_task,
            zone=zone,
            prefill=prefill,
        )
        for i, o in profile.scenarios
    )
    return TaskFootprint(
        profile,
        estimates[0].model,
        _envelope([e.total.energy_wh for e in estimates]),
        _envelope([e.total.gwp_g for e in estimates]),
        _envelope([mul(e.total.water_l, 1000) for e in estimates]),
        sum(e.prefill_share for e in estimates) / len(estimates),
        estimates,
    )


@dataclass(frozen=True)
class TaskRank:
    rank: int
    grade: str
    footprint: TaskFootprint
    token_rank: int
    metric: str = "energy"
    pareto: bool = False

    @property
    def shift(self) -> Optional[int]:
        """Only energy ranks are comparable to energy per-token ranks."""
        return self.rank - self.token_rank if self.metric == "energy" else None


def rank_tasks(
    profiles: list[TaskProfile],
    *,
    estimator: Optional[Estimator] = None,
    zone: Optional[str] = "WOR",
    prefill="default",
    by: str = "energy",
    min_score: Optional[float] = None,
) -> tuple[list[TaskRank], list[tuple[TaskProfile, str]]]:
    """Rank one workload by energy, carbon, water, or benchmark points per Wh.

    A common zone keeps grid factors consistent; provider PUE/WUE remain visible
    deployment assumptions. Grades and Pareto flags use interval midpoints and
    are relative to the selected set, with no claim of statistical significance.
    """
    if by not in {"energy", "carbon", "water", "efficiency"}:
        raise ValueError("by must be 'energy', 'carbon', 'water' or 'efficiency'")
    if min_score is not None and (not math.isfinite(min_score) or min_score < 0):
        raise ValueError("Minimum score must be finite and non-negative")
    if len({(p.task_set, p.index_version) for p in profiles}) > 1:
        raise ValueError("Compare one task set and index version at a time")
    est = estimator or default_estimator()
    fps, skipped = [], []
    for p in profiles:
        if min_score is not None and (p.score is None or p.score < min_score):
            skipped.append((p, "Below the minimum score, or no comparable score available"))
            continue
        if by == "efficiency" and p.score is None:
            skipped.append((p, "No quality score available for an efficiency ranking"))
            continue
        try:
            fps.append(task_footprint(p, estimator=est, zone=zone, prefill=prefill))
        except LookupError as exc:
            skipped.append((p, str(exc)))

    def value(f):
        return {
            "energy": mean(f.energy_wh),
            "carbon": mean(f.gwp_g),
            "water": mean(f.water_ml),
            "efficiency": -(f.efficiency or 0),
        }[by]

    fps.sort(key=lambda f: (value(f), f.profile.name))
    per_token = {
        id(f): mean(
            est.estimate(
                f.model.name, provider=f.model.provider, output_tokens=1000, zone=zone, prefill="none"
            ).total.energy_wh
        )
        for f in fps
    }

    # Competition ranks keep identical estimates tied, including dated aliases.
    def rank(v, vals):
        return 1 + sum(x < v and not math.isclose(x, v, rel_tol=1e-12) for x in vals)

    vals, token_vals = [value(f) for f in fps], list(per_token.values())

    def impact_value(f):
        return mean(f.gwp_g) if by == "carbon" else mean(f.water_ml) if by == "water" else mean(f.energy_wh)

    def pareto(f):
        if f.profile.score is None:
            return False
        impact = impact_value(f)
        return not any(
            g is not f
            and g.profile.score is not None
            and g.profile.score >= f.profile.score
            and impact_value(g) <= impact
            and (g.profile.score > f.profile.score or impact_value(g) < impact)
            for g in fps
        )

    ranking = []
    for f in fps:
        r = rank(value(f), vals)
        ranking.append(TaskRank(r, _grade(r - 1, len(fps)), f, rank(per_token[id(f)], token_vals), by, pareto(f)))
    return ranking, skipped


CSV_ALIASES = {
    "model": ["model", "model_name", "slug", "name"],
    "provider": ["provider", "model_creator", "creator"],
    "label": ["label", "display_name"],
    "input_tokens": ["input_tokens_per_task", "input_tokens", "prompt_tokens_per_task", "prompt_tokens"],
    "output_tokens": ["output_tokens_per_task", "output_tokens", "completion_tokens_per_task", "completion_tokens"],
    "answer_tokens": ["answer_tokens_per_task", "answer_tokens"],
    "reasoning_tokens": ["reasoning_tokens_per_task", "reasoning_tokens", "thinking_tokens_per_task"],
    "cached_tokens": ["cached_tokens_per_task", "cached_tokens", "cache_read_tokens_per_task"],
    "score": ["score", "intelligence_index", "artificial_analysis_intelligence_index"],
    "cost_per_task": ["cost_per_task", "cost_per_task_usd", "usd_per_task"],
    "requests_per_task": ["requests_per_task", "calls_per_task"],
    "task_set": ["task_set", "benchmark"],
    "index_version": ["index_version", "intelligence_index_version"],
    "source": ["source"],
}


def normalize_column(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def load_profiles(
    path,
    *,
    source: Optional[str] = None,
    registry: Optional[Registry] = None,
    column_map: Optional[dict] = None,
    output_includes_reasoning: bool = True,
    input_ratio: Optional[Range] = None,
) -> list[TaskProfile]:
    """Read actual per-task CSV values, including AA exports. No totals/cost division.

    Input includes cache reads. Output includes reasoning unless explicitly set
    otherwise. When only answer + reasoning columns exist, they are summed once.
    """
    text = path.read() if hasattr(path, "read") else Path(path).read_text(encoding="utf-8-sig")
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO("\n".join(x for x in text.splitlines() if not x.lstrip().startswith("#"))))
    cols = {normalize_column(c): c for c in (reader.fieldnames or [])}
    mapping = {k: next((cols[n] for n in aliases if n in cols), None) for k, aliases in CSV_ALIASES.items()}
    mapping.update(column_map or {})
    if (
        not mapping["model"]
        or not (mapping["input_tokens"] or input_ratio is not None)
        or not (mapping["output_tokens"] or mapping["answer_tokens"])
    ):
        raise ValueError("Task CSV needs model, input tokens per task and output (or answer) tokens per task")
    profiles = []
    for line, row in enumerate(reader, 2):

        def get(k, current_row=row):
            return (current_row.get(mapping.get(k)) or "").strip()

        def num(k, default=None, current_line=line, lookup=get):
            try:
                return float(lookup(k)) if lookup(k) else default
            except ValueError:
                raise ValueError(f"Invalid numeric {k} on task CSV line {current_line}") from None

        inp = num("input_tokens")
        out = num("output_tokens")
        if out is None:
            answer = num("answer_tokens")
            out = None if answer is None else answer + num("reasoning_tokens", 0)
        elif not output_includes_reasoning:
            out += num("reasoning_tokens", 0)
        if (inp is None and input_ratio is None) or out is None:
            raise ValueError(f"Missing token usage on task CSV line {line}")
        model, provider = get("model"), get("provider") or None
        if not model:
            raise ValueError(f"Missing model on task CSV line {line}")
        if registry is not None:
            from .aa import match_item

            spec, _ = match_item({"slug": model, "name": model, "model_creator": {"name": provider or ""}}, registry)
            if spec:
                model, provider = spec.name, spec.provider
        origin = source or get("source") or "csv"
        if get("source").startswith("synthetic"):
            origin = get("source")
        basis = "synthetic" if origin.startswith("synthetic") else "observed"
        if inp is None:
            scenarios = ((out * input_ratio.lo, out), (out * input_ratio.hi, out))
            basis = "output-observed/input-assumed"
        else:
            scenarios = ((inp, out),)
        profiles.append(
            TaskProfile(
                model,
                scenarios,
                provider=provider,
                label=get("label") or get("model"),
                cached_tokens=num("cached_tokens", 0),
                score=num("score"),
                cost_per_task=num("cost_per_task"),
                requests_per_task=num("requests_per_task", 1),
                task_set=get("task_set")
                or (
                    "Artificial Analysis Intelligence Index"
                    if origin.startswith("artificial_analysis")
                    else "user workload"
                ),
                index_version=get("index_version") or None,
                source=origin,
                token_basis=basis,
                note="Input tokens inferred from the selected input/output ratio" if inp is None else "",
            )
        )
    return profiles


def ranking_rows(ranking: list[TaskRank]) -> list[dict]:
    rows = []
    for r in ranking:
        f, p = r.footprint, r.footprint.profile
        row = {
            "rank": r.rank,
            "grade": r.grade,
            "rank_metric": r.metric,
            "model": p.name,
            "registry_model": f"{f.model.provider}/{f.model.name}",
            "score": p.score,
            "input_tokens_per_task": mean(p.input_tokens),
            "output_tokens_per_task": mean(p.output_tokens),
            "cached_tokens_per_task": p.cached_tokens,
            "requests_per_task": p.requests_per_task,
            "tokens_estimated": p.is_estimated,
            "token_basis": p.token_basis,
            "prefill_share": f.prefill_share,
            "cost_per_task_usd": p.cost_per_task,
            "points_per_wh": f.efficiency,
            "rank_per_token": r.token_rank,
            "rank_shift": r.shift,
            "pareto": r.pareto,
            "source": p.source,
            "source_attribution": "Artificial Analysis (https://artificialanalysis.ai)"
            if p.source.startswith("artificial_analysis")
            else p.source,
            "task_set": p.task_set,
            "index_version": p.index_version,
            "note": p.note,
            "electricity_zone": f.estimates[0].mix.zone,
            "prefill_alpha": f.estimates[0].prefill_model.alpha,
            "cached_factor": f.estimates[0].prefill_model.cached_factor,
        }
        for name, val in (("energy_wh", f.energy_wh), ("gco2eq", f.gwp_g), ("water_ml", f.water_ml)):
            row[f"{name}_per_task"] = mean(val)
            row[f"{name}_min"] = lo(val)
            row[f"{name}_max"] = hi(val)
        rows.append(row)
    return rows


def save_ranking(ranking: list[TaskRank], path: Union[str, Path]) -> Path:
    rows = ranking_rows(ranking)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["rank"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v for k, v in row.items()}
            )
    return Path(path)
