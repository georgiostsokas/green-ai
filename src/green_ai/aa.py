"""Artificial Analysis adapters with explicit task-level provenance.

Official model-list API responses expose benchmark token TOTALS and weighted
cost per task. Their ratio is not a task count. API-only profiles are labelled
cost-inferred scenarios. Use actual per-task CSV exports, or an explicit
versioned benchmark plan plus evaluation-level counts, for observed profiles.
The dashboard uses session memory; the optional CLI cache is key/tier scoped.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .ranges import Range
from .registry import Registry, canonical_provider, default_registry
from .tasks import TaskProfile
from .update import cache_root

BASE = "https://artificialanalysis.ai/api/v2"
ENDPOINTS = {"pro": "/language/models", "free": "/language/models/free"}
ATTRIBUTION = "Benchmark token and cost data: Artificial Analysis (artificialanalysis.ai)"
CACHE_HOURS = 24
DEFAULT_OUTPUT_SHARE = Range(0.5, 0.9)

CREATORS = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google_genai",
    "mistral": "mistralai",
    "mistral ai": "mistralai",
    "cohere": "cohere",
}

_EFFORT = re.compile(
    r"[\s\-_]*\(?(reasoning|non[\s\-]?reasoning|thinking|extended thinking|adaptive reasoning|"
    r"x?high|medium|low|minimal|max|preview)\)?$"
)


class AAError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------------


def _cache_file(key: str, tier: str) -> Path:
    account = hashlib.sha256(f"{tier}:{key}".encode()).hexdigest()[:24]
    return cache_root() / f"aa_{account}.json"


def _request(path: str, key: str, page: Optional[int]) -> dict:
    url = f"{BASE}{path}" + (f"?page={page}" if page is not None else "")
    req = urllib.request.Request(url, headers={"x-api-key": key, "User-Agent": "green-ai"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - fixed https URL
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise AAError("Artificial Analysis rejected the API key (401). Check AA_API_KEY.") from None
        if exc.code == 403:
            raise PermissionError("tier") from None
        if exc.code == 429:
            raise AAError("Artificial Analysis rate limit reached (429). Cached data is reused for 24 h.") from None
        raise AAError(f"Artificial Analysis API error {exc.code}") from None


def _fetch_all(path: str, key: str) -> dict:
    page, items, first = 1, [], None
    while True:
        payload = _request(path, key, page)
        first = first or payload
        items.extend(payload.get("data") or [])
        pag = payload.get("pagination") or {}
        if not pag.get("has_more"):
            break
        if page >= 20:
            raise AAError("AA pagination exceeded 20 pages; refusing to return an incomplete model list")
        page += 1
    return {
        "tier": first.get("tier"),
        "intelligence_index_version": first.get("intelligence_index_version"),
        "data": items,
    }


def fetch(api_key: Optional[str] = None, *, refresh: bool = False, tier: str = "auto", use_cache: bool = True) -> dict:
    """Fetch the language model list, from the local cache when it is fresh.

    ``tier="auto"`` tries the Pro endpoint first and falls back to the free one.
    Returns ``{"tier", "intelligence_index_version", "fetched_at", "data"}``.
    """
    if tier not in {"auto", "pro", "free"}:
        raise AAError("tier must be 'auto', 'pro' or 'free'")
    key = api_key or os.environ.get("AA_API_KEY")
    if not key:
        raise AAError("No Artificial Analysis API key. Get one at artificialanalysis.ai and set AA_API_KEY.")
    cache = _cache_file(key, tier)
    if use_cache and not refresh and cache.exists():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if time.time() - cached.get("fetched_at", 0) < CACHE_HOURS * 3600:
                return cached
        except (OSError, ValueError):
            pass
    order = ["pro", "free"] if tier == "auto" else [tier]
    result = None
    for t in order:
        try:
            result = _fetch_all(ENDPOINTS[t], key)
            result["tier"] = result.get("tier") or t
            break
        except PermissionError:
            continue
    if result is None:
        raise AAError("This API key has no access to the language model endpoints.")
    result["fetched_at"] = time.time()
    try:
        if use_cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result), encoding="utf-8")
            cache.chmod(0o600)
    except OSError:
        pass
    return result


# ---------------------------------------------------------------------------------
# Mapping Artificial Analysis models to registry entries
# ---------------------------------------------------------------------------------


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", s.lower()).strip("-")


def name_candidates(item: dict) -> list[str]:
    """Registry style names to try for an Artificial Analysis model entry."""
    raw = [item.get("slug") or "", item.get("name") or ""]
    out: list[str] = []

    def add(c: str) -> None:
        c = c.strip("-")
        if c and c not in out:
            out.append(c)

    for r in raw:
        base = _slugify(r)
        stripped = base
        while True:
            nxt = _EFFORT.sub("", stripped).strip("-")
            if nxt == stripped:
                break
            stripped = nxt
        for c in (base, stripped):
            add(c)
            # claude-4-5-sonnet -> claude-sonnet-4-5 ; claude-4-sonnet -> claude-sonnet-4
            m = re.match(r"^claude-(\d+)(?:[.-](\d+))?-(sonnet|opus|haiku)(.*)$", c)
            if m:
                ver = m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
                add(f"claude-{m.group(3)}-{ver}{m.group(4)}")
            # gemini-2-5-pro -> gemini-2.5-pro ; gpt-4-1 -> gpt-4.1
            dotted = re.sub(r"(?<=\D)(\d)-(\d)(?=-|$)", r"\1.\2", c)
            add(dotted)
    return out


def match_item(item: dict, registry: Registry, overrides: Optional[dict[str, str]] = None):
    """Return (ModelSpec or None, how) for one Artificial Analysis entry."""
    slug = item.get("slug") or ""
    if overrides and slug in overrides:
        target = overrides[slug]
        prov, _, name = target.partition("/") if "/" in target else (None, "", target)
        m = registry.match(name, prov)
        return (m.model, "override") if m.ok else (None, f"override '{target}' not found")

    creator = ((item.get("model_creator") or {}).get("name") or "").strip().lower()
    provider = CREATORS.get(creator)
    for cand in name_candidates(item):
        for prov in [provider, None] if provider else [None]:
            m = registry.match(cand, prov)
            if m.ok:
                return m.model, "name"

    hf = item.get("huggingface_url") or ""
    hm = re.search(r"huggingface\.co/([^/]+/[^/?#]+)", hf)
    if hm:
        m = registry.match(hm.group(1), "huggingface_hub")
        if m.ok:
            return m.model, "huggingface"

    params = item.get("parameters") or {}
    total = params.get("total")
    if total:
        # Open weights model with disclosed size: register it from Artificial Analysis metadata.
        active = params.get("active") or None
        name = f"aa/{slug or _slugify(item.get('name', 'model'))}"
        spec = registry.register(
            name,
            total_params=float(total),
            active_params=float(active) if active and float(active) != float(total) else None,
            provider=canonical_provider(creator) or "open_weights",
            source=hf or "artificialanalysis.ai",
            replace=True,
        )
        return spec, "parameters"
    return None, "no registry match"


# ---------------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class AAProfiles:
    profiles: list[TaskProfile]
    unmatched: list[tuple[str, str]]
    tier: str
    index_version: Optional[str]

    @property
    def attribution(self) -> str:
        v = f" v{self.index_version}" if self.index_version else ""
        return f"{ATTRIBUTION}, Intelligence Index{v}, {self.tier} tier."


def _num(d: Any, *path: str) -> Optional[float]:
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    try:
        value = float(d) if d is not None else None
        return value if value is not None and math.isfinite(value) and value >= 0 else None
    except (TypeError, ValueError):
        return None


def weighted_tokens(item: dict, plan: dict, index_version) -> tuple[float, float, float, float]:
    """Combine evaluation totals with explicit tasks, repeat counts and weights.

    `tasks` is the denominator corresponding to that evaluation's token totals;
    if totals include repeats, include those repeats in the denominator. Plans
    are supplied by the caller and pinned to the payload's index version.
    No missing evaluations are silently dropped or reweighted.
    """
    if str(plan.get("index_version")) != str(index_version):
        raise ValueError("Benchmark plan and Artificial Analysis index versions differ")
    entries = plan.get("benchmarks") or {}
    if not entries:
        raise ValueError("Benchmark plan needs benchmark weights and task denominators")
    weight_sum = sum(float(v["weight"]) for v in entries.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Benchmark weights must sum to 1; missing benchmarks cannot be reweighted")
    counts = item.get("evaluation_token_counts") or {}
    inp = out = cached = requests = 0.0
    for name, meta in entries.items():
        weight, tasks = float(meta["weight"]), float(meta["tasks"])
        calls = float(meta.get("requests_per_task", 1))
        if not all(math.isfinite(v) for v in (weight, tasks, calls)) or weight < 0 or tasks <= 0 or calls <= 0:
            raise ValueError("Weights must be non-negative; task counts and calls must be positive and finite")
        if weight == 0:
            continue
        c = counts.get(name) or {}
        i, o = _num(c, "input_tokens"), _num(c, "output_tokens")
        if o is None:
            answer, reasoning = _num(c, "answer_tokens"), _num(c, "reasoning_tokens")
            if answer is not None and reasoning is not None:
                o = answer + reasoning
        if i is None or o is None:
            raise ValueError(f"Missing evaluation token counts: {name}")
        cache = _num(c, "cached_tokens") or 0
        if cache > i:
            raise ValueError(f"Cached tokens exceed input: {name}")
        inp += weight * i / tasks
        out += weight * o / tasks
        cached += weight * cache / tasks
        requests += weight * calls
    return inp, out, cached, requests


def to_profiles(
    payload: dict,
    *,
    registry: Optional[Registry] = None,
    output_share: Range = DEFAULT_OUTPUT_SHARE,
    overrides: Optional[dict[str, str]] = None,
    benchmark_plan: Optional[dict] = None,
) -> AAProfiles:
    """Use weighted evaluation counts with a plan, otherwise cost-derived scenarios.

    A model's aggregate benchmark token counts are never divided by total cost /
    weighted task cost. Cost inference assumes all input is fresh, one call per
    task and no price breaks. It is a sensitivity envelope under that assumption,
    not a physical bound or an observed token count.
    """
    if output_share.lo < 0 or output_share.hi > 1:
        raise ValueError("Output cost share must be within [0, 1]")
    reg = registry or default_registry()
    profiles, unmatched = [], []
    version = payload.get("intelligence_index_version")
    data = payload.get("data") or []
    if isinstance(data, dict):
        data = [data]
    for item in data:
        label = item.get("name") or item.get("slug") or "?"
        score = _num(item, "evaluations", "artificial_analysis_intelligence_index")
        if score is None:
            continue
        per_task = _num(item, "artificial_analysis_intelligence_index_cost", "cost_per_task", "total_cost")
        spec, how = match_item(item, reg, overrides)
        if spec is None:
            unmatched.append((label, how))
            continue
        cached, requests = 0.0, 1.0
        note = f"matched by {how}"
        if benchmark_plan is not None:
            try:
                inp, out, cached, requests = weighted_tokens(item, benchmark_plan, version)
            except (ValueError, KeyError, TypeError) as exc:
                unmatched.append((label, str(exc)))
                continue
            scenarios = ((inp, out),)
            basis, source = "benchmark-weighted", "artificial_analysis:evaluation-tokens"
            note += "; explicit versioned weights and task denominators; unreported cache reads treated as fresh"
        else:
            p_in = _num(item, "pricing", "price_1m_input_tokens")
            p_out = _num(item, "pricing", "price_1m_output_tokens")
            if per_task is None or not p_in or not p_out:
                unmatched.append((label, "Need per-task export or positive task cost and input/output prices"))
                continue
            scenarios = tuple(
                ((1 - s) * per_task / (p_in / 1e6), s * per_task / (p_out / 1e6))
                for s in (output_share.lo, output_share.hi)
            )
            basis, source = "cost-inferred", "artificial_analysis:cost-inferred"
            note += (
                f"; output cost share {output_share.lo:.0%} to {output_share.hi:.0%}; "
                "assumes fresh input and one call/task; benchmark token totals are not tokens/task"
            )
        profiles.append(
            TaskProfile(
                model=spec.name,
                provider=spec.provider,
                scenarios=scenarios,
                cached_tokens=cached,
                requests_per_task=requests,
                label=label,
                score=score,
                cost_per_task=per_task,
                source=source,
                note=note,
                token_basis=basis,
                task_set="Artificial Analysis Intelligence Index",
                index_version=str(version) if version is not None else None,
            )
        )
    return AAProfiles(
        profiles, unmatched, payload.get("tier") or "unknown", str(version) if version is not None else None
    )


def fetch_details(slugs: list[str], api_key: Optional[str] = None, *, index_version=None) -> dict:
    """Fetch selected Pro model details (evaluation counts); never cache on disk."""
    from urllib.parse import quote

    key = api_key or os.environ.get("AA_API_KEY")
    if not key:
        raise AAError("An Artificial Analysis Pro API key is required for model details")
    if len(slugs) > 20:
        raise AAError("Fetch at most 20 selected model details at a time")
    items = []
    version = index_version
    for slug in dict.fromkeys(slugs):
        try:
            payload = _request("/language/models/" + quote(slug, safe=""), key, None)
        except PermissionError:
            raise AAError("Model details require a Pro subscription") from None
        actual = payload.get("intelligence_index_version")
        if version is not None and str(actual) != str(version):
            raise AAError("Index version changed during fetch; reload the model list")
        version = actual
        item = payload.get("data")
        if not isinstance(item, dict):
            raise AAError(f"Invalid model detail response for {slug}")
        items.append(item)
    return {"tier": "pro", "intelligence_index_version": version, "data": items}


def load_mapping(path: Optional[str]) -> Optional[dict[str, str]]:
    """Optional YAML/JSON file mapping Artificial Analysis slugs to ``provider/model`` registry names."""
    if not path:
        return None
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return {str(k): str(v) for k, v in (yaml.safe_load(text) or {}).items()}
    return {str(k): str(v) for k, v in json.loads(text).items()}
