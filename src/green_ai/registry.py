"""Model, provider and electricity mix registries, plus robust model name matching.

Usage logs rarely contain the exact identifier a registry expects. A gateway may
log ``azure/gpt-4o-2024-08-06``, Bedrock logs ``anthropic.claude-sonnet-4-5-20250929-v1:0``
and Vertex logs ``claude-sonnet-4-5@20250929``. :meth:`Registry.match` resolves
these deterministically and reports *how* each name was matched, so that every
number in a report can be traced back to a registry entry.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any, Optional, Union

from .engine import ElectricityMix
from .ranges import Range, Value, hi, lo, parse

# ---------------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    name: str
    architecture: str  # "dense" or "moe"
    total_params_b: Value
    active_params_b: Value
    tps: Optional[float] = None
    ttft: Optional[float] = None
    warnings: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    origin: str = "ecologits"  # "ecologits" or "custom"

    def __post_init__(self) -> None:
        for v in (self.total_params_b, self.active_params_b):
            if lo(v) <= 0 or not math.isfinite(lo(v)) or not math.isfinite(hi(v)):
                raise ValueError("Model parameter counts must be finite and positive")
        if lo(self.active_params_b) > lo(self.total_params_b) or hi(self.active_params_b) > hi(self.total_params_b):
            raise ValueError("Active parameters cannot exceed total parameters")
        if self.tps is not None and (not math.isfinite(self.tps) or self.tps <= 0):
            raise ValueError("Model throughput must be finite and positive")
        if self.ttft is not None and (not math.isfinite(self.ttft) or self.ttft < 0):
            raise ValueError("Time to first token must be finite and non-negative")

    @property
    def key(self) -> tuple[str, str]:
        return self.provider, self.name


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    location: str
    pue: Value
    wue: Value


class MatchStatus(str, Enum):
    EXACT = "exact"
    ALIAS = "alias"
    NORMALIZED = "normalized"
    UNMATCHED = "unmatched"


@dataclass(frozen=True)
class Match:
    query: str
    status: MatchStatus
    model: Optional[ModelSpec] = None
    suggestions: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.model is not None


class ModelNotFoundError(LookupError):
    def __init__(self, match: Match):
        hint = f" Did you mean: {', '.join(match.suggestions)}?" if match.suggestions else ""
        super().__init__(f"Model '{match.query}' is not in the registry.{hint}")
        self.match = match


class ZoneNotFoundError(LookupError):
    pass


# ---------------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------------

PROVIDER_SYNONYMS = {
    "openai": "openai",
    "azure": "openai",
    "azure_openai": "openai",
    "azureopenai": "openai",
    "anthropic": "anthropic",
    "bedrock": "anthropic",
    "claude": "anthropic",
    "google": "google_genai",
    "gemini": "google_genai",
    "vertex": "google_genai",
    "vertex_ai": "google_genai",
    "google_genai": "google_genai",
    "mistral": "mistralai",
    "mistralai": "mistralai",
    "cohere": "cohere",
    "huggingface": "huggingface_hub",
    "hf": "huggingface_hub",
    "huggingface_hub": "huggingface_hub",
}

_DATE_SUFFIX = re.compile(r"(-|@|_)(20\d{2}-?\d{2}-?\d{2}|\d{4})$")
_BEDROCK_VERSION = re.compile(r"-v\d+(:\d+)?$")
_PREFIX = re.compile(r"^(?:[a-z_\-]+[/.])+")  # "openai/", "azure/", "anthropic.", "models/", "us.anthropic."


def canonical_provider(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    key = re.sub(r"[\s\-]+", "_", str(name).strip().lower())
    return PROVIDER_SYNONYMS.get(key, key)


def normalize_name(name: str) -> str:
    """Lowercase, strip routing prefixes, Bedrock versions and ``:latest`` style tags."""
    n = str(name).strip().lower()
    n = n.replace(" ", "-")
    n = re.sub(r":(latest|free|beta|preview)$", "", n)
    n = _BEDROCK_VERSION.sub("", n)
    # Keep HuggingFace style "org/model" names intact when the full name exists;
    # the caller tries both variants.
    return n


def strip_prefix(name: str) -> str:
    return _PREFIX.sub("", name)


def strip_date(name: str) -> str:
    n = name.replace("@", "-")
    return _DATE_SUFFIX.sub("", n)


# ---------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------


def _data_path(name: str) -> Path:
    return Path(str(resources.files("green_ai") / "data" / "ecologits" / name))


class Registry:
    """In-memory registry of models, provider datacenters and electricity mixes."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], ModelSpec] = {}
        self._aliases: dict[tuple[str, str], tuple[str, str]] = {}
        self._providers: dict[str, ProviderSpec] = {}
        self._zones: dict[str, ElectricityMix] = {}
        self._index: Optional[dict[str, list[tuple[str, str]]]] = None
        self.data_version: str = "custom"
        self.data_source: str = "custom"

    # loading ------------------------------------------------------------------

    @classmethod
    def default(cls) -> Registry:
        """Registry with the newest available reference data.

        Uses the data cached by ``green-ai update`` when it is newer than the
        copy bundled with the package, and falls back to the bundled copy if the
        cache cannot be read.
        """
        from .update import bundled_version, cached_data_dir

        cache = cached_data_dir()
        if cache is not None:
            try:
                reg = cls._from_dir(cache)
                reg.data_version = (cache / "VERSION").read_text(encoding="utf-8").strip()
                reg.data_source = "cache"
                return reg
            except (OSError, ValueError, KeyError, TypeError):
                pass
        reg = cls._from_dir(_data_path(""))
        reg.data_version = bundled_version()
        reg.data_source = "bundled"
        return reg

    @classmethod
    def _from_dir(cls, directory: Path) -> Registry:
        reg = cls()
        reg.load_ecologits_models(Path(directory) / "models.json")
        reg.load_providers(_data_path("providers.json"))
        reg.load_zones(Path(directory) / "electricity_mixes.json")
        reg._ensure_providers()
        return reg

    def _ensure_providers(self) -> None:
        """Give providers that appear in newer data but have no datacenter profile a neutral default."""
        for prov in {p for p, _ in self._models}:
            if prov not in self._providers:
                self.add_provider(prov, location="WOR", pue=Range(1.1, 1.5), wue=Range(0.0, 1.0))

    def load_ecologits_models(self, path: Union[str, Path]) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for m in data.get("models") or []:
            self.add_model(_spec_from_ecologits(m))
        for a in data.get("aliases") or []:
            self._aliases[(a["provider"], a["name"])] = (a["provider"], a["alias"])
        self._index = None

    def load_providers(self, path: Union[str, Path]) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for name, p in data["providers"].items():
            self._providers[name] = ProviderSpec(name, p["datacenter_location"], parse(p["pue"]), parse(p["wue"]))

    def load_zones(self, path: Union[str, Path]) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for z in data["electricity_mixes"]:
            self._zones[z["name"]] = ElectricityMix(
                zone=z["name"],
                gwp=float(z["gwp"]),
                adpe=float(z["adpe"]),
                pe=float(z["pe"]),
                wue=float(z["wue"]),
                warnings=tuple(z.get("warnings") or ()),
            )

    def load_custom(self, path: Union[str, Path]) -> list[ModelSpec]:
        """Load custom models and providers from a YAML or JSON file.

        Example (YAML)::

            providers:
              onprem:
                location: FRA
                pue: {min: 1.2, max: 1.5}
                wue: 0.0
            models:
              - name: my-moe-model
                provider: onprem
                total_params: 284
                active_params: 13
                source: https://huggingface.co/...
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        return self.load_custom_data(data)

    def load_custom_data(self, data: dict) -> list[ModelSpec]:
        """Load an in-memory configuration into this registry only."""
        if not isinstance(data, dict):
            raise ValueError("Custom configuration must be a mapping")
        for name, p in (data.get("providers") or {}).items():
            self.add_provider(
                name, location=p.get("location", "WOR"), pue=parse(p.get("pue", 1.0)), wue=parse(p.get("wue", 0.0))
            )
        added = []
        for m in data.get("models") or []:
            added.append(
                self.register(
                    m["name"],
                    provider=m.get("provider", "custom"),
                    total_params=parse(m["total_params"]),
                    active_params=parse(m["active_params"]) if m.get("active_params") is not None else None,
                    tps=m.get("tps"),
                    ttft=m.get("ttft"),
                    source=m.get("source"),
                    replace=m.get("replace", True),
                )
            )
        return added

    # mutation -----------------------------------------------------------------

    def add_model(self, spec: ModelSpec, replace: bool = False) -> None:
        if spec.key in self._models and not replace:
            raise ValueError(f"Duplicate model {spec.key}")
        self._models[spec.key] = spec
        self._index = None

    def register(
        self,
        name: str,
        *,
        total_params: Value,
        active_params: Optional[Value] = None,
        provider: str = "custom",
        tps: Optional[float] = None,
        ttft: Optional[float] = None,
        source: Optional[str] = None,
        replace: bool = False,
    ) -> ModelSpec:
        """Register a model that is missing from the reference registry.

        ``total_params`` and ``active_params`` are in billions. Omit ``active_params``
        for a dense model. Either can be a :class:`Range` for unknown architectures.
        """
        provider = canonical_provider(provider) or "custom"
        active = total_params if active_params is None else active_params
        spec = ModelSpec(
            provider=provider,
            name=name,
            architecture="dense" if active_params is None else "moe",
            total_params_b=total_params,
            active_params_b=active,
            tps=tps,
            ttft=ttft,
            sources=(source,) if source else (),
            origin="custom",
        )
        self.add_model(spec, replace=replace)
        if provider not in self._providers:
            # Unknown datacenter: world average mix and a neutral PUE range.
            self.add_provider(provider, location="WOR", pue=Range(1.1, 1.5), wue=Range(0.0, 1.0))
        return spec

    def add_provider(self, name: str, *, location: str, pue: Value, wue: Value) -> ProviderSpec:
        spec = ProviderSpec(name, location.upper(), pue, wue)
        self._providers[name] = spec
        return spec

    def add_zone(
        self,
        zone: str,
        *,
        gwp: float,
        adpe: Optional[float] = None,
        pe: Optional[float] = None,
        wue: Optional[float] = None,
    ) -> ElectricityMix:
        """Add or override an electricity mix. Missing factors default to the world average."""
        world = self._zones.get("WOR")
        mix = ElectricityMix(
            zone=zone.upper(),
            gwp=gwp,
            adpe=adpe if adpe is not None else world.adpe,
            pe=pe if pe is not None else world.pe,
            wue=wue if wue is not None else world.wue,
        )
        self._zones[mix.zone] = mix
        return mix

    # queries ------------------------------------------------------------------

    @property
    def models(self) -> list[ModelSpec]:
        return list(self._models.values())

    @property
    def identifiers(self) -> list[str]:
        return sorted(f"{p}/{n}" for p, n in set(self._models) | set(self._aliases))

    @property
    def providers(self) -> dict[str, ProviderSpec]:
        return dict(self._providers)

    @property
    def zones(self) -> dict[str, ElectricityMix]:
        return dict(self._zones)

    def provider(self, name: str) -> ProviderSpec:
        key = canonical_provider(name)
        if key not in self._providers:
            raise LookupError(f"Unknown provider '{name}'. Known: {', '.join(sorted(self._providers))}")
        return self._providers[key]

    def zone(self, code: str) -> ElectricityMix:
        key = str(code).strip().upper()
        if key not in self._zones:
            raise ZoneNotFoundError(
                f"Unknown electricity mix zone '{code}'. Use an ISO 3166-1 alpha-3 code "
                f"(FRA, USA, DEU...) or WOR for the world average."
            )
        return self._zones[key]

    def get(self, name: str, provider: Optional[str] = None) -> ModelSpec:
        m = self.match(name, provider)
        if not m.ok:
            raise ModelNotFoundError(m)
        return m.model  # type: ignore[return-value]

    def _build_index(self) -> dict[str, list[tuple[str, str]]]:
        idx: dict[str, list[tuple[str, str]]] = {}

        def put(k: str, target: tuple[str, str]) -> None:
            lst = idx.setdefault(k, [])
            if target not in lst:
                lst.append(target)

        for key, spec in self._models.items():
            n = spec.name.lower()
            for variant in {n, strip_prefix(n), strip_date(n), strip_date(strip_prefix(n))}:
                put(variant, key)
        for (_prov, name), target in self._aliases.items():
            n = name.lower()
            for variant in {n, strip_date(n)}:
                put(variant, target)
        self._index = idx
        return idx

    def match(self, name: str, provider: Optional[str] = None) -> Match:
        """Resolve a raw model identifier to a registry entry.

        Resolution order: exact name, declared alias, then normalised variants
        (routing prefix removed, Bedrock version removed, release date removed).
        A normalised match is only accepted when it is unambiguous for the
        provider; fuzzy similarity is used for suggestions, never silently.
        """
        query = str(name)
        prov = canonical_provider(provider)
        providers = [prov] if prov else sorted({p for p, _ in self._models})

        for p in providers:
            if (p, query) in self._models:
                return Match(query, MatchStatus.EXACT, self._models[(p, query)])
        for p in providers:
            if (p, query) in self._aliases:
                return Match(query, MatchStatus.ALIAS, self._models[self._aliases[(p, query)]])

        idx = self._index or self._build_index()
        n = normalize_name(query)
        candidates_names = [n, strip_prefix(n), strip_date(n), strip_date(strip_prefix(n))]
        for cand in candidates_names:
            keys = [k for k in idx.get(cand, []) if prov is None or k[0] == prov]
            # Prefer the undated, canonical entry when several dated snapshots collapse together.
            exact_name = [k for k in keys if k[1].lower() == cand]
            if len(exact_name) == 1:
                return Match(query, MatchStatus.NORMALIZED, self._models[exact_name[0]])
            archs = {(self._models[k].total_params_b, self._models[k].active_params_b) for k in keys}
            if keys and len(archs) == 1:
                # Several dated snapshots share one architecture: use the most recent one.
                best = sorted(keys, key=lambda k: k[1])[-1]
                return Match(query, MatchStatus.NORMALIZED, self._models[best])

        pool = [k[1] for k in self._models if prov is None or k[0] == prov]
        suggestions = tuple(difflib.get_close_matches(strip_date(strip_prefix(n)), pool, n=3, cutoff=0.6))
        return Match(query, MatchStatus.UNMATCHED, None, suggestions)

    def search(self, text: str = "", provider: Optional[str] = None) -> list[ModelSpec]:
        prov = canonical_provider(provider)
        t = text.lower()
        return sorted(
            (m for m in self._models.values() if (prov is None or m.provider == prov) and t in m.name.lower()),
            key=lambda m: (m.provider, m.name),
        )


def _spec_from_ecologits(m: dict[str, Any]) -> ModelSpec:
    arch = m["architecture"]
    params = arch["parameters"]
    if isinstance(params, dict) and "total" in params:
        total, active = parse(params["total"]), parse(params["active"])
    else:
        total = active = parse(params)
    dep = m.get("deployment") or {}
    return ModelSpec(
        provider=m["provider"],
        name=m["name"],
        architecture=arch["type"],
        total_params_b=total,
        active_params_b=active,
        tps=dep.get("tps"),
        ttft=dep.get("ttft"),
        warnings=tuple(m.get("warnings") or ()),
        sources=tuple(m.get("sources") or ()),
    )


_DEFAULT: Optional[Registry] = None


def default_registry() -> Registry:
    """Process-wide registry, loaded lazily. Mutations (``register``) are shared."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Registry.default()
    return _DEFAULT


def iter_models(providers: Optional[Iterable[str]] = None) -> list[ModelSpec]:
    reg = default_registry()
    if providers is None:
        return reg.models
    wanted = {canonical_provider(p) for p in providers}
    return [m for m in reg.models if m.provider in wanted]
