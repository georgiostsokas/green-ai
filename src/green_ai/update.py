"""Keep the model registry current without waiting for a Green AI release.

New models are added to the open EcoLogits dataset regularly. ``green-ai update``
looks up the latest EcoLogits release on PyPI, downloads that release's model and
electricity mix files from GitHub, validates them, and stores them in a user cache.
The registry then prefers the cached data whenever it is newer than the bundled copy.

Set ``GREEN_AI_USE_BUNDLED=1`` to ignore the cache (the test suite does this).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Optional

PYPI_URL = "https://pypi.org/pypi/ecologits/json"
RAW_URL = "https://raw.githubusercontent.com/genai-impact/ecologits/{ref}/ecologits/data/{name}"
DATA_FILES = ("models.json", "electricity_mixes.json")
TIMEOUT = 20


def bundled_dir() -> Path:
    return Path(str(resources.files("green_ai") / "data" / "ecologits"))


def bundled_version() -> str:
    return (bundled_dir() / "VERSION").read_text(encoding="utf-8").strip()


def cache_root() -> Path:
    env = os.environ.get("GREEN_AI_CACHE_DIR")
    if env:
        return Path(env)
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "green-ai"


def registry_cache_dir() -> Path:
    return cache_root() / "registry"


def parse_version(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def cached_data_dir() -> Optional[Path]:
    """Directory of cached registry data, if present, valid and newer than the bundled copy."""
    if os.environ.get("GREEN_AI_USE_BUNDLED", "").lower() in {"1", "true", "yes"}:
        return None
    d = registry_cache_dir()
    pointer = d / "CURRENT"
    if pointer.exists():
        name = pointer.read_text(encoding="utf-8").strip()
        if not name or Path(name).name != name or name.startswith("."):
            return None
        d = d / name
    vf = d / "VERSION"
    if not vf.exists() or not all((d / n).exists() for n in DATA_FILES):
        return None
    try:
        if parse_version(vf.read_text(encoding="utf-8").strip()) <= parse_version(bundled_version()):
            return None
    except (ValueError, OSError):
        return None
    return d


def active_version() -> tuple[str, str]:
    """(version, source) of the registry data the default registry will load."""
    d = cached_data_dir()
    if d is not None:
        return (d / "VERSION").read_text(encoding="utf-8").strip(), "cache"
    return bundled_version(), "bundled"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "green-ai"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:  # noqa: S310 - fixed https URLs
        return r.read()


def latest_version() -> str:
    return json.loads(_get(PYPI_URL))["info"]["version"]


@dataclass(frozen=True)
class UpdateResult:
    previous: str
    current: str
    source: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: bool

    def __str__(self) -> str:
        if not self.changed:
            return f"Registry is up to date (EcoLogits data {self.current}, {self.source})."
        lines = [f"Registry updated: EcoLogits data {self.previous} -> {self.current}."]
        if self.added:
            lines.append(
                f"{len(self.added)} new models: "
                + ", ".join(self.added[:20])
                + (" ..." if len(self.added) > 20 else "")
            )
        if self.removed:
            lines.append(f"{len(self.removed)} models no longer listed: " + ", ".join(self.removed[:10]))
        return "\n".join(lines)


def _model_keys(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {f"{m['provider']}/{m['name']}" for m in data.get("models") or []}


def _check_reference_compatibility(version: str) -> None:
    """Do not silently combine future equations/provider assumptions with an old engine."""
    manifest = json.loads((bundled_dir() / "engine_reference.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        url = f"https://raw.githubusercontent.com/genai-impact/ecologits/{version}/ecologits/{name}"
        if hashlib.sha256(_get(url)).hexdigest() != expected:
            raise ValueError(
                "Reference equations or provider assumptions changed. "
                "Install a reviewed Green AI release before using this registry."
            )


def update(version: Optional[str] = None, force: bool = False) -> UpdateResult:
    """Download the registry data of an EcoLogits release (latest by default) into the cache."""
    from .registry import Registry

    previous, prev_source = active_version()
    target = version or latest_version()
    if not re.fullmatch(r"\d+\.\d+\.\d+[a-zA-Z0-9.+-]*", target):
        raise ValueError("Invalid reference version")
    if not force and parse_version(target) <= parse_version(previous):
        return UpdateResult(previous, previous, prev_source, (), (), False)
    _check_reference_compatibility(target)

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        for name in DATA_FILES:
            (tmpd / name).write_bytes(_get(RAW_URL.format(ref=target, name=name)))
        # Validate before replacing anything: the files must load into a working registry.
        reg = Registry()
        reg.load_ecologits_models(tmpd / "models.json")
        reg.load_zones(tmpd / "electricity_mixes.json")
        if len(reg.models) < 50 or "WOR" not in reg.zones:
            raise ValueError(f"Downloaded data for EcoLogits {target} looks incomplete, cache left unchanged.")

        old_dir = cached_data_dir() or bundled_dir()
        old_keys = _model_keys(old_dir / "models.json")
        new_keys = _model_keys(tmpd / "models.json")

        root = registry_cache_dir()
        root.mkdir(parents=True, exist_ok=True)
        dest = Path(tempfile.mkdtemp(prefix=f"{target}-", dir=root))
        for name in DATA_FILES:
            shutil.copyfile(tmpd / name, dest / name)
        (dest / "VERSION").write_text(target + "\n", encoding="utf-8")
        # Readers see the complete old or complete new snapshot, never a partial copy.
        pending = root / "CURRENT.pending"
        pending.write_text(dest.name, encoding="utf-8")
        pending.replace(root / "CURRENT")

    return UpdateResult(
        previous, target, "cache", tuple(sorted(new_keys - old_keys)), tuple(sorted(old_keys - new_keys)), True
    )


def clear_cache() -> bool:
    d = registry_cache_dir()
    if d.exists():
        shutil.rmtree(d)
        return True
    return False
