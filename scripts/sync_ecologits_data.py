"""Refresh the vendored reference data from the installed EcoLogits package.

pip install -U ecologits
python scripts/sync_ecologits_data.py
pytest -m parity
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "src" / "green_ai" / "data" / "ecologits"


def main() -> None:
    import ecologits
    from ecologits.tracers.utils import PROVIDER_CONFIG_MAP
    from ecologits.utils.range_value import RangeValue

    src = Path(ecologits.__file__).parent / "data"
    for name in ("models.json", "electricity_mixes.json"):
        shutil.copyfile(src / name, DEST / name)

    def val(v):
        return {"min": v.min, "max": v.max} if isinstance(v, RangeValue) else v

    providers = {
        name: {
            "datacenter_location": cfg.datacenter_location,
            "pue": val(cfg.datacenter_pue),
            "wue": val(cfg.datacenter_wue),
        }
        for name, cfg in PROVIDER_CONFIG_MAP.items()
    }
    version = getattr(ecologits, "__version__", None)
    if version is None:
        from importlib.metadata import version as _v

        version = _v("ecologits")
    doc = {
        "_comment": f"Provider datacenter assumptions reproduced from EcoLogits {version} "
        "(ecologits/tracers/utils.py, PROVIDER_CONFIG_MAP). Distributed under MPL-2.0, see LICENSE-MPL-2.0.",
        "providers": providers,
    }
    (DEST / "providers.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    (DEST / "VERSION").write_text(f"{version}\n", encoding="utf-8")
    manifest = {
        "reference": f"EcoLogits {version}",
        "sha256": {
            name: hashlib.sha256((Path(ecologits.__file__).parent / name).read_bytes()).hexdigest()
            for name in ("impacts/llm.py", "tracers/utils.py")
        },
    }
    (DEST / "engine_reference.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    n = len(json.loads((DEST / "models.json").read_text(encoding="utf-8"))["models"])
    print(f"Synced EcoLogits {version}: {n} models, provider configs and electricity mixes -> {DEST}")
    print("Now run: pytest -m parity")


if __name__ == "__main__":
    main()
