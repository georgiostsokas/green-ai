# Vendored reference data

The files in this folder come from [EcoLogits](https://github.com/genai-impact/ecologits)
(GenAI Impact), version 0.11.3, and are distributed under the
[Mozilla Public License 2.0](LICENSE-MPL-2.0). They are kept unmodified so that the
base model gives the same results as the EcoLogits reference code.

| File | Content |
|---|---|
| `models.json` | Model registry: architecture (dense or MoE), parameter counts or ranges, tokens per second, time to first token, warnings, sources |
| `electricity_mixes.json` | Per-country electricity impact factors (GWP, ADPe, PE, WUE) |
| `providers.json` | Datacenter location, PUE and WUE per provider (transcribed from EcoLogits source) |

To refresh them from a newer EcoLogits release:

```bash
pip install -U ecologits
python scripts/sync_ecologits_data.py
pytest -m parity
```

Everything outside this folder is original code released under the MIT license.
