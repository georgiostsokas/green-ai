# Contributing

Thanks for your interest. The most valuable contributions are:

1. **Measured data.** Prefill and decode energy per token at production batch
   sizes, on recent accelerators, to refine `alpha`.
2. **Model parameters.** Disclosed sizes of new models, with a source link.
   Models in the reference registry should be contributed upstream to
   [EcoLogits](https://github.com/genai-impact/ecologits), then synced here with
   `python scripts/sync_ecologits_data.py`.
3. **Log formats.** Column names used by LLM gateways and observability tools,
   added to `SYNONYMS` in `src/green_ai/batch.py`.

## Development

```bash
# In your fork or local checkout:
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app]"
pytest                    # full suite, including the EcoLogits consistency tests
ruff check .
ruff format --check .
python scripts/make_figures.py
```

Any change to `engine.py` must keep `pytest -m parity` green. Changes to the
input correction need a test in `tests/test_prefill.py` and an update to
`docs/METHODOLOGY.md`. The dashboard shows a packaged copy of that file, so copy
it to `src/green_ai/data/methodology.md` as well (a test checks they match).

Design choices and the reasons behind them are in `docs/DESIGN.md`.
