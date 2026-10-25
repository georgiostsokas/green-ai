# Validation

What is tested, how, and what testing cannot show. Everything below runs in CI
on Linux, macOS and Windows with Python 3.10 to 3.14.

```bash
pip install -e ".[dev]"
pytest                                # full suite
python scripts/validation_report.py   # base model against EcoLogits, full grid
python scripts/reproduce_evidence.py  # alpha presets from the bundled aggregates
```

## 1. Base model against the reference implementation

`tests/test_parity.py` compares the base model with the EcoLogits 0.11.3
reference code for every registered model, at several output lengths and
electricity zones, with and without a measured latency cap, and on the
regression path used when a model has no throughput data.

`scripts/validation_report.py` runs the wider grid: 351 model identifiers × 7
output lengths × 6 zone selections, lower and upper bound of 6 impact quantities,
**176,904 comparisons**.

| Quantity | Worst relative difference |
|---|---:|
| Electricity | 4.1e-16 |
| GHG total | 5.1e-16 |
| GHG embodied | 3.9e-16 |
| ADPe | 4.6e-16 |
| Primary energy | 5.9e-16 |
| Water | 4.9e-16 |

The differences are floating point rounding. This shows the implementation is
faithful to the reference method. It does not show the method is accurate.

When a newer EcoLogits release changes the reference equations or provider
assumptions, `green-ai update` refuses the new data (the source files are
checked against stored hashes) until a reviewed release adapts the base model.

## 2. Input correction

`tests/test_prefill.py` and `tests/test_regressions.py` check that:

* with no prompt tokens, or `alpha = 0`, results equal the base model,
* the correction is linear in prompt tokens and cached tokens cost `cache_factor`
  times a fresh token,
* PUE is applied once, and host energy and embodied impacts are unchanged
  (a larger host server changes the base model but not the correction),
* GHG, water, primary energy and ADPe follow from the added energy with the same
  grid factors,
* presets are ordered: `none` < `low` < `central`.

`scripts/reproduce_evidence.py` recomputes both presets from the bundled
research aggregates and asserts the stored values.

## 3. Usage logs

`tests/test_batch.py` checks that the vectorised log engine returns the same
value as the single request estimator for every row (relative tolerance 1e-12),
that monthly, per model and per provider totals add up, and that the scenario
table equals full reruns. Regression tests cover malformed, negative and
non-finite counts, cached counts above input, duplicate DataFrame indices,
unknown models (reported, never counted as zero), export column selection and
spreadsheet formula injection.

## 4. Tasks and Artificial Analysis data

`tests/test_tasks_aa.py` and `tests/test_regressions.py` cover calls per task,
ranking by energy, carbon, water and score per Wh, minimum score filters, ties
and exported ranges. A two benchmark counterexample checks that weighted tokens per task are computed from
the plan, not from `total_cost / cost_per_task`. Plans with missing benchmarks,
weights that do not sum to one, or a different index version are rejected.

Artificial Analysis responses are mocked in the documented response shapes. The
tests cover the fallback from Pro to Free, key and tier specific caching, the
in memory dashboard path and the `cost-inferred` labels. No real key is used in
CI and no benchmark snapshot is stored in the repository.

## 5. Dashboard and packaging

`tests/test_dashboard.py` runs the real Streamlit app through `AppTest`: all five
pages, every registered model offered in the request explorer, sidebar settings
reaching every page, a zero carbon grid override, an API shaped AA response and
CSV provenance. CI also builds the wheel and source distribution, installs the wheel in a clean
environment and runs the CLI and the bundled examples from it.

## 6. What testing cannot show

Agreement with the reference code and internal consistency do not establish
physical accuracy. That would need metered energy from real deployments,
particularly prefill and decode energy at production batch sizes. Contributions
of such data are welcome.
