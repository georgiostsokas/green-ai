# Design notes

How the code is organised and why. The physics and coefficients are in
[METHODOLOGY.md](METHODOLOGY.md).

## Layout

```
src/green_ai/
  ranges.py      interval arithmetic for uncertain inputs
  engine.py      base model and input correction, pure functions
  registry.py    models, providers, grid zones, name matching, custom YAML
  update.py      registry refresh from upstream releases, local snapshots
  prefill.py     alpha and cache factor presets
  estimator.py   public API: Estimator, estimate(), Estimate
  batch.py       vectorised estimation over usage logs
  tasks.py       task profiles, per task footprint, rankings
  aa.py          Artificial Analysis client and adapters
  tracker.py     token usage from OpenAI, Anthropic, Mistral and Gemini responses
  report.py      Excel, charts, Markdown
  cli.py         command line
  dashboard.py   Streamlit app
```

Data flows one way: `registry` and `engine` know nothing about logs, tasks or
the UI. `estimator` combines them, and everything else (`batch`, `tasks`,
`tracker`, `cli`, `dashboard`) goes through `estimator`.

## Decisions

**Keep the base model separate from the correction.** Every `Estimate` carries
`baseline` and `prefill` impacts separately, and `total` is their sum. The base
model can then be checked number for number against the EcoLogits reference
code, which would be impossible if the correction were folded in. It also lets
users switch the correction off or compare presets without a second code path.

**Intervals, not Monte Carlo.** Undisclosed model sizes and datacenter figures
are ranges. All formulas are increasing in their inputs, so evaluating the low
and high ends gives exact bounds at twice the cost of a point estimate. Sampling
would add noise and run time and would need distributions the sources do not
provide.

**Linearity drives the log engine.** Impacts are linear in tokens and requests.
`Estimator.units()` and `Estimator.prefill_unit()` compute, once per model and
deployment, the impact of one output token, one request and one prompt token
equivalent, and cache them. A usage log
is then a matrix product: a million rows take about two seconds, and aggregated
logs (one row per model and month) give the same totals as per request logs.
The cache key includes the full model spec, so replacing a custom model
invalidates it.

**Correction scoped to GPU energy.** `alpha` is a ratio of GPU energy per phase.
Applying it to the full marginal decode impact, which includes host server time
and embodied allocation, would extend a measurement beyond what it measured. An
earlier version did that. The current scope is narrower and testable: changing
the host server power must not change the correction, and a test checks it.

**No monkey patching.** The tracker reads the `usage` object of a response the
caller already has. Patching SDK clients is fragile across SDK versions and
surprising in production code. The cost is one line per call.

**Name matching is explicit.** Logs contain `azure/gpt-4o-2024-08-06`,
`anthropic.claude-...-v1:0` or `claude-...@2025...`. The matcher tries exact
names, declared aliases, then normalised variants, and reports which one it
used. Fuzzy matching is only used for suggestions, never silently, because a
wrong match produces a confident wrong number.

**Unknown is not zero.** Unmatched models stay in the coverage report and are
excluded from totals. Invalid counts stop the analysis. A footprint that quietly
drops 20% of usage looks precise and is not.

**Every profile says where its tokens came from.** Artificial Analysis publishes cost per
task as a weighted average. Its ratio to total cost is not a task count, so token
counts are never divided by it. Profiles carry a `token_basis`
(`observed`, `output-observed/input-assumed`, `benchmark-weighted`,
`cost-inferred`, `synthetic`) that follows them into every export.

**Registry updates as snapshots.** `green-ai update` downloads new model data
into a fresh directory, validates it, then switches a pointer file in one atomic
rename, so a reader sees either the old or the new snapshot. Before that it
compares the upstream equation files with stored hashes and refuses data from a
release whose equations changed, because new data with old equations would be
silently wrong. A weekly workflow does the same for the bundled copy and opens a
pull request only after the full test suite passes.

**Session isolation in the dashboard.** Each session builds its own registry,
uploads are read from memory, and API responses stay in session state. A hosted
copy can serve several users without one user's custom models or data reaching
another.

## Testing approach

* Reference agreement for the base model over every registered model.
* Property style checks for the correction (linearity, scope, ordering).
* Equivalence checks: vectorised log engine against the scalar estimator, task
  footprint against a direct estimate, scenario tables against full reruns.
* Regression tests for each bug found in review.
* The real Streamlit app executed through `AppTest`, without a browser.

Details: [VALIDATION.md](VALIDATION.md).
