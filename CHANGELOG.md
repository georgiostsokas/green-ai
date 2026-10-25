# Changelog

## 0.2.0 (2026-10-23)

### Added
- Dashboard with request explorer, task rankings, usage analytics, model library
  and methodology pages, plus double-click launchers for macOS and Windows.
- Ranking by gCO2eq, Wh or mL of water per task, or score per Wh, with a minimum
  score filter, Pareto choices and the per token rank for comparison.
- Calls per task in task profiles, for multi step agents.
- Import of Artificial Analysis per task exports, output only exports with an
  input ratio range, and Pro evaluation counts with a versioned benchmark plan.
- Research aggregates and `scripts/reproduce_evidence.py` for the alpha presets.
- Hosted demo support: Artificial Analysis key from Streamlit secrets.
- CI on Python 3.10 to 3.14 and Windows/macOS, plus a clean wheel install check.

### Changed
- The input correction applies to GPU energy only, with PUE applied once. Host
  energy and embodied allocation stay in the base model.
- Artificial Analysis API profiles are derived from cost per task and labelled
  `cost-inferred`. Aggregate token counts are no longer divided by
  `total_cost / cost_per_task`, which is not a task count.
- Registry updates are stored as snapshots and switched atomically, and are
  refused when the upstream equations change.

### Fixed
- Invalid, negative or non-finite counts in logs now stop the analysis instead
  of becoming zero.
- The tracker infers the provider from the model name and rejects responses
  without usage.
- Replacing a custom model invalidates cached per token impacts.
- Dashboard pages share the sidebar input correction setting.
- Uploads are processed in memory, and exports omit unrelated log columns and
  escape spreadsheet formulas.

## 0.1.0 (2026-10-22)

First working version: base model with reference agreement tests, input
correction with low and central presets, name matching for gateway style model
names, custom models from YAML, vectorised usage log engine, Excel and chart
reports, response tracker, task profiles, registry updates and CLI.
