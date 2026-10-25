# Artificial Analysis: tokens, quality and CO2 per task

The Task rankings page supports three AA paths. All footprints are model-based estimates, not AA energy measurements.

## 1. Import actual tokens per task

Choose **Task rankings → AA token export** and upload your account's CSV export. Map its columns in the app. Accepted common headers include `Model`, `Intelligence Index`, `Input Tokens Per Task`, `Output Tokens Per Task`, `Answer Tokens Per Task`, `Reasoning Tokens Per Task` and `Cost Per Task`.

- Input includes cached reads. Add a cached-tokens column when available.
- Output includes reasoning by default. Uncheck that option only if the output column excludes reasoning, or supply answer and reasoning separately.
- Add `index_version` to keep comparisons tied to a benchmark version.
- `requests_per_task` defaults to one. Supply the mean call count for multi-call tasks; token counts are totals across the task, not per-call counts.
- If only output tokens/task are available, select an input/output ratio interval. The app marks input tokens as assumed and carries the interval into energy/CO2 results.

Then choose **CO2 / task**, set a minimum quality score, inspect ranges and the quality/impact scatter, and download the full ranking. Grades and Pareto flags use midpoints; they are not certifications or significance tests.

```bash
green-ai tasks --csv my_aa_export.csv --aa-export --by carbon --min-score 50 --export ranking.csv
green-ai tasks --csv output_only.csv --aa-export --input-ratio 10:50 --by carbon
```

## 2. API scenarios

Choose **AA API**, enter your own key, and click **Load AA model data**. Free and Pro model lists are supported. The dashboard uses session memory and does not write your key or API data to disk. Reload snapshots older than 24 hours.

The list's aggregate benchmark tokens are not weighted tokens/task. Without an explicit plan the app infers scenarios from cost per task and input/output prices, using the selected output-cost-share interval. These rows are always marked `cost-inferred`, including Pro responses. The assumptions include fresh input, unchanged list prices and one call/task.

```bash
export AA_API_KEY='your-own-key'
green-ai tasks --aa --by carbon --output-share 0.5:0.9 --export ranking.csv
```

The CLI uses an optional key/tier-specific 24-hour local cache. No credentials or live AA data are bundled in this project.

## 3. Evaluation counts and an explicit benchmark plan

Pro detail responses can include `evaluation_token_counts`. To convert them correctly, upload a JSON plan with the same `index_version` as the API response. Select the models to fetch (up to twenty). A plan has this structure:

```json
{
  "index_version": "the actual version from your AA response",
  "benchmarks": {
    "first_actual_evaluation_id": {"weight": 0.5, "tasks": 100, "requests_per_task": 1},
    "second_actual_evaluation_id": {"weight": 0.5, "tasks": 20, "requests_per_task": 3}
  }
}
```

These values illustrate the format, not an actual AA index plan. Verify the source weights, included evaluations, task counts and handling of repeats before using a plan. The denominator must match the token totals. The adapter refuses version mismatches, missing evaluations and weights that do not sum to one; it does not normalize an incomplete suite.

Python:

```python
import json
from green_ai import aa
from green_ai.tasks import rank_tasks, save_ranking

listing = aa.fetch(tier="pro")
details = aa.fetch_details(["your-selected-aa-slug"], index_version=listing["intelligence_index_version"])
profiles = aa.to_profiles(details, benchmark_plan=json.load(open("plan.json", encoding="utf-8")))
ranking, skipped = rank_tasks(profiles.profiles, by="carbon", min_score=50)
save_ranking(ranking, "carbon_per_task.csv")
```

AA supplies benchmark scores, token/cost data and the index definition. Green AI supplies environmental estimates using its stated deployment assumptions. Keep the source and version attached when sharing results, and follow your AA account's data terms.
