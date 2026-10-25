"""Generate the synthetic usage log shipped in examples/sample_usage.csv.

The data is entirely made up. It mimics the shape of a company LLM gateway log:
daily totals per model, a growing adoption curve, prompt heavy workloads (RAG,
agents, document analysis), partial prompt caching, gateway style model names
and one self hosted model that is not in the public registry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)

# model, provider, share of traffic, input/output ratio, cache hit rate, tokens per request out
WORKLOADS = [
    ("gpt-4o", "openai", 0.22, 35, 0.30, 450),
    ("azure/gpt-4o-mini-2024-07-18", "azure", 0.18, 60, 0.20, 300),
    ("claude-sonnet-4-5-20250929", "anthropic", 0.16, 110, 0.55, 650),
    ("anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock", 0.10, 140, 0.40, 250),
    ("gemini-2.5-flash", "google", 0.12, 90, 0.10, 400),
    ("mistral-large-latest", "mistral", 0.08, 25, 0.0, 500),
    ("gpt-5", "openai", 0.09, 45, 0.35, 1800),
    ("acme-internal-qwen", "onprem", 0.05, 30, 0.0, 350),
]


def main(out: Path) -> None:
    days = pd.date_range("2026-01-01", "2026-06-30", freq="D")
    t = np.linspace(0, 1, len(days))
    daily_requests = 4_000 * np.exp(2.2 * t)  # roughly 9x growth over six months
    weekday = np.where(days.dayofweek < 5, 1.0, 0.35)
    rows = []
    for i, day in enumerate(days):
        for model, provider, share, io, cache, out_per_req in WORKLOADS:
            # prompts get longer over time as agents and RAG take over
            io_t = io * (0.7 + 0.6 * t[i])
            req = max(1, int(daily_requests[i] * weekday[i] * share * RNG.lognormal(0, 0.15)))
            out_tok = int(req * out_per_req * RNG.lognormal(0, 0.1))
            in_tok = int(out_tok * io_t * RNG.lognormal(0, 0.1))
            cached = int(in_tok * min(0.9, cache * (0.6 + 0.8 * t[i])))
            rows.append((day.date().isoformat(), provider, model, req, in_tok, cached, out_tok))
    df = pd.DataFrame(
        rows, columns=["date", "provider", "model", "requests", "prompt_tokens", "cached_tokens", "completion_tokens"]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "examples" / "sample_usage.csv")
