"""Vectorised estimation over usage logs (CSV, Excel, Parquet or a DataFrame).

Rows can be individual requests or pre-aggregated totals (for example one row per
model and month). Impacts are linear in token counts, so each distinct
(provider, model, zone) is resolved once and every row is computed with array
arithmetic. A million rows take seconds.

Requires the ``report`` extra: ``pip install "green-ai[report]"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from . import prefill as _prefill
from .estimator import Estimator, default_estimator
from .registry import MatchStatus

try:  # pragma: no cover - import guard
    import numpy as np
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError('Batch processing needs pandas: pip install "green-ai[report]"') from exc


SYNONYMS: dict[str, list[str]] = {
    "model": ["model", "model_name", "model_id", "deployment", "llm", "engine"],
    "provider": ["provider", "vendor", "api_provider", "platform"],
    "input_tokens": ["input_tokens", "prompt_tokens", "tokens_in", "input", "n_input_tokens", "prompt"],
    "output_tokens": [
        "output_tokens",
        "completion_tokens",
        "tokens_out",
        "output",
        "n_output_tokens",
        "generated_tokens",
        "completion",
    ],
    "cached_tokens": [
        "cached_tokens",
        "cache_read_tokens",
        "cached_input_tokens",
        "cache_read_input_tokens",
        "prompt_cached_tokens",
    ],
    "requests": ["requests", "n_requests", "request_count", "num_requests", "calls", "count"],
    "date": ["date", "timestamp", "datetime", "day", "month", "created_at", "time", "period"],
    "zone": ["zone", "electricity_zone", "country", "region_iso3"],
}

METRICS = {
    "energy_kwh": "energy_kwh",
    "gwp_kg": "gwp",
    "gwp_usage_kg": "gwp_usage",
    "gwp_embodied_kg": "gwp_embodied",
    "water_l": "water_l",
    "pe_mj": "pe",
    "adpe_kgsbeq": "adpe",
}


@dataclass
class ColumnMap:
    """Names of the columns in the input. ``None`` means absent."""

    model: str = "model"
    output_tokens: str = "output_tokens"
    input_tokens: Optional[str] = "input_tokens"
    cached_tokens: Optional[str] = None
    provider: Optional[str] = None
    requests: Optional[str] = None
    date: Optional[str] = None
    zone: Optional[str] = None

    @classmethod
    def detect(cls, columns: list[str], **overrides: Optional[str]) -> ColumnMap:
        lower = {c.lower().strip(): c for c in columns}
        found: dict[str, Optional[str]] = {}
        for key, names in SYNONYMS.items():
            found[key] = next((lower[n] for n in names if n in lower), None)
        found.update({k: v for k, v in overrides.items() if v is not None})
        if not found.get("model"):
            raise ValueError(f"Could not find a model column in {columns}. Use --map model=<column>.")
        if not found.get("output_tokens"):
            raise ValueError(f"Could not find an output tokens column in {columns}. Use --map output_tokens=<column>.")
        return cls(**found)  # type: ignore[arg-type]


def read_table(path, *, filename: Optional[str] = None) -> pd.DataFrame:
    """Read a path or an in-memory upload. Upload bytes never need a temporary file."""
    suffix = Path(filename or getattr(path, "name", str(path))).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    return pd.read_csv(path, sep=None, engine="python")


@dataclass
class BatchResult:
    rows: pd.DataFrame
    columns: ColumnMap
    prefill_model: _prefill.PrefillModel
    settings: dict[str, Any] = field(default_factory=dict)

    # selections ---------------------------------------------------------------

    @property
    def matched(self) -> pd.DataFrame:
        return self.rows[self.rows["match_status"] != MatchStatus.UNMATCHED.value]

    @property
    def unmatched(self) -> pd.DataFrame:
        u = self.rows[self.rows["match_status"] == MatchStatus.UNMATCHED.value]
        if u.empty:
            return pd.DataFrame(columns=["model", "rows", "output_tokens", "suggestions"])
        return (
            u.groupby(self.columns.model, dropna=False)
            .agg(rows=("match_status", "size"), output_tokens=("_out", "sum"), suggestions=("suggestions", "first"))
            .reset_index()
            .rename(columns={self.columns.model: "model"})
            .sort_values("output_tokens", ascending=False)
        )

    def _value_columns(self) -> list[str]:
        cols = ["_in", "_cached", "_out", "_req"]
        for m in METRICS:
            for part in ("baseline", "prefill", "total"):
                cols.append(f"{part}_{m}")
            cols += [f"total_{m}_min", f"total_{m}_max"]
        return cols

    def _agg(self, by: Union[str, list[str]]) -> pd.DataFrame:
        g = self.matched.groupby(by, dropna=False)[self._value_columns()].sum().reset_index()
        g = g.rename(
            columns={"_in": "input_tokens", "_cached": "cached_tokens", "_out": "output_tokens", "_req": "requests"}
        )
        g["prefill_share"] = (g["prefill_energy_kwh"] / g["total_energy_kwh"]).fillna(0.0)
        g["io_ratio"] = (g["input_tokens"] / g["output_tokens"].replace(0, np.nan)).fillna(0.0)
        return g

    # aggregations -------------------------------------------------------------

    def totals(self) -> dict[str, float]:
        m = self.matched
        out = {
            "rows": int(len(self.rows)),
            "rows_matched": int(len(m)),
            "input_tokens": float(m["_in"].sum()),
            "cached_tokens": float(m["_cached"].sum()),
            "output_tokens": float(m["_out"].sum()),
            "requests": float(m["_req"].sum()),
            "unmatched_output_tokens": float(self.rows.loc[self.rows.index.difference(m.index), "_out"].sum()),
        }
        for k in self._value_columns()[4:]:
            out[k] = float(m[k].sum())
        b, t = out["baseline_energy_kwh"], out["total_energy_kwh"]
        out["prefill_share"] = (t - b) / t if t else 0.0
        out["correction_factor"] = t / b if b else 1.0
        return out

    def by_model(self) -> pd.DataFrame:
        return self._agg(["matched_provider", "matched_model"]).sort_values("total_energy_kwh", ascending=False)

    def by_provider(self) -> pd.DataFrame:
        return self._agg("matched_provider").sort_values("total_energy_kwh", ascending=False)

    def by_month(self) -> pd.DataFrame:
        if "month" not in self.rows:
            raise ValueError("No date column was found, monthly aggregation is unavailable.")
        return self._agg("month").sort_values("month")

    def scenarios(self, names: tuple[str, ...] = ("none", "low", "central")) -> pd.DataFrame:
        """Totals under several prefill assumptions. Exact, since the correction is linear in alpha."""
        m = self.matched
        base_e = float(m["baseline_energy_kwh"].sum())
        base_g = float(m["baseline_gwp_kg"].sum())
        base_w = float(m["baseline_water_l"].sum())
        rows = []
        for n in names:
            pm = _prefill.resolve(n)
            eq = pm.alpha * ((m["_in"] - m["_cached"]) + pm.cached_factor * m["_cached"])
            pref_e = float((eq * m["_unit_energy_kwh"]).sum())
            rows.append(
                {
                    "scenario": pm.name,
                    "alpha": pm.alpha,
                    "baseline_energy_kwh": base_e,
                    "prefill_energy_kwh": pref_e,
                    "total_energy_kwh": base_e + pref_e,
                    "total_gwp_kg": base_g + float((eq * m["_unit_gwp_kg"]).sum()),
                    "total_water_l": base_w + float((eq * m["_unit_water_l"]).sum()),
                }
            )
        df = pd.DataFrame(rows)
        df["vs_baseline"] = df["total_energy_kwh"] / base_e - 1 if base_e else 0.0
        return df

    def export_rows(self, *, include_original: bool = False) -> pd.DataFrame:
        """Normalized analysis fields by default. Original log columns are explicit opt-in."""
        if include_original:
            return self.rows.drop(columns=[c for c in self.rows if c.startswith("_")])
        cols = ["source_row", "match_status", "matched_provider", "matched_model", "suggestions"]
        cols += [c for c in ("month", "electricity_zone") if c in self.rows]
        metrics = [c for c in self.rows if any(c.startswith(p) for p in ("baseline_", "prefill_", "total_"))]
        out = self.rows[cols + metrics].copy()
        for src, name in (
            ("_in", "input_tokens"),
            ("_out", "output_tokens"),
            ("_cached", "cached_tokens"),
            ("_req", "requests"),
        ):
            out[name] = self.rows[src]
        return out


def run(
    data: Union[pd.DataFrame, str, Path],
    *,
    estimator: Optional[Estimator] = None,
    columns: Optional[ColumnMap] = None,
    prefill: _prefill.PrefillLike = "default",
    zone: Optional[str] = None,
    default_provider: Optional[str] = None,
    **column_overrides: Optional[str],
) -> BatchResult:
    """Estimate every row of a usage table.

    Columns are detected automatically from common names (``prompt_tokens``,
    ``completion_tokens``, ``model``...). Pass ``columns=`` or keyword overrides
    such as ``output_tokens="tokens_generated"`` to map others.
    """
    est = estimator or default_estimator()
    df = read_table(data) if not isinstance(data, pd.DataFrame) else data.copy()
    if df.columns.duplicated().any():
        raise ValueError("Log column names must be unique")
    df = df.reset_index(drop=True)
    df["source_row"] = np.arange(1, len(df) + 1)
    cmap = columns or ColumnMap.detect(list(df.columns), **column_overrides)
    pm = _prefill.resolve(est.prefill if prefill == "default" else prefill)

    def col(name: Optional[str], default: float = 0.0) -> pd.Series:
        if name is None:
            return pd.Series(default, index=df.index, dtype="float64")
        if name not in df:
            raise ValueError(f"Mapped column '{name}' is missing")
        value = pd.to_numeric(df[name], errors="coerce").astype("float64")
        bad = ~np.isfinite(value) | (value < 0)
        if bad.any():
            rows = ", ".join(str(x + 1) for x in np.flatnonzero(bad.to_numpy())[:8])
            raise ValueError(f"Column '{name}' needs finite, non-negative numbers. Invalid data rows: {rows}.")
        return value

    df["_out"] = col(cmap.output_tokens)
    df["_in"] = col(cmap.input_tokens)
    df["_cached"] = col(cmap.cached_tokens)
    df["_req"] = col(cmap.requests, 1.0) if cmap.requests else 1.0
    df["_eq"] = pm.alpha * ((df["_in"] - df["_cached"]) + pm.cached_factor * df["_cached"])
    if (df["_cached"] > df["_in"]).any():
        raise ValueError("Cached tokens cannot exceed input tokens (input includes cache reads)")
    if ((df["_req"] == 0) & ((df["_in"] > 0) | (df["_out"] > 0))).any():
        raise ValueError("Rows with non-zero tokens require a positive request count")
    if df[cmap.model].isna().any() or df[cmap.model].astype(str).str.strip().eq("").any():
        raise ValueError("Every row needs a model identifier")

    if cmap.date:
        dates = pd.to_datetime(df[cmap.date], errors="coerce", utc=True)
        df["month"] = dates.dt.tz_convert(None).dt.to_period("M").astype(str)

    prov_series = df[cmap.provider].astype(str) if cmap.provider else pd.Series(default_provider, index=df.index)
    zone_series = df[cmap.zone].astype(str) if cmap.zone else pd.Series(zone, index=df.index)
    keys = pd.DataFrame({"m": df[cmap.model].astype(str), "p": prov_series, "z": zone_series})

    outputs: dict[str, np.ndarray] = {}
    n = len(df)
    for m in METRICS:
        for part in ("baseline", "prefill", "total"):
            outputs[f"{part}_{m}"] = np.full(n, np.nan)
        outputs[f"total_{m}_min"] = np.full(n, np.nan)
        outputs[f"total_{m}_max"] = np.full(n, np.nan)
    for u in ("energy_kwh", "gwp_kg", "water_l"):
        outputs[f"_unit_{u}"] = np.full(n, np.nan)
    status = np.full(n, MatchStatus.UNMATCHED.value, dtype=object)
    matched_model = np.full(n, None, dtype=object)
    matched_provider = np.full(n, None, dtype=object)
    suggestions = np.full(n, "", dtype=object)
    electricity_zone = np.full(n, "", dtype=object)

    for (mname, pname, zname), idx in keys.groupby(["m", "p", "z"], dropna=False).groups.items():
        pos = df.index.get_indexer(idx)
        prov = None if pname in (None, "None", "nan") or pd.isna(pname) else pname
        zn = None if zname in (None, "None", "nan") or pd.isna(zname) else zname
        match = est.registry.match(mname, prov)
        if not match.ok and prov is not None:
            match = est.registry.match(mname, None)  # provider column may hold a gateway name
        if not match.ok:
            suggestions[pos] = ", ".join(match.suggestions)
            continue
        spec = match.model
        _, mix, p, w = est.deployment(spec, zn)
        per_tok, per_req = est.units(spec, mix, p, w)
        pref_unit = est.prefill_unit(spec, mix, p, w)
        pref_f = pref_unit.flat()
        status[pos] = match.status.value
        matched_model[pos] = spec.name
        matched_provider[pos] = spec.provider
        electricity_zone[pos] = mix.zone

        out = df["_out"].to_numpy()[pos]
        req = df["_req"].to_numpy()[pos]
        eq = df["_eq"].to_numpy()[pos]
        tok_f, req_f = per_tok.flat(), per_req.flat()
        for m, attr in METRICS.items():
            for suffix, bound in (("", ""), ("_min", "_min"), ("_max", "_max")):
                t_unit, r_unit = tok_f[attr + bound], req_f[attr + bound]
                base = out * t_unit + req * r_unit
                pref = eq * pref_f[attr + bound]
                if suffix == "":
                    outputs[f"baseline_{m}"][pos] = base
                    outputs[f"prefill_{m}"][pos] = pref
                    outputs[f"total_{m}"][pos] = base + pref
                else:
                    outputs[f"total_{m}{suffix}"][pos] = base + pref
        outputs["_unit_energy_kwh"][pos] = pref_f["energy_kwh"]
        outputs["_unit_gwp_kg"][pos] = pref_f["gwp"]
        outputs["_unit_water_l"][pos] = pref_f["water_l"]

    df["match_status"] = status
    df["matched_provider"] = matched_provider
    df["matched_model"] = matched_model
    df["suggestions"] = suggestions
    df["electricity_zone"] = electricity_zone
    for k, v in outputs.items():
        df[k] = v
    df["prefill_share"] = (df["prefill_energy_kwh"] / df["total_energy_kwh"]).where(df["total_energy_kwh"] > 0)

    settings = {
        "prefill_model": pm.name,
        "alpha": pm.alpha,
        "cached_factor": pm.cached_factor,
        "zone_override": zone or est.zone or "provider default",
        "pue_override": str(est.pue) if est.pue is not None else "provider default",
        "carbon_intensity_override": est.carbon_intensity,
        "wue_override": str(est.wue) if est.wue is not None else "provider default",
        "registry_version": est.registry.data_version,
        "prefill_scope": "GPU operational impacts; baseline server and embodied impacts unchanged",
        "uncertainty": "Assumption intervals, not statistical confidence intervals",
        "batch_size": est.hardware.batch_size,
    }
    return BatchResult(df, cmap, pm, settings)


__all__ = ["BatchResult", "ColumnMap", "run", "read_table"]
