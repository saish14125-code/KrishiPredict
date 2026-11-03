"""Rainfall analytics. Historical (subdivision, annual) and current (district, daily) are kept strictly separate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


# ---------------- Historical: meteorological subdivisions, annual ----------------
def add_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    g = d.groupby("Rainfall_Subdivision")["Annual_Rainfall_mm"]
    d["Baseline_mm"] = g.transform("mean")
    d["Anomaly_mm"] = d["Annual_Rainfall_mm"] - d["Baseline_mm"]
    d["Anomaly_pct"] = 100 * d["Anomaly_mm"] / d["Baseline_mm"]
    d["Z"] = d["Anomaly_mm"] / g.transform("std")
    return d


def subdivision_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sub, g in df.groupby("Rainfall_Subdivision"):
        g = g.sort_values("Year")
        slope = np.polyfit(g["Year"], g["Annual_Rainfall_mm"], 1)[0] * 10 if len(g) > 5 else np.nan
        rows.append({"Subdivision": sub, "Years": len(g), "Mean_mm": g["Annual_Rainfall_mm"].mean(),
                     "Std_mm": g["Annual_Rainfall_mm"].std(), "CV_pct": 100 * g["Annual_Rainfall_mm"].std() / g["Annual_Rainfall_mm"].mean(),
                     "Wettest_year": int(g.loc[g["Annual_Rainfall_mm"].idxmax(), "Year"]), "Wettest_mm": g["Annual_Rainfall_mm"].max(),
                     "Driest_year": int(g.loc[g["Annual_Rainfall_mm"].idxmin(), "Year"]), "Driest_mm": g["Annual_Rainfall_mm"].min(),
                     "Trend_mm_per_decade": slope})
    return pd.DataFrame(rows).sort_values("Mean_mm", ascending=False)


def mean_series(df: pd.DataFrame) -> pd.DataFrame:
    """Simple (unweighted) mean across the selected subdivisions per year, plus a 10-year rolling mean."""
    s = df.groupby("Year")["Annual_Rainfall_mm"].mean().reset_index()
    s["Rolling_10y"] = s["Annual_Rainfall_mm"].rolling(10, min_periods=5).mean()
    s["YoY_change_pct"] = 100 * s["Annual_Rainfall_mm"].pct_change()
    return s


# ---------------- Current: district, daily ----------------
def window(daily: pd.DataFrame, start, end) -> pd.DataFrame:
    return daily[(daily["Date"] >= pd.Timestamp(start)) & (daily["Date"] <= pd.Timestamp(end))]


def _paired(daily: pd.DataFrame) -> pd.DataFrame:
    """Blank the normal on days without an observation so actual and normal always cover the same days."""
    d = daily.copy()
    d.loc[d["Daily_Rainfall_mm"].isna(), "Normal_Rainfall_mm"] = np.nan
    return d


def district_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Totals over the supplied window. Daily values are non-overlapping, so summing is valid.
    Days with no data (ND) are skipped, and their count is reported."""
    g = _paired(daily).groupby(["State", "District"])
    out = g.agg(Actual_mm=("Daily_Rainfall_mm", lambda s: s.sum(min_count=1)), Normal_mm=("Normal_Rainfall_mm", lambda s: s.sum(min_count=1)), Days=("Date", "nunique"),
                No_data_days=("Daily_Rainfall_mm", lambda s: int(s.isna().sum())),
                Rainy_days=("Daily_Rainfall_mm", lambda s: int((s >= 2.5).sum())), Max_daily_mm=("Daily_Rainfall_mm", "max")).reset_index()
    out["Departure_pct"] = np.where(out["Normal_mm"] > 0, 100 * (out["Actual_mm"] - out["Normal_mm"]) / out["Normal_mm"], np.nan)
    return out


def state_summary(dist: pd.DataFrame) -> pd.DataFrame:
    s = dist.groupby("State").agg(Districts=("District", "nunique"), Mean_district_actual_mm=("Actual_mm", "mean"),
                                  Mean_district_normal_mm=("Normal_mm", "mean")).reset_index()
    s["Departure_pct"] = np.where(s["Mean_district_normal_mm"] > 0,
                                  100 * (s["Mean_district_actual_mm"] - s["Mean_district_normal_mm"]) / s["Mean_district_normal_mm"], np.nan)
    return s.sort_values("Mean_district_actual_mm", ascending=False)


def daily_series(daily: pd.DataFrame) -> pd.DataFrame:
    """Mean over districts per day (NOT a sum across districts)."""
    s = _paired(daily).groupby("Date").agg(Actual=("Daily_Rainfall_mm", "mean"), Normal=("Normal_Rainfall_mm", "mean"), Districts=("District", "nunique")).reset_index()
    s["Cum_Actual"] = s["Actual"].cumsum()
    s["Cum_Normal"] = s["Normal"].cumsum()
    return s


def period_series(daily: pd.DataFrame, col: str) -> pd.DataFrame:
    """Per-district period totals, then mean across districts. col: 'Week_Start' or 'Month'."""
    pd_ = _paired(daily).groupby([col, "State", "District"]).agg(A=("Daily_Rainfall_mm", lambda s: s.sum(min_count=1)), N=("Normal_Rainfall_mm", lambda s: s.sum(min_count=1)), D=("Date", "nunique")).reset_index()
    s = pd_.groupby(col).agg(Actual=("A", "mean"), Normal=("N", "mean"), Days=("D", "max")).reset_index()
    s["Partial"] = s["Days"] < (7 if col == "Week_Start" else 28)
    return s


# ---------------- Valid rainfall-crop link (state level, 1997-2015) ----------------
def state_rain_crop_link(crop: pd.DataFrame, annual: pd.DataFrame, state: str, crop_name: str) -> tuple[pd.DataFrame, dict]:
    """Annual state yield of a crop vs annual rainfall of the matching meteorological subdivision.
    Valid only for states that map 1:1 to a subdivision, and only for years present in both datasets."""
    sub = C.STATE_TO_SUBDIVISION.get(state)
    if sub is None:
        return pd.DataFrame(), {"error": "State has no 1:1 subdivision"}
    v = crop[(crop["State_Name"] == state) & (crop["Crop"] == crop_name) & crop["Yield_Valid"]]
    y = v.groupby("Crop_Year", observed=True).agg(P=("Production", "sum"), A=("Area", "sum")).reset_index()
    y["Yield"] = y["P"] / y["A"]
    r = annual[annual["Rainfall_Subdivision"] == sub][["Year", "Annual_Rainfall_mm"]]
    j = y.merge(r, left_on="Crop_Year", right_on="Year")
    if len(j) < 8:
        return j, {"error": f"Only {len(j)} overlapping years (need >= 8)"}
    raw_r = float(np.corrcoef(j["Annual_Rainfall_mm"], j["Yield"])[0, 1])
    res = lambda c: j[c] - np.polyval(np.polyfit(j["Crop_Year"], j[c], 1), j["Crop_Year"])  # noqa: E731
    det_r = float(np.corrcoef(res("Annual_Rainfall_mm"), res("Yield"))[0, 1])
    return j, {"n": len(j), "subdivision": sub, "r_raw": raw_r, "r_detrended": det_r, "years": f"{j['Crop_Year'].min()}-{j['Crop_Year'].max()}"}
