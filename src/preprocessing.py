"""Reusable preprocessing pipeline.

Source files are NEVER modified. Cleaned outputs are written to ``processed/``.

Crop data source priority:
  1. data/crop_production.csv  (original file - cleaned from scratch here)
  2. data/krishipredict_crop_master.csv (previously cleaned file - re-validated here)
Rainfall data: the previously processed files are re-validated and standardised.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

REQUIRED_CROP_COLS = ["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production"]
_ALIASES = {
    "state_name": "State_Name", "state": "State_Name", "district_name": "District_Name",
    "district": "District_Name", "crop_year": "Crop_Year", "year": "Crop_Year", "season": "Season",
    "crop": "Crop", "area": "Area", "production": "Production", "yield": "Yield",
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: _ALIASES.get(str(c).strip().lower().replace(" ", "_"), c) for c in df.columns})


def _step(log: list, name: str, before: int, after: int, note: str = "") -> None:
    log.append({"step": name, "rows_before": before, "rows_after": after, "rows_removed": before - after, "note": note})


# --------------------------------------------------------------------------------------
# Crop data
# --------------------------------------------------------------------------------------
def flag_yield_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add quality flags. Nothing is deleted: flagged rows stay in the processed file."""
    df = df.copy()
    df["Production_Unit"] = df["Crop"].map(C.NON_TONNE_CROPS).fillna("tonnes")
    df["Zero_Production"] = df["Production"] <= 0
    df["Tiny_Area"] = df["Area"] < C.MIN_VALID_AREA_HA
    # Far-out HIGH outliers per crop on log(yield) (Tukey k=3). Low yields are kept: crop failure is real.
    pos = df["Yield"].where(~df["Zero_Production"] & (df["Area"] > 0))
    ly = np.log(pos)
    grp = ly.groupby(df["Crop"], observed=True)
    q1, q3, cnt = grp.quantile(0.25), grp.quantile(0.75), grp.count()
    fence = (q3 + C.OUTLIER_IQR_K * (q3 - q1)).where(cnt >= 20)
    df["Yield_Outlier"] = (ly > df["Crop"].map(fence)).fillna(False).astype(bool)
    df["Yield_Valid"] = ~(df["Zero_Production"] | df["Tiny_Area"] | df["Yield_Outlier"])
    return df


def clean_crop_frame(raw: pd.DataFrame, source: str, from_original: bool) -> tuple[pd.DataFrame, list]:
    log: list = []
    df = _standardize_columns(raw)
    missing = [c for c in REQUIRED_CROP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Crop file '{source}' is missing required columns: {missing}. Found: {list(df.columns)}")
    df = df[REQUIRED_CROP_COLS].copy()
    n0 = len(df)
    _step(log, "Load source file", n0, n0, source)

    for c in ["State_Name", "District_Name", "Season", "Crop"]:
        df[c] = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    if from_original:  # the original file has inconsistent capitalisation
        for c in ["State_Name", "District_Name", "Season"]:
            df[c] = df[c].str.title()
    for c in ["Crop_Year", "Area", "Production"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n = len(df)
    df = df.dropna(subset=REQUIRED_CROP_COLS)
    _step(log, "Drop rows with missing key/Area/Production values", n, len(df))
    n = len(df)
    df = df.drop_duplicates()
    _step(log, "Drop exact duplicate rows", n, len(df))
    n = len(df)
    df = df[np.isfinite(df["Area"]) & (df["Area"] > 0)]
    _step(log, "Drop rows with Area <= 0 (yield undefined)", n, len(df))
    n = len(df)
    df = df[np.isfinite(df["Production"]) & (df["Production"] >= 0)]
    _step(log, "Drop rows with negative Production", n, len(df))

    df["Crop_Year"] = df["Crop_Year"].astype(int)
    df["Yield"] = df["Production"] / df["Area"]
    df["District_Key"] = df["State_Name"] + " | " + df["District_Name"]
    df = flag_yield_quality(df)
    for c in ["State_Name", "District_Name", "Season", "Crop", "District_Key"]:
        df[c] = df[c].astype(str)
    key_dups = int(df.duplicated(["State_Name", "District_Name", "Crop_Year", "Season", "Crop"]).sum())
    _step(log, "Check duplicate (state,district,year,season,crop) keys", len(df), len(df), f"{key_dups} found (kept, reported only)")
    df = df.sort_values(["State_Name", "District_Name", "Crop", "Crop_Year", "Season"]).reset_index(drop=True)
    return df, log


def process_crop() -> tuple[pd.DataFrame, list, dict]:
    if C.RAW_CROP.exists():
        src, from_original = C.RAW_CROP, True
    elif C.MASTER_CROP.exists():
        src, from_original = C.MASTER_CROP, False
    else:
        raise FileNotFoundError(f"No crop file found. Expected {C.RAW_CROP} or {C.MASTER_CROP}.")
    raw = pd.read_csv(src)
    df, log = clean_crop_frame(raw, src.name, from_original)
    C.PROCESSED_DIR.mkdir(exist_ok=True)
    df.to_csv(C.PROC_CROP, index=False)
    info = {"source": src.name, "original_rows": len(raw), "from_original": from_original,
            "raw_missing": raw.isna().sum().to_dict(), "raw_dtypes": {k: str(v) for k, v in raw.dtypes.items()}}
    return df, log, info


# --------------------------------------------------------------------------------------
# Rainfall data
# --------------------------------------------------------------------------------------
def process_rain_annual() -> tuple[pd.DataFrame | None, dict]:
    if not C.MASTER_ANNUAL_RAIN.exists():
        return None, {"error": f"missing {C.MASTER_ANNUAL_RAIN.name}"}
    raw = pd.read_csv(C.MASTER_ANNUAL_RAIN)
    df = raw.rename(columns={c: c.strip() for c in raw.columns}).copy()
    df["Rainfall_Subdivision"] = df["Rainfall_Subdivision"].astype(str).str.strip().str.upper()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Annual_Rainfall_mm"] = pd.to_numeric(df["Annual_Rainfall_mm"], errors="coerce")
    n = len(df)
    df = df.dropna().drop_duplicates(["Rainfall_Subdivision", "Year"])
    df = df[df["Annual_Rainfall_mm"] >= 0].astype({"Year": int}).sort_values(["Rainfall_Subdivision", "Year"])
    df.to_csv(C.PROC_RAIN_ANNUAL, index=False)
    return df, {"original_rows": len(raw), "removed": n - len(df), "missing": raw.isna().sum().to_dict()}


def process_rain_daily() -> tuple[pd.DataFrame | None, dict]:
    if not C.MASTER_DAILY_RAIN.exists():
        return None, {"error": f"missing {C.MASTER_DAILY_RAIN.name}"}
    raw = pd.read_csv(C.MASTER_DAILY_RAIN)
    df = raw.copy()
    df["State"] = df["State"].astype(str).str.strip().str.upper().replace(C.RAIN_STATE_FIXES)
    df["District"] = df["District"].astype(str).str.strip().str.upper()
    n_state_fix = int((raw["State"].str.strip().str.upper() != df["State"]).sum())
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Rainfall_Category"] = df["Rainfall_Category"].astype(str).str.strip().str.upper()
    for c in ["Daily_Rainfall_mm", "Normal_Rainfall_mm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_nd = int((df["Rainfall_Category"] == "ND").sum())
    # 'ND' = no data: the 0.0 in the rainfall column is a placeholder, not a measurement.
    df.loc[df["Rainfall_Category"] == "ND", "Daily_Rainfall_mm"] = np.nan
    n = len(df)
    df = df.dropna(subset=["Date"]).drop_duplicates(["State", "District", "Date"])
    df["Category_Label"] = df["Rainfall_Category"].map(C.RAIN_CATEGORY_LABELS).fillna("Unknown")
    df["Week_Start"] = df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")
    df["Month"] = df["Date"].dt.strftime("%Y-%m")
    df["State"] = df["State"].str.title().str.replace(" And ", " and ").replace({"Nct Of Delhi": "Delhi"})
    df["District"] = df["District"].str.title()
    df = df.sort_values(["State", "District", "Date"]).reset_index(drop=True)
    df.to_csv(C.PROC_RAIN_DAILY, index=False)
    all_days = pd.date_range(df["Date"].min(), df["Date"].max())
    missing_days = [d.strftime("%Y-%m-%d") for d in all_days.difference(pd.DatetimeIndex(df["Date"].unique()))]
    return df, {"original_rows": len(raw), "removed": n - len(df), "state_name_fixes": n_state_fix,
                "no_data_rows": n_nd, "missing_calendar_days": missing_days, "missing": raw.isna().sum().to_dict()}


# --------------------------------------------------------------------------------------
# Data quality report
# --------------------------------------------------------------------------------------
def build_quality_report(crop, crop_log, crop_info, rain_a, a_info, rain_d, d_info) -> pd.DataFrame:
    rows: list[tuple[str, str, str]] = []
    add = lambda s, i, v: rows.append((s, i, str(v)))  # noqa: E731

    for name, path in [("crop_production.csv", C.RAW_CROP), ("crop_clean.csv", C.RAW_CROP_CLEAN),
                       ("rainfall.csv", C.RAW_ANNUAL_RAIN), ("rainfall_districtwise_daily_imd.csv", C.RAW_DAILY_RAIN),
                       ("krishipredict_crop_master.csv", C.MASTER_CROP),
                       ("krishipredict_annual_rainfall.csv", C.MASTER_ANNUAL_RAIN),
                       ("krishipredict_current_daily_rainfall.csv", C.MASTER_DAILY_RAIN)]:
        add("Source files", name, "present" if path.exists() else "MISSING")

    add("Crop data", "Source used", crop_info["source"])
    add("Crop data", "Original record count", f"{crop_info['original_rows']:,}")
    add("Crop data", "Cleaned record count", f"{len(crop):,}")
    for s in crop_log:
        add("Crop cleaning steps", s["step"], f"removed {s['rows_removed']:,} | rows after {s['rows_after']:,} {s['note']}".strip())
    for k, v in crop_info["raw_missing"].items():
        add("Crop missing values (source)", k, v)
    for k, v in crop_info["raw_dtypes"].items():
        add("Crop data types (source)", k, v)
    add("Crop flags (kept, not deleted)", "Zero production (area>0) - treated as 'not reported' for yield", f"{int(crop['Zero_Production'].sum()):,}")
    add("Crop flags (kept, not deleted)", f"Tiny area (< {C.MIN_VALID_AREA_HA} ha)", f"{int(crop['Tiny_Area'].sum()):,}")
    add("Crop flags (kept, not deleted)", "Far-out high yield outliers (per-crop log-IQR, k=3)", f"{int(crop['Yield_Outlier'].sum()):,}")
    add("Crop flags (kept, not deleted)", "Records valid for yield statistics / ML", f"{int(crop['Yield_Valid'].sum()):,}")
    for u, g in crop.groupby("Production_Unit"):
        add("Production units (assumption)", u, f"{g['Crop'].nunique()} crop(s), {len(g):,} records: {', '.join(sorted(g['Crop'].unique())[:8])}")
    add("Crop coverage", "Years", f"{crop['Crop_Year'].min()}-{crop['Crop_Year'].max()}")
    add("Crop coverage", "States / districts / crops", f"{crop['State_Name'].nunique()} / {crop['District_Key'].nunique()} / {crop['Crop'].nunique()}")
    by_year = crop.groupby("Crop_Year").size()
    add("Crop coverage", "Records in first vs last year (reporting coverage grows)", f"{by_year.iloc[0]:,} vs {by_year.iloc[-1]:,}")

    if rain_a is not None:
        add("Annual rainfall", "Original / cleaned records", f"{a_info['original_rows']:,} / {len(rain_a):,}")
        add("Annual rainfall", "Years / subdivisions", f"{rain_a['Year'].min()}-{rain_a['Year'].max()} / {rain_a['Rainfall_Subdivision'].nunique()}")
        add("Annual rainfall", "Missing values (source)", sum(a_info["missing"].values()))
    else:
        add("Annual rainfall", "STATUS", a_info.get("error"))
    if rain_d is not None:
        add("Current daily rainfall", "Original / cleaned records", f"{d_info['original_rows']:,} / {len(rain_d):,}")
        add("Current daily rainfall", "Date range", f"{rain_d['Date'].min().date()} to {rain_d['Date'].max().date()}")
        add("Current daily rainfall", "States / districts", f"{rain_d['State'].nunique()} / {rain_d[['State','District']].drop_duplicates().shape[0]}")
        add("Current daily rainfall", "State spelling variants merged (rows)", d_info["state_name_fixes"])
        add("Current daily rainfall", "'ND' (no data) rows - rainfall set to missing", f"{d_info['no_data_rows']:,}")
        add("Current daily rainfall", "Calendar days absent from file", ", ".join(d_info["missing_calendar_days"]) or "none")
        add("Current daily rainfall", "Columns available", "Daily actual, daily normal, IMD category only (no weekly/cumulative columns)")
    else:
        add("Current daily rainfall", "STATUS", d_info.get("error"))

    rep = pd.DataFrame(rows, columns=["Section", "Item", "Value"])
    rep.to_csv(C.QUALITY_REPORT, index=False)
    return rep


def run_all(verbose: bool = True) -> pd.DataFrame:
    crop, log, info = process_crop()
    ra, ai = process_rain_annual()
    rd, di = process_rain_daily()
    rep = build_quality_report(crop, log, info, ra, ai, rd, di)
    if verbose:
        print(rep.to_string(index=False, max_colwidth=110))
        print(f"\nSaved: {C.PROC_CROP.name}, {C.PROC_RAIN_ANNUAL.name}, {C.PROC_RAIN_DAILY.name}, {C.QUALITY_REPORT.name}")
    return rep


if __name__ == "__main__":
    run_all()
