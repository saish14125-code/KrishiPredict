"""Cached loaders for processed datasets. Raise DataMissingError with an actionable message."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


class DataMissingError(Exception):
    pass


def _cache(fn):
    try:
        import streamlit as st
        return st.cache_data(show_spinner="Loading data...")(fn)
    except Exception:  # streamlit not installed / used from scripts
        return fn


def _ensure(path: Path, builder_hint: str = "python run_preprocessing.py") -> None:
    if path.exists():
        return
    try:  # try to build automatically once
        from src.preprocessing import run_all
        run_all(verbose=False)
    except Exception as e:
        raise DataMissingError(f"{path.name} not found and could not be built ({e}). Run: {builder_hint}") from e
    if not path.exists():
        raise DataMissingError(f"{path.name} not found. Run: {builder_hint}")


@_cache
def load_crop() -> pd.DataFrame:
    _ensure(C.PROC_CROP)
    df = pd.read_csv(C.PROC_CROP)
    missing = {"State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Production", "Yield", "Yield_Valid"} - set(df.columns)
    if missing:
        raise DataMissingError(f"crop_processed.csv lacks columns {sorted(missing)}. Re-run python run_preprocessing.py")
    for c in ["State_Name", "District_Name", "District_Key", "Season", "Crop", "Production_Unit"]:
        df[c] = df[c].astype("category")
    return df


@_cache
def load_rain_annual() -> pd.DataFrame:
    _ensure(C.PROC_RAIN_ANNUAL)
    return pd.read_csv(C.PROC_RAIN_ANNUAL)


@_cache
def load_rain_daily() -> pd.DataFrame:
    _ensure(C.PROC_RAIN_DAILY)
    df = pd.read_csv(C.PROC_RAIN_DAILY, parse_dates=["Date", "Week_Start"])
    return df


@_cache
def load_quality_report() -> pd.DataFrame:
    _ensure(C.QUALITY_REPORT)
    return pd.read_csv(C.QUALITY_REPORT)


def load_json(path: Path):
    import json
    return json.loads(path.read_text()) if path.exists() else None


def load_csv_optional(path: Path):
    return pd.read_csv(path) if path.exists() else None
