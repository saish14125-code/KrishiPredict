"""Load the saved model and produce validated, caveated yield predictions."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


class ModelMissingError(Exception):
    pass


def _cache_resource(fn):
    try:
        import streamlit as st
        return st.cache_resource(show_spinner="Loading model...")(fn)
    except Exception:
        return fn


@_cache_resource
def load_bundle(path: str = str(C.MODEL_FILE)) -> dict:
    p = Path(path)
    if not p.exists():
        raise ModelMissingError(f"Trained model not found at {p}. Train it first: python train_model.py")
    try:
        return joblib.load(p)
    except Exception as e:  # version mismatch, corrupt file...
        raise ModelMissingError(f"Could not load the model ({e}). Re-train with: python train_model.py") from e


def predict_yield(bundle: dict, crop_df: pd.DataFrame, state: str, district: str, crop: str, season: str,
                  year: int, area: float) -> dict:
    """Returns dict with ok/errors/warnings/yield/production/range/history."""
    res = {"ok": False, "errors": [], "warnings": []}
    try:
        area = float(area)
    except (TypeError, ValueError):
        area = float("nan")
    if not np.isfinite(area) or area <= 0:
        res["errors"].append("Cultivated area must be a positive number of hectares.")
    if not all([state, district, crop, season]):
        res["errors"].append("Select state, district, crop and season.")
    if crop in C.NON_TONNE_CROPS:
        res["errors"].append(f"{crop} is recorded in {C.NON_TONNE_CROPS[crop]}, not tonnes, so it is not supported by the tonnes/hectare model.")
    elif crop and crop not in bundle["crops_supported"]:
        res["errors"].append(f"{crop} was not present in the training data.")
    if res["errors"]:
        return res

    cov = bundle["coverage"]
    y0, y1 = bundle["train_year_min"], bundle["train_year_max"]
    if year < y0 or year > y1:
        if year > bundle["data_year_max"]:
            res["warnings"].append(f"{year} is beyond the available data ({y0}-{bundle['data_year_max']}). Tree models cannot extrapolate trends; this is a rough, unreliable estimate.")
        elif year > y1:
            res["warnings"].append(f"The deployed model was trained on {y0}-{y1}; {year} lies after that period (its records exist in the data but were held out for testing). Treat as an out-of-sample estimate.")
        else:
            res["warnings"].append(f"{year} is before the training range ({y0}-{y1}); estimate may be unreliable.")
    cc = cov[(cov["State_Name"] == state) & (cov["District_Name"] == district) & (cov["Crop"] == crop)]
    if cc.empty:
        if cov[(cov["State_Name"] == state) & (cov["Crop"] == crop)].empty:
            res["warnings"].append(f"No training history for {crop} in {state}. The estimate relies mostly on all-India patterns for this crop and is LOW CONFIDENCE.")
        else:
            res["warnings"].append(f"No training history for {crop} in {district}. The estimate relies on {state}-level patterns for this crop.")
    else:
        if (cc["Season"] == season).sum() == 0:
            res["warnings"].append(f"{crop} was never recorded in {season} season for this district in the training data.")
        lo, hi = cc["Area_min"].min(), cc["Area_max"].max()
        if area > 3 * hi or area < lo / 3:
            res["warnings"].append(f"Area {area:,.0f} ha is far outside the historical range for this district-crop ({lo:,.0f}-{hi:,.0f} ha).")

    X = pd.DataFrame([{"State_Name": state, "District_Name": district, "Crop_Year": int(year), "Season": season, "Crop": crop, "Area": area}])
    y = float(max(bundle["pipeline"].predict(X)[0], 0))
    q = bundle["error_ratio_q"].get(crop, bundle["error_ratio_q"]["__all__"])
    res.update(ok=True, yield_t_ha=y, production_t=y * area, range_t_ha=(y * q[0], y * q[1]),
               range_basis="crop-specific" if crop in bundle["error_ratio_q"] else "all-crop")
    res["history"] = history_comparison(crop_df, state, district, crop)
    return res


def history_comparison(crop_df: pd.DataFrame, state: str, district: str, crop: str) -> pd.DataFrame:
    """Area-weighted historical yield at district, state and all-India level for the same crop."""
    v = crop_df[(crop_df["Crop"] == crop) & crop_df["Yield_Valid"]]
    rows = []
    for label, d in [(f"{district} (district)", v[(v["State_Name"] == state) & (v["District_Name"] == district)]),
                     (f"{state} (state)", v[v["State_Name"] == state]), ("India (all districts)", v)]:
        if len(d):
            rows.append({"Level": label, "Yield_t_ha": d["Production"].sum() / d["Area"].sum(),
                         "Median_record_yield": d["Yield"].median(), "Records": len(d),
                         "Last_year": int(d["Crop_Year"].max())})
    return pd.DataFrame(rows)
