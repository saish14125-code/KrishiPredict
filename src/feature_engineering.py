"""Feature engineering shared by training and prediction (so both use identical logic)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

# Raw inputs the user/app must supply. Production is deliberately NOT here (it defines the target).
RAW_FEATURES = ["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area"]
CAT_COLS = ["State_Name", "Season", "Crop", "District_Key", "Crop_State", "Crop_District"]
NUM_COLS = ["Crop_Year", "Log_Area"]
ONEHOT_COLS = ["State_Name", "Season", "Crop", "District_Key", "Crop_State"]


def derive_features(X: pd.DataFrame) -> pd.DataFrame:
    """Raw columns -> model columns. Used as the first step inside the sklearn Pipeline."""
    X = pd.DataFrame(X).copy()
    for c in ["State_Name", "District_Name", "Season", "Crop"]:
        X[c] = X[c].astype(str)
    out = pd.DataFrame(index=X.index)
    out["State_Name"] = X["State_Name"].astype(object)
    out["Season"] = X["Season"].astype(object)
    out["Crop"] = X["Crop"].astype(object)
    out["District_Key"] = (X["State_Name"] + " | " + X["District_Name"]).astype(object)
    out["Crop_State"] = (X["Crop"] + " @ " + X["State_Name"]).astype(object)
    out["Crop_District"] = (X["Crop"] + " @ " + out["District_Key"]).astype(object)
    out["Crop_Year"] = pd.to_numeric(X["Crop_Year"]).astype(float)
    out["Log_Area"] = np.log1p(pd.to_numeric(X["Area"]).astype(float))
    return out


def make_ml_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select rows usable for yield modelling and report what was excluded (and why)."""
    n = len(df)
    non_t = df["Production_Unit"] != "tonnes"
    d1 = df[~non_t]
    bad = ~d1["Yield_Valid"]
    info = {
        "total_rows": n,
        "excluded_non_tonne_units": int(non_t.sum()),
        "excluded_zero_production": int(d1["Zero_Production"].sum()),
        "excluded_tiny_area": int((d1["Tiny_Area"] & ~d1["Zero_Production"]).sum()),
        "excluded_yield_outliers": int((d1["Yield_Outlier"] & ~d1["Zero_Production"] & ~d1["Tiny_Area"]).sum()),
    }
    ml = d1[~bad].copy()
    info["modelling_rows"] = len(ml)
    return ml, info


def chronological_split(ml: pd.DataFrame):
    y = ml["Crop_Year"]
    train = ml[y <= C.TRAIN_END_YEAR]
    val = ml[(y > C.TRAIN_END_YEAR) & (y <= C.VAL_END_YEAR)]
    test = ml[y >= C.TEST_START_YEAR]
    return train, val, test
