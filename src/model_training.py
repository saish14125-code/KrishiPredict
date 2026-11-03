"""Train, compare and persist yield-prediction models.

Protocol (no leakage, chronological):
  Stage 1  fit on 1997-2017  -> score on validation (2018-19) and test (2020-22)
           model selection uses VALIDATION RMSE only.
  Stage 2  refit the selected model on 1997-2019 -> score once more on the untouched test years.
           This refitted model is the one saved for the app.
Baselines (historical medians) are reported so that ML skill can be judged honestly.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, TargetEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.feature_engineering import (CAT_COLS, NUM_COLS, ONEHOT_COLS, RAW_FEATURES,  # noqa: E402
                                     chronological_split, derive_features, make_ml_frame)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # optional dependency
    HAS_XGB = False


def _log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def build_pipeline(kind: str) -> TransformedTargetRegressor:
    """Pipeline = derive features -> encode -> regressor, wrapped so the model learns log(1+yield)."""
    if kind == "Linear Regression":
        enc = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), ONEHOT_COLS),
            ("num", StandardScaler(), NUM_COLS)])
        reg = LinearRegression()
    else:
        enc = ColumnTransformer([
            ("cat", TargetEncoder(target_type="continuous", smooth="auto", cv=5, random_state=C.RANDOM_STATE), CAT_COLS),
            ("num", "passthrough", NUM_COLS)])
        if kind == "Random Forest":
            reg = RandomForestRegressor(n_estimators=60, min_samples_leaf=15, max_features=0.8, max_samples=0.5,
                                        n_jobs=-1, random_state=C.RANDOM_STATE)
        elif kind == "XGBoost":
            reg = XGBRegressor(n_estimators=400, learning_rate=0.06, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                               min_child_weight=5, tree_method="hist", n_jobs=-1, random_state=C.RANDOM_STATE)
        else:
            raise ValueError(kind)
    pipe = Pipeline([("derive", FunctionTransformer(derive_features, validate=False)), ("encode", enc), ("model", reg)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, None)
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p)), "MedAE": float(median_absolute_error(y, p)),
            "R2_log": float(r2_score(np.log1p(y), np.log1p(p))), "n": int(len(y))}


def baseline_predictions(train: pd.DataFrame, test: pd.DataFrame, level: str) -> np.ndarray:
    """Historical-median baselines. level: 'crop' or 'crop_district' (falls back crop-state -> crop -> global)."""
    glob = train["Yield"].median()
    crop = test["Crop"].map(train.groupby("Crop")["Yield"].median())
    if level == "crop":
        return crop.fillna(glob).to_numpy()
    cs_med = train.groupby(["Crop", "State_Name"])["Yield"].median()
    cd_med = train.groupby(["Crop", "District_Key"])["Yield"].median()
    cs = pd.Series(list(zip(test["Crop"], test["State_Name"])), index=test.index).map(cs_med)
    cd = pd.Series(list(zip(test["Crop"], test["District_Key"])), index=test.index).map(cd_med)
    return cd.fillna(cs).fillna(crop).fillna(glob).to_numpy()


def _fit(kind, df):
    t0 = time.time()
    m = build_pipeline(kind)
    m.fit(df[RAW_FEATURES], df["Yield"])
    return m, time.time() - t0


def run_training(quick: bool = False, skip_xgb: bool = False) -> dict:
    C.MODELS_DIR.mkdir(exist_ok=True)
    if not C.PROC_CROP.exists():
        raise FileNotFoundError("processed/crop_processed.csv not found. Run: python run_preprocessing.py")
    df = pd.read_csv(C.PROC_CROP)
    ml, excl = make_ml_frame(df)
    if quick:
        ml = ml.sample(frac=0.12, random_state=C.RANDOM_STATE)
        _log(f"QUICK MODE: using a 12% sample ({len(ml):,} rows) - for smoke tests only")
    train, val, test = chronological_split(ml)
    # test set that also keeps flagged high-yield outliers / tiny areas (zero-production rows still excluded)
    full_test = df[(df["Production_Unit"] == "tonnes") & ~df["Zero_Production"] & (df["Crop_Year"] >= C.TEST_START_YEAR)]
    if quick:
        full_test = full_test.sample(frac=0.12, random_state=C.RANDOM_STATE)
    _log(f"rows: train {len(train):,} (<= {C.TRAIN_END_YEAR}) | val {len(val):,} | test {len(test):,} | test incl. outliers {len(full_test):,}")

    kinds = ["Linear Regression", "Random Forest"] + (["XGBoost"] if (HAS_XGB and not skip_xgb) else [])
    if not HAS_XGB:
        _log("XGBoost not available - continuing with scikit-learn models only")

    rows, fitted, preds_test = [], {}, {}
    for name, level in [("Baseline: crop median", "crop"), ("Baseline: crop-district median", "crop_district")]:
        r = {"Model": name, "Type": "baseline", "Fit_seconds": 0.0}
        for split, d in [("val", val), ("test", test), ("test_all", full_test)]:
            r[split] = metrics(d["Yield"], baseline_predictions(train, d, level))
        rows.append(r)
        preds_test[name] = baseline_predictions(train, test, level)

    for k in kinds:
        _log(f"fitting {k} on 1997-{C.TRAIN_END_YEAR} ...")
        m, secs = _fit(k, train)
        fitted[k] = m
        r = {"Model": k, "Type": "ml", "Fit_seconds": round(secs, 1)}
        for split, d in [("val", val), ("test", test), ("test_all", full_test)]:
            r[split] = metrics(d["Yield"], m.predict(d[RAW_FEATURES]))
        preds_test[k] = np.clip(m.predict(test[RAW_FEATURES]), 0, None)
        rows.append(r)
        _log(f"  {k}: val RMSE {r['val']['RMSE']:.3f} | test RMSE {r['test']['RMSE']:.3f} | test R2 {r['test']['R2']:.3f} ({secs:.0f}s)")

    best_name = min([r for r in rows if r["Type"] == "ml"], key=lambda r: r["val"]["RMSE"])["Model"]
    _log(f"selected on VALIDATION RMSE: {best_name}. Refitting on 1997-{C.VAL_END_YEAR} ...")
    trainval = pd.concat([train, val])
    final, secs = _fit(best_name, trainval)
    final_pred = np.clip(final.predict(test[RAW_FEATURES]), 0, None)
    final_row = {"Model": f"{best_name} (deployed, refit to {C.VAL_END_YEAR})", "Type": "deployed", "Fit_seconds": round(secs, 1),
                 "val": None, "test": metrics(test["Yield"], final_pred),
                 "test_all": metrics(full_test["Yield"], final.predict(full_test[RAW_FEATURES]))}
    rows.append(final_row)
    _log(f"  deployed model test RMSE {final_row['test']['RMSE']:.3f} | R2 {final_row['test']['R2']:.3f}")

    # --- permutation importance on raw inputs (model-agnostic, test sample) ---
    samp = test.sample(min(8000, len(test)), random_state=C.RANDOM_STATE)
    imp_rows = []
    for k, m in list(fitted.items()) + [("Deployed", final)]:
        pi = permutation_importance(m, samp[RAW_FEATURES], samp["Yield"], scoring="neg_mean_absolute_error",
                                    n_repeats=3, random_state=C.RANDOM_STATE, n_jobs=1)
        for f, mu, sd in zip(RAW_FEATURES, pi.importances_mean, pi.importances_std):
            imp_rows.append({"Model": k, "Feature": f, "MAE_increase": mu, "Std": sd})
    pd.DataFrame(imp_rows).to_csv(C.IMPORTANCE_FILE, index=False)

    # --- per-crop error table + empirical error ratios (actual / predicted) for the deployed model ---
    te = test[["State_Name", "District_Name", "Crop_Year", "Season", "Crop", "Area", "Yield"]].copy()
    te["Pred_Deployed"] = final_pred
    for k, p in preds_test.items():
        te["Pred_" + k] = p
    te["Abs_Error"] = (te["Yield"] - te["Pred_Deployed"]).abs()
    ce = te.groupby("Crop").agg(Test_Records=("Yield", "size"), Mean_Actual=("Yield", "mean"), MAE=("Abs_Error", "mean")).reset_index()
    ce["MAE_pct_of_mean"] = 100 * ce["MAE"] / ce["Mean_Actual"]
    base_cd = baseline_predictions(train, test, "crop_district")
    ce = ce.merge(te.assign(b=np.abs(te["Yield"] - base_cd)).groupby("Crop")["b"].mean().rename("Baseline_MAE").reset_index(), on="Crop")
    ce.sort_values("Test_Records", ascending=False).to_csv(C.CROP_ERROR_FILE, index=False)
    ratio = (te["Yield"] / te["Pred_Deployed"].clip(lower=1e-3))
    ratio_q = {"__all__": [float(ratio.quantile(.1)), float(ratio.quantile(.9))]}
    for crop, g in ratio.groupby(te["Crop"]):
        if len(g) >= 40:
            ratio_q[crop] = [float(g.quantile(.1)), float(g.quantile(.9))]
    te.sample(min(40000, len(te)), random_state=C.RANDOM_STATE).to_csv(C.TEST_PRED_FILE, index=False)

    coverage = trainval.groupby(["State_Name", "District_Name", "Crop", "Season"]).agg(
        Records=("Yield", "size"), Area_min=("Area", "min"), Area_max=("Area", "max")).reset_index()
    comp = pd.DataFrame([{
        "Model": r["Model"], "Type": r["Type"],
        **{f"Val_{k}": (r["val"][k] if r["val"] else np.nan) for k in ["MAE", "RMSE", "R2"]},
        **{f"Test_{k}": r["test"][k] for k in ["MAE", "RMSE", "R2", "MedAE", "R2_log"]},
        **{f"TestAll_{k}": r["test_all"][k] for k in ["MAE", "RMSE", "R2"]},
        "Fit_seconds": r["Fit_seconds"]} for r in rows])
    comp.to_csv(C.COMPARISON_FILE, index=False)

    bundle = {"pipeline": final, "model_name": best_name, "raw_features": RAW_FEATURES,
              "train_year_min": int(trainval["Crop_Year"].min()), "train_year_max": int(trainval["Crop_Year"].max()),
              "data_year_max": int(df["Crop_Year"].max()), "coverage": coverage, "error_ratio_q": ratio_q,
              "crops_supported": sorted(trainval["Crop"].unique().tolist()), "quick_mode": quick,
              "trained_at": datetime.now().isoformat(timespec="seconds"),
              "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__}}
    joblib.dump(bundle, C.MODEL_FILE, compress=3)
    out = {"rows": rows, "selected": best_name, "excluded": excl, "quick_mode": quick,
           "split": {"train": f"1997-{C.TRAIN_END_YEAR}", "validation": f"{C.TRAIN_END_YEAR+1}-{C.VAL_END_YEAR}",
                     "test": f"{C.TEST_START_YEAR}-{int(df['Crop_Year'].max())}"},
           "sizes": {"train": len(train), "val": len(val), "test": len(test), "test_all": len(full_test)},
           "xgboost_available": HAS_XGB, "trained_at": bundle["trained_at"]}
    C.METRICS_FILE.write_text(json.dumps(out, indent=2))
    _log(f"saved {C.MODEL_FILE.name} ({C.MODEL_FILE.stat().st_size/1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="12%% sample smoke test")
    ap.add_argument("--no-xgb", action="store_true")
    a = ap.parse_args()
    run_training(a.quick, a.no_xgb)
