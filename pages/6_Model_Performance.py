import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as C
from src.data_loader import load_csv_optional, load_json
from src.utils import csv_button, empty_state, kpi, note, page_header, section, show

page_header("Model Performance", "Chronological evaluation on held-out years, with baselines and honest caveats")
meta, comp = load_json(C.METRICS_FILE), load_csv_optional(C.COMPARISON_FILE)
if meta is None or comp is None:
    st.error("No evaluation results found. Train the models first: `python train_model.py`"); st.stop()
if meta.get("quick_mode"):
    st.warning("Results are from QUICK (12% sample) mode and are not representative.")

sp, sz = meta["split"], meta["sizes"]
note(f"<b>Split (chronological):</b> train {sp['train']} ({sz['train']:,} rows) | validation {sp['validation']} ({sz['val']:,}) - used to select the model | "
     f"test {sp['test']} ({sz['test']:,}) - never used for selection. The deployed model is refit on train+validation. "
     "Rows excluded from modelling: " + ", ".join(f"{k.replace('excluded_', '').replace('_', ' ')} {v:,}" for k, v in meta["excluded"].items() if k.startswith("excluded")) + ".")

dep = comp[comp["Type"] == "deployed"].iloc[0]
k = st.columns(4)
kpi(k[0], "Selected on validation", meta["selected"], "lowest validation RMSE")
kpi(k[1], "Test MAE (deployed)", f"{dep['Test_MAE']:.2f} t/ha")
kpi(k[2], "Test RMSE (deployed)", f"{dep['Test_RMSE']:.2f} t/ha")
kpi(k[3], "Test R² (deployed)", f"{dep['Test_R2']:.3f}", "variance explained, NOT accuracy")

section("Model comparison")
show_cols = ["Model", "Type", "Val_MAE", "Val_RMSE", "Val_R2", "Test_MAE", "Test_RMSE", "Test_R2", "Test_MedAE", "TestAll_RMSE", "Fit_seconds"]
st.dataframe(comp[show_cols].round(3), hide_index=True, width="stretch")
csv_button(comp, "model_comparison.csv", key="mp_dl")
best_test = comp[comp["Type"] == "ml"].sort_values("Test_RMSE").iloc[0]
base = comp[comp["Model"] == "Baseline: crop-district median"].iloc[0]
st.markdown(f"""
**Reading the table honestly**
- Best on the **test** years by RMSE: **{best_test['Model']}** ({best_test['Test_RMSE']:.2f}); the model chosen on validation was **{meta['selected']}**. Differences between the ML models on test are small.
- The simple *crop-district historical median* baseline reaches MAE {base['Test_MAE']:.2f} and R² {base['Test_R2']:.3f}. ML models improve MAE most clearly (deployed: {dep['Test_MAE']:.2f}); on RMSE (dominated by large-yield crops such as sugarcane and banana) the gain is much smaller.
- `TestAll_RMSE` repeats the test including the few extreme-yield records that were excluded from the main test set; it is higher, showing how outliers affect the numbers.
- R² here is the share of yield variance explained across all crops (much of it is simply which crop it is). It is **not** "accuracy".
""")

m = comp[comp["Type"] != "deployed"].melt(id_vars="Model", value_vars=["Test_MAE", "Test_RMSE", "Test_R2"], var_name="Metric", value_name="Value")
cols = st.columns(3)
for col, (met, ttl) in zip(cols, [("Test_MAE", "MAE (t/ha, lower is better)"), ("Test_RMSE", "RMSE (t/ha, lower is better)"), ("Test_R2", "R² (higher is better)")]):
    d = m[m["Metric"] == met]
    with col:
        show(px.bar(d, x="Model", y="Value", color="Model", title=ttl, text_auto=".3f"), height=360, legend=False)

with st.expander("What do these metrics mean?"):
    st.markdown("""
- **MAE** - the average size of the error in tonnes/hectare. MAE 1.0 means predictions are off by 1 t/ha on average.
- **RMSE** - like MAE but large mistakes count more; it is dominated by high-yield crops (sugarcane, banana).
- **R²** - the fraction of variation in yield the model explains (1 = all, 0 = no better than predicting the average). It is not a percentage of correct predictions.
- **MedAE** - the typical (median) error; much smaller than MAE because a few large misses inflate the mean.
""")

tp = load_csv_optional(C.TEST_PRED_FILE)
if tp is not None and len(tp):
    section("Actual vs predicted (deployed model, held-out test sample of 40,000 records max)")
    a, b = st.columns(2)
    s = tp.sample(min(6000, len(tp)), random_state=1)
    with a:
        fig = px.scatter(s, x="Yield", y="Pred_Deployed", opacity=.4, log_x=True, log_y=True,
                         hover_data=["Crop", "State_Name", "District_Name", "Crop_Year"], title="Actual vs predicted yield (log-log, 6,000 points)")
        mx = max(s["Yield"].max(), s["Pred_Deployed"].max())
        fig.add_scatter(x=[0.01, mx], y=[0.01, mx], mode="lines", line=dict(color="#b45309", dash="dash"), name="Perfect prediction")
        fig.update_traces(marker=dict(color=C.GREEN), selector=dict(mode="markers"))
        fig.update_layout(xaxis_title="Actual (t/ha)", yaxis_title="Predicted (t/ha)")
        show(fig, height=420)
    with b:
        res = (tp["Yield"] - tp["Pred_Deployed"])
        lim = res.quantile([.01, .99]).tolist()
        fig = px.histogram(res[(res >= lim[0]) & (res <= lim[1])], nbins=80, title="Residuals (actual - predicted, t/ha; middle 98%)", color_discrete_sequence=[C.GREEN])
        fig.update_layout(showlegend=False, xaxis_title="Residual (t/ha)")
        show(fig, height=420)
        st.caption(f"Mean residual {res.mean():+.3f} t/ha, median {res.median():+.3f}. Positive = model under-predicts.")

imp = load_csv_optional(C.IMPORTANCE_FILE)
if imp is not None:
    section("Feature importance (permutation importance)")
    d = imp[imp["Model"] == "Deployed"].sort_values("MAE_increase")
    fig = px.bar(d, x="MAE_increase", y="Feature", orientation="h", error_x="Std", title="Increase in test MAE (t/ha) when a feature is shuffled - deployed model",
                 color_discrete_sequence=[C.GREEN])
    show(fig, height=340)
    st.caption("Crop and State dominate. Crop_Year shows ~0 importance: the model barely uses the year (the reason is not established here; tree models also cannot extrapolate beyond training years). "
               "Production is never a feature.")
    with st.expander("Importance for all models"):
        st.dataframe(imp.round(4), hide_index=True, width="stretch")

ce = load_csv_optional(C.CROP_ERROR_FILE)
if ce is not None:
    section("Where the deployed model errs: by crop (test years)")
    ce2 = ce.head(20).copy()
    fig = go.Figure()
    fig.add_bar(x=ce2["Crop"], y=ce2["MAE"], name="Deployed model MAE", marker_color=C.GREEN)
    fig.add_bar(x=ce2["Crop"], y=ce2["Baseline_MAE"], name="Crop-district median MAE", marker_color="#fcd34d")
    fig.update_layout(barmode="group", title="MAE by crop (20 crops with most test records, t/ha)")
    show(fig, height=380)
    st.dataframe(ce.round(2), hide_index=True, width="stretch")
    st.caption("Errors are relatively larger for crops with few records or volatile reporting. Check this table before relying on a specific crop.")

with st.expander("Methodology and limitations"):
    st.markdown("""
- Target: yield = production / area (t/ha), learned on a log(1+yield) scale; metrics are reported on the original t/ha scale.
- Features: state, district, crop, season, year, area (+ crop-district and crop-state keys, target-encoded within the pipeline). **Production is excluded.**
- Excluded from modelling: crops not measured in tonnes, zero-production rows (treated as not reported), areas under 1 ha and extreme high-yield outliers.
- Target encoding uses internal 5-fold cross-fitting on the training data (not time-aware), a minor within-training approximation; the test years remain untouched.
- No weather/rainfall input; no random-split experiment was run. Results describe 2020-2022 only and may not hold for other years.
""")
