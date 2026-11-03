import plotly.graph_objects as go
import streamlit as st

import config as C
from src.data_loader import DataMissingError, load_crop
from src.prediction import ModelMissingError, load_bundle, predict_yield
from src.utils import csv_button, kpi, note, page_header, show

page_header("Crop Yield Prediction", "Model-based yield estimate (tonnes/hectare) from state, district, crop, season, year and area")
try:
    df = load_crop()
    bundle = load_bundle()
except (DataMissingError, ModelMissingError) as e:
    st.error(str(e)); st.stop()

y0, y1 = bundle["train_year_min"], bundle["train_year_max"]
note(f"<b>Model:</b> {bundle['model_name']} trained on {y0}-{y1} district records (Production is never an input). "
     "Output is a statistical estimate for demonstration - <b>not guaranteed agricultural advice</b>. "
     "Rainfall is <b>not</b> a model input: no historical district-level rainfall for 1997-2022 is available.")
if bundle.get("quick_mode"):
    st.warning("This model was trained in QUICK (sample) mode. Run `python train_model.py` for the full model.")

tonne_crops = sorted(set(bundle["crops_supported"]))
c1, c2, c3 = st.columns(3)
state = c1.selectbox("State", sorted(df["State_Name"].cat.categories), index=sorted(df["State_Name"].cat.categories).index("Punjab"))
dists = sorted(df.loc[df["State_Name"] == state, "District_Name"].cat.remove_unused_categories().unique())
district = c2.selectbox("District", dists)
in_dist = sorted(set(df.loc[(df["State_Name"] == state) & (df["District_Name"] == district) & df["Crop"].isin(tonne_crops), "Crop"].astype(str)))
crop_scope = c3.radio("Crop list", ["Grown in this district", "All supported crops"], horizontal=True)
crops = in_dist if (crop_scope == "Grown in this district" and in_dist) else tonne_crops
c4, c5, c6, c7 = st.columns(4)
crop = c4.selectbox("Crop", crops, index=crops.index("Wheat") if "Wheat" in crops else 0)
seasons = sorted(set(df.loc[(df["Crop"] == crop) & (df["State_Name"] == state), "Season"].astype(str))) or sorted(df["Season"].cat.categories)
season = c5.selectbox("Season", seasons)
year = c6.number_input("Year", min_value=1990, max_value=2035, value=min(y1, 2019), step=1,
                       help=f"Default {min(y1, 2019)} lies inside the training range {y0}-{y1}.")
hist_area = df[(df["State_Name"] == state) & (df["District_Name"] == district) & (df["Crop"] == crop)]["Area"]
area = c7.number_input("Cultivated area (ha)", min_value=0.0, value=float(round(hist_area.median(), 0)) if len(hist_area) else 100.0, step=10.0)

if st.button("Predict yield", type="primary"):
    res = predict_yield(bundle, df, state, district, crop, season, int(year), area)
    for e in res["errors"]:
        st.error(e)
    if res["ok"]:
        for w in res["warnings"]:
            st.warning(w)
        lo, hi = res["range_t_ha"]
        k = st.columns(3)
        kpi(k[0], "Estimated yield", f"{res['yield_t_ha']:.2f} t/ha", f"typical error band {lo:.2f} - {hi:.2f} t/ha")
        kpi(k[1], "Estimated production", f"{res['production_t']:,.0f} t", f"= yield x {area:,.0f} ha")
        kpi(k[2], "Model", bundle["model_name"], f"trained {y0}-{y1}")
        st.caption(f"The band is the 10th-90th percentile of (actual / predicted) from held-out 2020-2022 test records "
                   f"({res['range_basis']}). It describes past errors and is not a guaranteed interval.")
        h = res["history"]
        if not h.empty:
            fig = go.Figure()
            fig.add_bar(x=h["Level"], y=h["Yield_t_ha"], name="Historical yield (area-weighted)", marker_color=C.GREEN_MID)
            fig.add_hline(y=res["yield_t_ha"], line_dash="dash", line_color=C.GREEN_DARK, annotation_text="Model estimate")
            fig.update_layout(title=f"{crop}: model estimate vs historical yield (t/ha)", yaxis_title="t/ha")
            show(fig, height=340, legend=False)
            st.dataframe(h.round(2), hide_index=True, width="stretch")
            csv_button(h, "prediction_history.csv", key="pred_dl")
        else:
            st.info("No historical records exist for this crop in the selected district, state or country.")
else:
    st.info("Choose inputs and click **Predict yield**.")

with st.expander("How to read this prediction"):
    st.markdown("""
- The model learns typical yield for a crop-district from past records plus the year and area. It captures **where and what** is grown far better than **year-to-year weather**.
- Tree models cannot extrapolate trends beyond the years they saw; dates outside the training range trigger a warning.
- See **Model Performance** for honest test results, including simple historical-median baselines the model must beat.
""")
