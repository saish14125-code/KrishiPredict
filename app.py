"""KrishiPredict - entry point (navigation + Overview page).   Run:  streamlit run app.py"""
import plotly.express as px
import streamlit as st

import config as C
from src import eda
from src.data_loader import DataMissingError, load_crop
from src.utils import (UNIT_NOTE, csv_button, empty_state, filter_bar, fmt, kpi, note, page_header, section, show)
from src.visualizations import donut, hbar, lines

st.set_page_config(page_title="KrishiPredict", page_icon="🌾", layout="wide")


def overview():
    page_header("Agricultural Overview", "India crop production 1997-2022 | state, crop, season and district performance at a glance")
    try:
        df = load_crop()
    except DataMissingError as e:
        st.error(str(e)); st.stop()
    f, sel = filter_bar(df, "ov")
    if f.empty:
        empty_state(); st.stop()
    t = eda.tonne(f)
    valid = f[f["Yield_Valid"]]
    c = st.columns(6)
    kpi(c[0], "Agricultural records", f"{len(f):,}", "district-crop-season-year rows")
    kpi(c[1], "States / UTs", f"{f['State_Name'].nunique()}")
    kpi(c[2], "Districts", f"{f['District_Key'].nunique():,}")
    kpi(c[3], "Crops", f"{f['Crop'].nunique()}")
    kpi(c[4], "Total production", fmt(t["Production"].sum(), "t"), "tonne-measured crops only")
    kpi(c[5], "Median yield", f"{valid['Yield'].median():.2f} t/ha" if len(valid) else "n/a", "median of valid records")
    note(UNIT_NOTE + " Reporting coverage grows over time (about 9.4 K records in 1997 vs 18.7 K in 2022), so raw yearly totals partly reflect more districts reporting.")
    if t.empty:
        empty_state("The selection contains only crops not recorded in tonnes (coconut, cotton, jute, mesta), so tonnage charts are unavailable."); st.stop()

    a, b = st.columns(2)
    with a:
        s = eda.production_by(t, "State_Name", 15)
        show(hbar(s, "Production", "State_Name", "State-wise production (top 15, tonnes)"), height=460)
    with b:
        cr = eda.production_by(t, "Crop", 10)
        show(hbar(cr, "Production", "Crop", "Top 10 crops by production (tonnes)"), height=460)

    a, b = st.columns([1.6, 1])
    with a:
        top5 = eda.production_by(t, "Crop", 5)["Crop"].tolist()
        yp = eda.yearly_production(t[t["Crop"].isin(top5)], "Crop")
        show(lines(yp, "Crop_Year", "Production", "Crop", "Production trend: top 5 crops (tonnes)"))
    with b:
        show(donut(eda.production_by(t, "Season"), "Season", "Production", "Season-wise production share"))

    a, b = st.columns(2)
    with a:
        d = (t.groupby(["State_Name", "District_Name"], observed=True)["Production"].sum().nlargest(10).reset_index())
        d["Label"] = d["District_Name"].astype(str) + " (" + d["State_Name"].astype(str) + ")"
        show(hbar(d, "Production", "Label", "Top 10 districts by production (tonnes)"), height=420)
    with b:
        sample = t[t["Area"] > 0].sample(min(4000, len(t)), random_state=1)
        top6 = eda.production_by(t, "Crop", 6)["Crop"].tolist()
        sample["Crop_Group"] = sample["Crop"].astype(str).where(sample["Crop"].isin(top6), "Other crops")
        fig = px.scatter(sample, x="Area", y="Production", color="Crop_Group", log_x=True, log_y=True, opacity=.55,
                         hover_data=["State_Name", "District_Name", "Crop_Year"],
                         title="Area vs production (random sample of 4,000 records, log scales)")
        fig.update_layout(xaxis_title="Area (ha)", yaxis_title="Production (t)")
        show(fig, height=420)
    with st.expander("Download filtered data"):
        csv_button(f.drop(columns=["District_Key"]), "krishipredict_filtered.csv", key="ov_dl")


pages = [
    st.Page(overview, title="Overview", icon="🏠", default=True, url_path="overview"),
    st.Page("pages/1_Agricultural_Analytics.py", title="Agricultural Analytics", icon="📊"),
    st.Page("pages/2_Crop_Yield_Prediction.py", title="Crop Yield Prediction", icon="🤖"),
    st.Page("pages/3_District_Intelligence.py", title="District Intelligence", icon="📍"),
    st.Page("pages/4_Rainfall_Intelligence.py", title="Rainfall Intelligence", icon="🌧️"),
    st.Page("pages/5_Crop_Comparison.py", title="Crop Comparison", icon="⚖️"),
    st.Page("pages/6_Model_Performance.py", title="Model Performance", icon="📈"),
    st.Page("pages/7_About.py", title="About the Project", icon="ℹ️"),
]
st.navigation(pages).run()
