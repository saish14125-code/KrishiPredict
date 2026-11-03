import pandas as pd
import plotly.express as px
import streamlit as st

import config as C
from src import eda
from src.data_loader import DataMissingError, load_crop
from src.utils import csv_button, empty_state, note, page_header, show
from src.visualizations import lines

page_header("Crop Comparison", "Compare two or more crops on production, area, yield, trends, states and seasons")
try:
    df = load_crop()
except DataMissingError as e:
    st.error(str(e)); st.stop()

allc = sorted(df["Crop"].cat.categories)
crops = st.multiselect("Crops to compare (select at least 2)", allc, default=[c for c in ["Rice", "Wheat", "Maize"] if c in allc], key="cc_c")
c1, c2, c3 = st.columns([1.3, 1.3, 1.6])
states = c1.multiselect("State", sorted(df["State_Name"].cat.categories), placeholder="All states", key="cc_s")
seasons = c2.multiselect("Season", sorted(df["Season"].cat.categories), placeholder="All seasons", key="cc_se")
yrs = c3.slider("Year", int(df["Crop_Year"].min()), int(df["Crop_Year"].max()), (int(df["Crop_Year"].min()), int(df["Crop_Year"].max())), key="cc_y")
if len(crops) < 2:
    empty_state("Select at least two crops to compare."); st.stop()

m = df["Crop"].isin(crops) & df["Crop_Year"].between(*yrs)
if states:
    m &= df["State_Name"].isin(states)
if seasons:
    m &= df["Season"].isin(seasons)
f = df[m]
if f.empty:
    empty_state(); st.stop()
mixed = [c for c in crops if c in C.NON_TONNE_CROPS]
if mixed:
    note("Production for " + ", ".join(f"{c} ({C.NON_TONNE_CROPS[c]})" for c in mixed) + " is not in tonnes, so its production must not be compared with tonne crops. "
         "Yield values for these crops are in their own units per hectare; production charts below show tonne crops only.")

v = f[f["Yield_Valid"]]
tab = f.groupby("Crop", observed=True).agg(Records=("Area", "size"), Districts=("District_Key", "nunique"), Area_ha=("Area", "sum"), Production=("Production", "sum")).reset_index()
w = v.groupby("Crop", observed=True).agg(P=("Production", "sum"), A=("Area", "sum")).reset_index()
w["Yield"] = w["P"] / w["A"]
tab = tab.merge(w[["Crop", "Yield"]], on="Crop", how="left")
tab["Unit"] = tab["Crop"].astype(str).map(C.NON_TONNE_CROPS).fillna("tonnes")
tab = tab.rename(columns={"Production": "Production (native unit)", "Yield": "Yield (unit/ha)"}).sort_values("Production (native unit)", ascending=False)
st.dataframe(tab.round(2), hide_index=True, width="stretch")
csv_button(tab, "crop_comparison_summary.csv", "Download comparison table", key="cc_dl1")

t = eda.tonne(f)
a, b = st.columns(2)
with a:
    if len(t):
        show(px.bar(eda.production_by(t, "Crop"), x="Crop", y="Production", title="Total production (tonnes)", color="Crop"), height=360, legend=False)
with b:
    show(px.bar(f.groupby("Crop", observed=True)["Area"].sum().reset_index(), x="Crop", y="Area", title="Cultivated area (hectares, summed over records)", color="Crop"), height=360, legend=False)
a, b = st.columns(2)
with a:
    if len(t):
        show(lines(eda.yearly_production(t, "Crop"), "Crop_Year", "Production", "Crop", "Production trend (tonnes)"))
with b:
    yy = eda.weighted_yield(v, ["Crop_Year", "Crop"])
    show(lines(yy, "Crop_Year", "Yield", "Crop", "Yield trend (unit/ha, area-weighted)"))
a, b = st.columns(2)
with a:
    sy = eda.weighted_yield(v, ["State_Name", "Crop"], 10)
    topst = sy.groupby("State_Name", observed=True)["Production"].sum().nlargest(10).index
    sy = sy[sy["State_Name"].isin(topst)]
    show(px.bar(sy, x="State_Name", y="Yield", color="Crop", barmode="group", title="Yield by state (10 largest producing states, >= 10 records)"), height=400)
with b:
    se = eda.weighted_yield(v, ["Season", "Crop"], 5)
    show(px.bar(se, x="Season", y="Yield", color="Crop", barmode="group", title="Yield by season (>= 5 records)"), height=400)
section_df = f.drop(columns=["District_Key"])
csv_button(section_df, "crop_comparison_filtered_records.csv", "Download filtered records", key="cc_dl2")
