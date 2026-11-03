import plotly.express as px
import streamlit as st

import config as C
from src import eda
from src.data_loader import DataMissingError, load_crop
from src.utils import csv_button, empty_state, fmt, kpi, note, page_header, section, show
from src.visualizations import hbar, lines

page_header("District Intelligence", "District production, area and yield compared with the state")
try:
    df = load_crop()
except DataMissingError as e:
    st.error(str(e)); st.stop()

c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.2, 1.4, 1])
states = sorted(df["State_Name"].cat.categories)
state = c1.selectbox("State", states, index=states.index("Maharashtra") if "Maharashtra" in states else 0)
sdf = df[df["State_Name"] == state]
dists = sorted(sdf["District_Name"].cat.remove_unused_categories().unique())
district = c2.selectbox("District", dists)
ddf = sdf[sdf["District_Name"] == district]
crops = ["All crops"] + sorted(ddf["Crop"].cat.remove_unused_categories().unique())
crop = c3.selectbox("Crop", crops)
y0, y1 = int(ddf["Crop_Year"].min()), int(ddf["Crop_Year"].max())
years = c4.slider("Year range", int(df["Crop_Year"].min()), int(df["Crop_Year"].max()), (max(y0, int(df["Crop_Year"].max()) - 9), y1))
seasons = c5.multiselect("Season", sorted(ddf["Season"].cat.remove_unused_categories().unique()), placeholder="All")

m = ddf["Crop_Year"].between(*years)
if crop != "All crops":
    m &= ddf["Crop"] == crop
if seasons:
    m &= ddf["Season"].isin(seasons)
d = ddf[m]
if d.empty:
    empty_state("No records for this district with the chosen filters."); st.stop()
s_m = sdf["Crop_Year"].between(*years) & ((sdf["Crop"] == crop) if crop != "All crops" else True) & (sdf["Season"].isin(seasons) if seasons else True)
s = sdf[s_m]
t = eda.tonne(d)

k = st.columns(4)
kpi(k[0], "Total production", fmt(t["Production"].sum(), "t") if len(t) else "n/a", "tonne-measured crops")
kpi(k[1], "Cultivated area (sum of records)", fmt(d["Area"].sum(), "ha"), "multi-season rows add up")
v = d[d["Yield_Valid"]]
if crop != "All crops" and len(v):
    kpi(k[2], f"{crop} yield", f"{v['Production'].sum() / v['Area'].sum():.2f} t/ha", "area-weighted")
else:
    kpi(k[2], "Median record yield", f"{v['Yield'].median():.2f} t/ha" if len(v) else "n/a", "pick a crop for a weighted yield")
if len(t):
    rank = (eda.production_by(s, "District_Name")["District_Name"].astype(str).tolist())
    r = rank.index(str(district)) + 1 if str(district) in rank else None
    kpi(k[3], "Rank in state (production)", f"{r} of {len(rank)}" if r else "n/a", f"within {state}, same filters")
else:
    kpi(k[3], "Rank in state", "n/a")

note("District boundary data is not bundled, so no map is drawn (guessed coordinates would be inaccurate). Rankings and state comparisons are shown instead. "
     + ("Selected crop is not recorded in tonnes." if crop in C.NON_TONNE_CROPS else ""))

a, b = st.columns(2)
with a:
    if len(t):
        show(hbar(eda.production_by(t, "Crop", 10), "Production", "Crop", f"Top crops in {district} (tonnes)"), height=420)
    else:
        empty_state("No tonne-measured crops here.")
with b:
    if crop == "All crops":
        yp = eda.yearly_production(t)
        if len(yp):
            show(lines(yp, "Crop_Year", "Production", title=f"{district}: production trend (tonnes)"), height=420)
    else:
        yy = eda.weighted_yield(d, "Crop_Year")
        sy = eda.weighted_yield(s, "Crop_Year")
        if len(yy):
            yy["Level"], sy["Level"] = district, f"{state} (state)"
            import pandas as pd
            show(lines(pd.concat([yy, sy]), "Crop_Year", "Yield", "Level", f"{crop}: yield, district vs state (t/ha)"), height=420)

section("District vs state average yield, by crop")
dy = eda.weighted_yield(d, "Crop", 3).set_index("Crop")
sy = eda.weighted_yield(s, "Crop", 3).set_index("Crop")
cmp_ = dy[["Yield", "Production"]].join(sy[["Yield"]], rsuffix="_state", how="inner")
cmp_ = cmp_[~cmp_.index.isin(C.NON_TONNE_CROPS)]
if cmp_.empty:
    empty_state("Not enough valid records to compare against the state.")
else:
    cmp_["Difference_pct"] = 100 * (cmp_["Yield"] - cmp_["Yield_state"]) / cmp_["Yield_state"]
    cmp_ = cmp_.sort_values("Production", ascending=False).head(12).reset_index()
    fig = px.bar(cmp_, x="Crop", y="Difference_pct", color="Difference_pct", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                 title="Yield vs state average (%): positive = district above state (top 12 crops by production)")
    fig.update_layout(coloraxis_showscale=False, yaxis_title="% difference")
    show(fig, height=380)
    st.dataframe(cmp_.rename(columns={"Yield": "District yield", "Yield_state": "State yield", "Production": "District production (t)"}).round(2),
                 hide_index=True, width="stretch")

section(f"District ranking inside {state}")
rk_crop = crop if crop != "All crops" else None
rk = s if rk_crop is None else s[s["Crop"] == rk_crop]
if rk_crop:
    rr = eda.weighted_yield(rk, "District_Name", 3).sort_values("Yield", ascending=False)
    if len(rr):
        rr["Highlight"] = rr["District_Name"].astype(str).eq(str(district)).map({True: "Selected", False: "Other"})
        fig = px.bar(rr.head(25), x="District_Name", y="Yield", color="Highlight", color_discrete_map={"Selected": C.GREEN_DARK, "Other": "#bbf7d0"},
                     title=f"{rk_crop}: yield by district (t/ha, top 25 of {len(rr)})")
        show(fig, height=360)
        csv_button(rr.drop(columns=["Highlight"]), f"{state}_{rk_crop}_district_ranking.csv", key="di_dl")
else:
    rr = eda.production_by(eda.tonne(rk), "District_Name", 20)
    if len(rr):
        show(hbar(rr, "Production", "District_Name", f"Top 20 districts in {state} by production (tonnes)"), height=520)
        csv_button(rr, f"{state}_district_ranking.csv", key="di_dl")
