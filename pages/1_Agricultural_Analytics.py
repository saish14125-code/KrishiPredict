import plotly.express as px
import streamlit as st

from src import eda
from src.data_loader import DataMissingError, load_crop
from src.utils import UNIT_NOTE, csv_button, empty_state, filter_bar, note, page_header, section, show
from src.visualizations import hbar, lines

page_header("Agricultural Analytics", "Production, yield and season analysis with insights generated from your filters")
try:
    df = load_crop()
except DataMissingError as e:
    st.error(str(e)); st.stop()

f, sel = filter_bar(df, "aa")
if f.empty:
    empty_state(); st.stop()
note(UNIT_NOTE)
t_prod, t_yield, t_season, t_ins = st.tabs(["Crop production", "Yield analysis", "Season analysis", "Insights"])

with t_prod:
    t = eda.tonne(f)
    if t.empty:
        empty_state("No tonne-measured crops in this selection.")
    else:
        a, b = st.columns(2)
        with a:
            show(hbar(eda.production_by(t, "Crop", 15), "Production", "Crop", "Total production by crop (top 15, tonnes)"), height=470)
        with b:
            show(lines(eda.yearly_production(t), "Crop_Year", "Production", title="Annual production (tonnes)"), height=470)
        a, b = st.columns(2)
        with a:
            show(hbar(eda.production_by(t, "State_Name", 15), "Production", "State_Name", "State-wise production (top 15)"), height=470)
        with b:
            d = t.groupby(["State_Name", "District_Name"], observed=True)["Production"].sum().nlargest(15).reset_index()
            d["Label"] = d["District_Name"].astype(str) + " (" + d["State_Name"].astype(str) + ")"
            show(hbar(d, "Production", "Label", "District-wise production (top 15)"), height=470)
        top = eda.production_by(t, "Crop", 5)["Crop"].tolist()
        ar = f[f["Crop"].isin(top)].groupby(["Crop_Year", "Crop"], observed=True)["Area"].sum().reset_index()
        show(lines(ar, "Crop_Year", "Area", "Crop", "Cultivated area trend, top 5 crops (hectares)"))

with t_yield:
    crops = sorted(f["Crop"].cat.remove_unused_categories().cat.categories)
    ycrops = [c for c in crops if c not in __import__("config").NON_TONNE_CROPS]
    if not ycrops:
        empty_state("No tonne-measured crops in this selection.")
    else:
        crop = st.selectbox("Crop for yield analysis", ycrops, index=ycrops.index("Rice") if "Rice" in ycrops else 0, key="aa_ycrop",
                            help="Yields are only comparable within one crop, so this tab analyses one crop at a time.")
        note("Yield = Production / Area, area-weighted (total production / total area) over records that pass quality checks. "
             "Zero-production records, areas under 1 ha and extreme high outliers are excluded from yield statistics (they stay in the data).")
        cf = f[f["Crop"] == crop]
        yv = cf[cf["Yield_Valid"]]
        if yv.empty:
            empty_state("No valid yield records for this crop in the current filters.")
        else:
            a, b = st.columns(2)
            with a:
                avg = eda.weighted_yield(f[f["Yield_Valid"] & ~f["Crop"].isin(__import__("config").NON_TONNE_CROPS)], "Crop", 30).nlargest(15, "Yield")
                show(hbar(avg, "Yield", "Crop", "Average yield by crop (t/ha, crops with >= 30 records)", text_fmt=".2f"), height=470)
            with b:
                sy = eda.weighted_yield(cf, "State_Name", 10).sort_values("Yield", ascending=False)
                show(hbar(sy.head(15), "Yield", "State_Name", f"{crop}: yield by state (t/ha, top 15, >= 10 records)", text_fmt=".2f"), height=470)
            show(lines(eda.weighted_yield(cf, "Crop_Year"), "Crop_Year", "Yield", title=f"{crop}: yield trend over time (t/ha)"))
            dy = eda.weighted_yield(cf, ["State_Name", "District_Name"], 5)
            dy["Region"] = dy["District_Name"].astype(str) + " (" + dy["State_Name"].astype(str) + ")"
            a, b = st.columns(2)
            with a:
                show(hbar(dy.nlargest(10, "Yield"), "Yield", "Region", f"{crop}: highest-yielding districts", text_fmt=".2f"), height=420)
            with b:
                show(hbar(dy.nsmallest(10, "Yield"), "Yield", "Region", f"{crop}: lowest-yielding districts", text_fmt=".2f",
                          scale=[[0, "#fde68a"], [1, "#b45309"]]), height=420)
            st.caption("Districts need >= 5 valid records. Low yields can reflect crop failure, partial reporting or unit/reporting issues - treat extremes with care.")
            csv_button(dy.drop(columns=["Region"]), f"{crop}_district_yield.csv", "Download district yield table", key="aa_dl1")

with t_season:
    t = eda.tonne(f)
    if t.empty:
        empty_state("No tonne-measured crops in this selection.")
    else:
        a, b = st.columns(2)
        with a:
            top8 = eda.production_by(t, "Crop", 8)["Crop"].tolist()
            sc = t[t["Crop"].isin(top8)].groupby(["Season", "Crop"], observed=True)["Production"].sum().reset_index()
            fig = px.bar(sc, x="Season", y="Production", color="Crop", title="Production by season and crop (top 8 crops, tonnes)")
            show(fig, height=420)
        with b:
            ys = t.groupby(["Crop_Year", "Season"], observed=True)["Production"].sum().reset_index()
            show(lines(ys, "Crop_Year", "Production", "Season", "Season production over time (tonnes)"), height=420)
        ycrops = [c for c in sorted(f["Crop"].cat.remove_unused_categories().cat.categories) if c not in __import__("config").NON_TONNE_CROPS]
        if ycrops:
            crop_s = st.selectbox("Crop for seasonal yield", ycrops, index=ycrops.index("Rice") if "Rice" in ycrops else 0, key="aa_scrop")
            sy = eda.weighted_yield(f[f["Crop"] == crop_s], "Season", 5)
            if sy.empty:
                empty_state("Not enough valid records for seasonal yield.")
            else:
                show(hbar(sy, "Yield", "Season", f"{crop_s}: yield by season (t/ha)", text_fmt=".2f"), height=300)

with t_ins:
    ycrops = [c for c in sorted(f["Crop"].cat.remove_unused_categories().cat.categories) if c not in __import__("config").NON_TONNE_CROPS]
    focus = st.selectbox("Focus crop (for state / district insights)", ["Auto"] + ycrops, key="aa_focus")
    with st.spinner("Computing insights..."):
        ins = eda.generate_insights(f, None if focus == "Auto" else focus)
    for s in ins:
        st.markdown(f'<div class="insight">{s}</div>', unsafe_allow_html=True)
    st.caption("Insights are computed from the current filters, not hard-coded. 'Improving' = positive linear trend of yearly yield "
               "(districts with >= 8 years and >= 100 ha average area). Declines compare the same districts only.")
    tr = eda.crop_trend_table(f)
    if not tr.empty:
        with st.expander("Like-for-like crop production change table"):
            st.dataframe(tr.round(1), width="stretch", hide_index=True)
            csv_button(tr, "crop_trend_table.csv", key="aa_dl2")
