import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as C
from src import rainfall_analysis as R
from src.data_loader import DataMissingError, load_crop, load_rain_annual, load_rain_daily
from src.utils import csv_button, empty_state, fmt, kpi, note, page_header, section, show
from src.visualizations import hbar

page_header("Rainfall Intelligence", "Long-term subdivision rainfall (1901-2015) and current district monitoring (Aug-Nov 2026) - analysed separately")
t_hist, t_cur, t_link = st.tabs(["Historical rainfall (subdivisions)", "Current district monitoring", "Rainfall & crops"])

# ------------------------------------------------------------------ historical
with t_hist:
    try:
        ann = load_rain_annual()
    except DataMissingError as e:
        st.warning(str(e)); ann = None
    if ann is not None:
        note("These are <b>IMD meteorological subdivisions</b> (36 regions), <b>not districts</b> and not always states. Annual totals in mm; "
             "anomalies are measured against each subdivision's own 1901-2015 mean.")
        subs = sorted(ann["Rainfall_Subdivision"].unique())
        pick = st.multiselect("Subdivisions (empty = all)", subs, key="rh_sub", placeholder="All subdivisions")
        yr = st.slider("Years", int(ann["Year"].min()), int(ann["Year"].max()), (1901, 2015), key="rh_y")
        a = ann[ann["Year"].between(*yr)]
        if pick:
            a = a[a["Rainfall_Subdivision"].isin(pick)]
        if a.empty:
            empty_state()
        else:
            ser = R.mean_series(a)
            summ = R.subdivision_summary(a)
            k = st.columns(4)
            wet, dry = ser.loc[ser["Annual_Rainfall_mm"].idxmax()], ser.loc[ser["Annual_Rainfall_mm"].idxmin()]
            kpi(k[0], "Mean annual rainfall", f"{a['Annual_Rainfall_mm'].mean():,.0f} mm", "all selected subdivision-years")
            kpi(k[1], "Wettest year (mean)", f"{int(wet['Year'])}", f"{wet['Annual_Rainfall_mm']:,.0f} mm")
            kpi(k[2], "Driest year (mean)", f"{int(dry['Year'])}", f"{dry['Annual_Rainfall_mm']:,.0f} mm")
            slope = pd.Series(summ["Trend_mm_per_decade"]).median()
            kpi(k[3], "Median trend", f"{slope:+.1f} mm/decade", "across subdivisions")
            fig = go.Figure()
            fig.add_scatter(x=ser["Year"], y=ser["Annual_Rainfall_mm"], name="Annual mean", line=dict(color="#86efac"))
            fig.add_scatter(x=ser["Year"], y=ser["Rolling_10y"], name="10-year rolling mean", line=dict(color=C.GREEN_DARK, width=3))
            fig.update_layout(title="Annual rainfall (unweighted mean of selected subdivisions, mm)", yaxis_title="mm")
            show(fig)
            x, y = st.columns(2)
            with x:
                s2 = summ.rename(columns={"Subdivision": "S"})
                show(hbar(s2.head(36), "Mean_mm", "S", "Mean annual rainfall by subdivision (mm)", text_fmt=",.0f"), height=820)
            with y:
                an = R.add_anomaly(a)
                gm = an.groupby("Year")["Anomaly_pct"].mean().reset_index()
                fig = px.bar(gm, x="Year", y="Anomaly_pct", color="Anomaly_pct", color_continuous_scale="RdYlBu", color_continuous_midpoint=0,
                             title="Rainfall anomaly vs long-term mean (%, mean over selection)")
                fig.update_layout(coloraxis_showscale=False)
                show(fig, height=400)
                yoy = px.bar(ser.dropna(subset=["YoY_change_pct"]), x="Year", y="YoY_change_pct", title="Year-on-year change (%)")
                yoy.update_traces(marker_color=C.GREEN_MID)
                show(yoy, height=400)
            hm = an.pivot_table(index="Rainfall_Subdivision", columns=(an["Year"] // 10 * 10), values="Anomaly_pct", aggfunc="mean")
            fig = px.imshow(hm, aspect="auto", color_continuous_scale="RdYlBu", color_continuous_midpoint=0,
                            title="Decadal mean anomaly (%) by subdivision", labels=dict(color="%", x="Decade"))
            show(fig, height=760)
            with st.expander("Subdivision statistics table"):
                st.dataframe(summ.round(1), hide_index=True, width="stretch")
                csv_button(summ, "subdivision_rainfall_summary.csv", key="rh_dl")

# ------------------------------------------------------------------ current
with t_cur:
    try:
        dly = load_rain_daily()
    except DataMissingError as e:
        st.warning(str(e)); dly = None
    if dly is not None:
        d0, d1 = dly["Date"].min().date(), dly["Date"].max().date()
        miss = sorted(set(pd.date_range(d0, d1).date) - set(dly["Date"].dt.date))
        note(f"<b>District-level, daily, {d0} to {d1}</b> (separate from the historical data above). The file has only daily actual, daily normal and the IMD category; "
             "weekly/monthly figures here are <b>computed by summing non-overlapping daily values</b>. "
             f"Days with no observation are skipped{(' - dates absent from the file: ' + ', '.join(map(str, miss))) if miss else ''}; "
             "partial weeks/months are labelled.")
        c1, c2 = st.columns([1, 2])
        states = sorted(dly["State"].unique())
        st_pick = c1.multiselect("State", states, key="rc_st", placeholder="All states")
        rng = c2.slider("Date window", d0, d1, (d0, d1), key="rc_dt")
        w = R.window(dly, *rng)
        if st_pick:
            w = w[w["State"].isin(st_pick)]
        if w.empty:
            empty_state()
        else:
            ds = R.district_summary(w)
            ss = R.state_summary(ds)
            k = st.columns(4)
            kpi(k[0], "Districts", f"{len(ds)}")
            kpi(k[1], "Mean rainfall / district", f"{ds['Actual_mm'].mean():,.0f} mm", f"normal {ds['Normal_mm'].mean():,.0f} mm")
            dep = 100 * (ds["Actual_mm"].sum() - ds["Normal_mm"].sum()) / ds["Normal_mm"].sum() if ds["Normal_mm"].sum() > 0 else float("nan")
            kpi(k[2], "Departure from normal", f"{dep:+.0f}%", "over window, all districts")
            kpi(k[3], "Max single-day", f"{ds['Max_daily_mm'].max():,.0f} mm")
            ts = R.daily_series(w)
            x, y = st.columns(2)
            with x:
                fig = go.Figure()
                fig.add_bar(x=ts["Date"], y=ts["Actual"], name="Actual", marker_color=C.GREEN_MID)
                fig.add_scatter(x=ts["Date"], y=ts["Normal"], name="Normal", line=dict(color="#b45309"))
                fig.update_layout(title="Daily rainfall: mean across districts (mm/day)", yaxis_title="mm")
                show(fig)
            with y:
                fig = go.Figure()
                fig.add_scatter(x=ts["Date"], y=ts["Cum_Actual"], name="Cumulative actual", line=dict(color=C.GREEN_DARK, width=3))
                fig.add_scatter(x=ts["Date"], y=ts["Cum_Normal"], name="Cumulative normal", line=dict(color="#b45309", dash="dash"))
                fig.update_layout(title="Cumulative rainfall since window start (mean per district, mm)", yaxis_title="mm")
                show(fig)
            x, y = st.columns(2)
            for col, key, label in [(x, "Week_Start", "Weekly"), (y, "Month", "Monthly")]:
                p = R.period_series(w, key)
                p["Label"] = p[key].astype(str) + p["Partial"].map({True: " *", False: ""})
                fig = go.Figure()
                fig.add_bar(x=p["Label"], y=p["Actual"], name="Actual", marker_color=C.GREEN_MID)
                fig.add_bar(x=p["Label"], y=p["Normal"], name="Normal", marker_color="#fcd34d")
                fig.update_layout(barmode="group", title=f"{label} rainfall per district (mean, mm)  * = partial period", xaxis_type="category")
                with col:
                    show(fig, height=340)
            x, y = st.columns(2)
            with x:
                show(hbar(ss.head(20), "Mean_district_actual_mm", "State", "States: mean rainfall per district (top 20, mm)", text_fmt=",.0f"), height=560)
            with y:
                fig = px.histogram(ds, x="Actual_mm", nbins=40, title="Distribution of district rainfall totals (mm)", color_discrete_sequence=[C.GREEN])
                show(fig, height=270)
                cat = w[w["Date"] == w["Date"].max()]["Category_Label"].value_counts().reset_index()
                cat.columns = ["Category", "Districts"]
                show(px.bar(cat, x="Category", y="Districts", title=f"IMD category on latest day ({w['Date'].max().date()})",
                            color_discrete_sequence=[C.GREEN_MID]), height=270)
            x, y = st.columns(2)
            ds["Label"] = ds["District"] + " (" + ds["State"] + ")"
            with x:
                show(hbar(ds.nlargest(10, "Actual_mm"), "Actual_mm", "Label", "Highest-rainfall districts (mm, window total)", text_fmt=",.0f"), height=420)
            with y:
                low = ds[ds["Days"] >= ds["Days"].max() * 0.8].nsmallest(10, "Actual_mm")
                show(hbar(low, "Actual_mm", "Label", "Lowest-rainfall districts (mm, window total)", text_fmt=",.0f",
                          scale=[[0, "#fde68a"], [1, "#b45309"]]), height=420)
            with st.expander("District table"):
                st.dataframe(ds.drop(columns=["Label"]).round(1), hide_index=True, width="stretch")
                csv_button(ds.drop(columns=["Label"]), "district_rainfall_summary.csv", key="rc_dl")

# ------------------------------------------------------------------ link
with t_link:
    note("<b>What can and cannot be shown.</b> The 2026 district rainfall cannot explain 1997-2022 yields (different years). Historical rainfall is by "
         "subdivision, not district. The only <b>valid</b> link is <b>state-level</b>: states that coincide with exactly one subdivision, for years 1997-2015 "
         "present in both datasets. It is exploratory and does not prove causation; annual rainfall ignores timing within the season, irrigation and inputs.")
    try:
        crop, ann = load_crop(), load_rain_annual()
    except DataMissingError as e:
        st.warning(str(e)); st.stop()
    c1, c2 = st.columns(2)
    states = [s for s in C.STATE_TO_SUBDIVISION if s in set(crop["State_Name"].astype(str))]
    stt = c1.selectbox("State (1:1 with a subdivision)", states, key="rl_s")
    cs = sorted(set(crop.loc[(crop["State_Name"] == stt) & ~crop["Crop"].isin(C.NON_TONNE_CROPS), "Crop"].astype(str)))
    cc = c2.selectbox("Crop", cs, index=cs.index("Rice") if "Rice" in cs else 0, key="rl_c")
    j, info = R.state_rain_crop_link(crop, ann, stt, cc)
    if "error" in info:
        empty_state(f"Cannot compute: {info['error']}.")
    else:
        k = st.columns(3)
        kpi(k[0], "Overlapping years", f"{info['n']}", info["years"])
        kpi(k[1], "Correlation (raw)", f"{info['r_raw']:+.2f}", "rainfall vs yield")
        kpi(k[2], "Correlation (detrended)", f"{info['r_detrended']:+.2f}", "both series minus linear trend")
        fig = px.scatter(j, x="Annual_Rainfall_mm", y="Yield", text="Crop_Year",
                         title=f"{cc} in {stt}: annual yield vs {info['subdivision'].title()} annual rainfall")
        fig.update_traces(textposition="top center", marker=dict(size=9, color=C.GREEN))
        import numpy as np
        m_, b_ = np.polyfit(j["Annual_Rainfall_mm"], j["Yield"], 1)
        xs = np.array([j["Annual_Rainfall_mm"].min(), j["Annual_Rainfall_mm"].max()])
        fig.add_scatter(x=xs, y=m_ * xs + b_, mode="lines", name="Linear fit", line=dict(color="#b45309", dash="dash"))
        fig.update_layout(xaxis_title="Annual rainfall (mm)", yaxis_title="Yield (t/ha)")
        show(fig)
        st.caption("Small samples (about 10-19 points). Reporting coverage within a state also changes over years, so treat the correlation as indicative only.")
