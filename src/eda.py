"""Aggregations and data-driven insights. All results derive from the (filtered) data passed in."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tonne(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Production_Unit"] == "tonnes"]


def production_by(df: pd.DataFrame, col: str, top: int | None = None) -> pd.DataFrame:
    g = tonne(df).groupby(col, observed=True)["Production"].sum().sort_values(ascending=False)
    g = g[g > 0].reset_index()
    return g.head(top) if top else g


def yearly_production(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    t = tonne(df)
    keys = ["Crop_Year"] + ([by] if by else [])
    return t.groupby(keys, observed=True)["Production"].sum().reset_index()


def weighted_yield(df: pd.DataFrame, by, min_records: int = 1) -> pd.DataFrame:
    """Area-weighted yield (sum production / sum area) over records valid for yield statistics."""
    v = df[df["Yield_Valid"]]
    g = v.groupby(by, observed=True).agg(Production=("Production", "sum"), Area=("Area", "sum"), Records=("Yield", "size"),
                                         Median_Yield=("Yield", "median"))
    g["Yield"] = g["Production"] / g["Area"]
    return g[g["Records"] >= min_records].reset_index()


def district_yield_slopes(df: pd.DataFrame, crop: str, min_years: int = 8, min_mean_area: float = 100) -> pd.DataFrame:
    """Per-district linear trend of area-weighted yearly yield for one crop (% of mean yield per year)."""
    v = df[(df["Crop"] == crop) & df["Yield_Valid"]]
    dy = v.groupby(["District_Key", "Crop_Year"], observed=True).agg(P=("Production", "sum"), A=("Area", "sum")).reset_index()
    dy = dy[dy["A"] > 0]
    dy["Y"] = dy["P"] / dy["A"]
    g = dy.groupby("District_Key", observed=True)
    dy["xm"] = g["Crop_Year"].transform("mean")
    dy["ym"] = g["Y"].transform("mean")
    dy["sxy"] = (dy["Crop_Year"] - dy["xm"]) * (dy["Y"] - dy["ym"])
    dy["sxx"] = (dy["Crop_Year"] - dy["xm"]) ** 2
    out = dy.groupby("District_Key", observed=True).agg(Years=("Y", "size"), Mean_Yield=("Y", "mean"), Mean_Area=("A", "mean"),
                                                         sxy=("sxy", "sum"), sxx=("sxx", "sum")).reset_index()
    out = out[(out["Years"] >= min_years) & (out["sxx"] > 0) & (out["Mean_Area"] >= min_mean_area)]
    out["Slope_t_ha_per_yr"] = out["sxy"] / out["sxx"]
    out["Trend_pct_per_yr"] = 100 * out["Slope_t_ha_per_yr"] / out["Mean_Yield"]
    return out.drop(columns=["sxy", "sxx"]).sort_values("Trend_pct_per_yr", ascending=False)


def crop_trend_table(df: pd.DataFrame, window: int = 3, min_districts: int = 10) -> pd.DataFrame:
    """Like-for-like production change: only districts reporting the crop in BOTH the first and last `window` years,
    so that growth in the number of reporting districts does not masquerade as growth in production."""
    t = tonne(df)
    years = sorted(t["Crop_Year"].unique())
    if len(years) < 2 * window:
        return pd.DataFrame()
    first, last = years[:window], years[-window:]
    a = t[t["Crop_Year"].isin(first)].groupby(["Crop", "District_Key"], observed=True)["Production"].sum()
    b = t[t["Crop_Year"].isin(last)].groupby(["Crop", "District_Key"], observed=True)["Production"].sum()
    j = pd.concat([a.rename("First"), b.rename("Last")], axis=1, join="inner")
    j = j[(j["First"] > 0) & (j["Last"] > 0)].reset_index()
    out = j.groupby("Crop", observed=True).agg(Districts=("District_Key", "nunique"), First=("First", "sum"), Last=("Last", "sum")).reset_index()
    out = out[out["Districts"] >= min_districts]
    out["Change_pct"] = 100 * (out["Last"] - out["First"]) / out["First"]
    out["First_years"] = f"{first[0]}-{first[-1]}"
    out["Last_years"] = f"{last[0]}-{last[-1]}"
    return out.sort_values("Change_pct")


def generate_insights(df: pd.DataFrame, focus_crop: str | None = None) -> list[str]:
    """Plain-language observations computed from the supplied (already filtered) data."""
    out: list[str] = []
    t = tonne(df)
    if t.empty:
        return ["No tonne-denominated records in the current selection, so production insights cannot be computed."]
    crops = production_by(t, "Crop")
    if not crops.empty:
        top = crops.iloc[0]
        out.append(f"**Highest-production crop:** {top['Crop']} with {top['Production']/1e6:,.1f} M tonnes "
                   f"({100*top['Production']/crops['Production'].sum():.0f}% of tonnage), cumulative over the selected years.")
    fc = focus_crop or ("Rice" if (t["Crop"] == "Rice").any() else (crops.iloc[0]["Crop"] if not crops.empty else None))
    if fc:
        st_ = production_by(t[t["Crop"] == fc], "State_Name", 1)
        if not st_.empty:
            out.append(f"**Top {fc} state:** {st_.iloc[0]['State_Name']} ({st_.iloc[0]['Production']/1e6:,.2f} M tonnes).")
        wy = weighted_yield(df[df["Crop"] == fc], "State_Name", min_records=30)
        if len(wy) >= 2:
            hi, lo = wy.sort_values("Yield").iloc[-1], wy.sort_values("Yield").iloc[0]
            out.append(f"**{fc} yield spread:** {hi['State_Name']} leads at {hi['Yield']:.2f} t/ha versus {lo['State_Name']} at {lo['Yield']:.2f} t/ha "
                       f"(states with >= 30 valid records).")
        sl = district_yield_slopes(df, fc)
        if len(sl) >= 5:
            best = sl.iloc[0]
            out.append(f"**Most improving {fc} district:** {best['District_Key']} (about {best['Trend_pct_per_yr']:+.1f}% of its mean yield per year; "
                       f"{int(best['Years'])} years of data). Share of {fc} districts with a rising trend: {100*(sl['Slope_t_ha_per_yr']>0).mean():.0f}%.")
    seas = production_by(t, "Season")
    if not seas.empty:
        s = seas.iloc[0]
        out.append(f"**Leading season:** {s['Season']} contributes {100*s['Production']/seas['Production'].sum():.0f}% of tonnage "
                   f"(sugarcane and other long-duration crops are recorded under 'Whole Year').")
    tr = crop_trend_table(df)
    if not tr.empty:
        dec = tr.head(3)
        txt = "; ".join(f"{r.Crop} ({r.Change_pct:+.0f}%)" for r in dec.itertuples())
        out.append(f"**Largest like-for-like production declines** ({tr.iloc[0]['First_years']} vs {tr.iloc[0]['Last_years']}, same districts only): {txt}. Large swings can also reflect changes in what districts report.")
    else:
        out.append("Trend comparisons need at least 6 years in the selected range.")
    return out
