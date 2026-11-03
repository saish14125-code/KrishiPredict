"""Shared UI helpers (styling, KPI cards, filter bar, formatting, downloads)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

CSS = f"""
<style>
.stApp {{ background: {C.BG}; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {C.GREEN_DARK} 0%, #0b3a1f 100%); }}
[data-testid="stSidebar"] * {{ color: #ecfdf5 !important; }}
[data-testid="stSidebarNav"] a, [data-testid="stSidebar"] a {{ border-radius: 8px; }}
.kp-hero {{ background: linear-gradient(120deg, {C.GREEN_DARK}, {C.GREEN} 60%, #15803d); color: #fff; padding: 22px 28px;
  border-radius: 16px; margin-bottom: 18px; box-shadow: 0 4px 14px rgba(20,83,45,.25); }}
.kp-hero h1 {{ color: #fff; font-size: 1.7rem; margin: 0; padding: 0; }}
.kp-hero p {{ color: #d1fae5; margin: 4px 0 0 0; font-size: .98rem; }}
.kpi {{ background: #fff; border-radius: 14px; padding: 14px 16px; box-shadow: 0 1px 6px rgba(0,0,0,.08);
  border-left: 5px solid {C.GREEN_MID}; height: 100%; }}
.kpi .l {{ font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; color: #6b7280; }}
.kpi .v {{ font-size: 1.55rem; font-weight: 700; color: {C.GREEN_DARK}; line-height: 1.25; }}
.kpi .s {{ font-size: .75rem; color: #6b7280; }}
.card {{ background: #fff; border-radius: 14px; padding: 16px 20px; box-shadow: 0 1px 6px rgba(0,0,0,.08); margin-bottom: 12px; }}
.note {{ background: #fefce8; border-left: 4px solid #ca8a04; border-radius: 8px; padding: 10px 14px; font-size: .88rem; color: #422006; margin: 6px 0 12px 0; }}
.insight {{ background: #fff; border-left: 4px solid {C.GREEN_MID}; border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }}
.sec {{ color: {C.GREEN_DARK}; font-weight: 700; font-size: 1.05rem; margin: 8px 0 2px 0; }}
div[data-testid="stPlotlyChart"] {{ background: #fff; border-radius: 14px; padding: 6px; box-shadow: 0 1px 6px rgba(0,0,0,.08); }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        logo = C.ASSETS_DIR / "logo.png"
        if logo.exists():
            st.image(str(logo), width=70)
        st.markdown("### KrishiPredict")
        st.caption("India's agricultural intelligence & crop-yield prediction system")


def page_header(title: str, subtitle: str) -> None:
    inject_css()
    st.markdown(f'<div class="kp-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>',
                 unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def section(text: str) -> None:
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def empty_state(msg: str = "No data matches the current filters. Widen the filters to see results.") -> None:
    st.info(msg)


def fmt(x, unit: str = "") -> str:
    """Compact number: 1.23 B / 45.6 M / 7.8 K."""
    if x is None or pd.isna(x):
        return "n/a"
    a = abs(x)
    for lim, s in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if a >= lim:
            return f"{x / lim:,.2f} {s}{(' ' + unit) if unit else ''}"
    return f"{x:,.2f}{(' ' + unit) if unit else ''}" if a < 100 else f"{x:,.0f}{(' ' + unit) if unit else ''}"


def csv_button(df: pd.DataFrame, filename: str, label: str = "Download CSV", key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", key=key)


def style_fig(fig, height: int = 380, legend: bool = True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Segoe UI, Arial", size=12, color="#1f2937"), colorway=C.PALETTE,
                      title=dict(font=dict(size=15, color=C.GREEN_DARK)), showlegend=legend,
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor="#eef2ee", zeroline=False)
    fig.update_yaxes(gridcolor="#eef2ee", zeroline=False)
    return fig


def show(fig, **kw) -> None:
    st.plotly_chart(style_fig(fig, **kw), width="stretch")


def filter_bar(df: pd.DataFrame, key: str, default_crops: list[str] | None = None, crop_multi: bool = True):
    """Top-of-page filters. Empty multiselect = all. Returns (filtered_df, selections)."""
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1.4])
    states = c1.multiselect("State", sorted(df["State_Name"].cat.categories), key=f"{key}_s", placeholder="All states")
    crops_all = sorted(df["Crop"].cat.categories)
    crops = c2.multiselect("Crop", crops_all, default=default_crops, key=f"{key}_c", placeholder="All crops")
    seasons = c3.multiselect("Season", sorted(df["Season"].cat.categories), key=f"{key}_se", placeholder="All seasons")
    y0, y1 = int(df["Crop_Year"].min()), int(df["Crop_Year"].max())
    years = c4.slider("Year", y0, y1, (y0, y1), key=f"{key}_y")
    m = df["Crop_Year"].between(*years)
    if states:
        m &= df["State_Name"].isin(states)
    if crops:
        m &= df["Crop"].isin(crops)
    if seasons:
        m &= df["Season"].isin(seasons)
    return df[m], {"states": states, "crops": crops, "seasons": seasons, "years": years}


UNIT_NOTE = ("Production is in <b>tonnes</b>, area in <b>hectares</b>, yield in <b>tonnes/hectare</b> (units assumed from the "
             "official statistics convention; the files carry no unit labels). Coconut (nuts) and Cotton/Jute/Mesta (bales) "
             "are <b>excluded from tonnage totals</b> because they are not recorded in tonnes.")
