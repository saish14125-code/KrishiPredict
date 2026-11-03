"""Reusable Plotly chart builders (styling is applied by utils.show)."""
import plotly.express as px
import plotly.graph_objects as go

import config as C


def hbar(df, x, y, title, color=None, text_fmt=",.3s", scale=None):
    d = df.iloc[::-1]
    fig = px.bar(d, x=x, y=y, orientation="h", title=title, color=color, text=x,
                 color_continuous_scale=scale or [[0, "#bbf7d0"], [1, C.GREEN_DARK]])
    fig.update_traces(texttemplate=f"%{{text:{text_fmt}}}", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
    return fig


def lines(df, x, y, color=None, title="", markers=True):
    fig = px.line(df, x=x, y=y, color=color, title=title, markers=markers)
    return fig


def donut(df, names, values, title):
    fig = px.pie(df, names=names, values=values, hole=0.55, title=title, color_discrete_sequence=C.PALETTE)
    fig.update_traces(textinfo="percent+label", textposition="outside")
    return fig


def empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=16))
    return fig
