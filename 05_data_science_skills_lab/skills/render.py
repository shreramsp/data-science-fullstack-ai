"""Turn a SkillResult into a visual dashboard in Streamlit.

This module exists because of the project's follow-up requirement: executing a
skill live must produce a readable dashboard, not a raw JSON dump. Every block
kind has a visual renderer here — metric tiles, Altair charts, dataframes,
formatted narrative and downloadable document artifacts.
"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from skills.base import Block, SkillResult

_PALETTE = ["#4C6FBF", "#E08A3C", "#4F9D69", "#B0555E", "#7A63A8", "#5E9AA8"]


def _theme():
    return {"config": {
        "axis": {"labelColor": "#5c5f66", "titleColor": "#5c5f66",
                 "gridColor": "#e9ecef", "domainColor": "#dee2e6"},
        "legend": {"labelColor": "#5c5f66", "titleColor": "#5c5f66"},
        "range": {"category": _PALETTE},
        "view": {"stroke": "transparent"},
    }}


alt.themes.register("lab", _theme)
alt.themes.enable("lab")


def render_chart(block: Block) -> None:
    df: pd.DataFrame = block.payload
    meta = block.meta
    kind, x, y, color = meta["kind"], meta["x"], meta["y"], meta.get("color")
    enc_color = alt.Color(f"{color}:N", title=None) if color else alt.value(_PALETTE[0])

    if kind == "hbar":
        sort = alt.EncodingSortField(field=x, order="descending") if meta.get("sort") else None
        chart = alt.Chart(df).mark_bar(cornerRadius=2).encode(
            x=alt.X(f"{x}:Q", title=x.replace("_", " ")),
            y=alt.Y(f"{y}:N", sort=sort, title=None),
            color=enc_color,
            tooltip=list(df.columns[:6]),
        )
    elif kind == "bar":
        chart = alt.Chart(df).mark_bar(cornerRadius=2).encode(
            x=alt.X(f"{x}:N", title=x.replace("_", " "), sort=None),
            y=alt.Y(f"{y}:Q", title=y.replace("_", " ")),
            color=enc_color,
            tooltip=list(df.columns[:6]),
        )
    elif kind == "line":
        chart = alt.Chart(df).mark_line(point=len(df) <= 40, strokeWidth=2).encode(
            x=alt.X(f"{x}:{'T' if pd.api.types.is_datetime64_any_dtype(df[x]) else 'N' if df[x].dtype == object else 'Q'}",
                    title=x.replace("_", " ")),
            y=alt.Y(f"{y}:Q", title=y.replace("_", " "), scale=alt.Scale(zero=False)),
            color=enc_color,
            tooltip=list(df.columns[:6]),
        )
    elif kind == "area":
        chart = alt.Chart(df).mark_area(opacity=0.7).encode(
            x=alt.X(f"{x}:Q"), y=alt.Y(f"{y}:Q"), color=enc_color)
    else:  # scatter
        size = "customers" if "customers" in df.columns else None
        enc = dict(x=alt.X(f"{x}:Q", title=x.replace("_", " "),
                           scale=alt.Scale(zero=False)),
                   y=alt.Y(f"{y}:Q", title=y.replace("_", " "),
                           scale=alt.Scale(zero=False)),
                   color=enc_color, tooltip=list(df.columns[:6]))
        if size:
            enc["size"] = alt.Size(f"{size}:Q", title="customers")
        chart = alt.Chart(df).mark_circle(opacity=0.8).encode(**enc)

    st.altair_chart(chart.properties(height=300).interactive(), width="stretch")
    if meta.get("note"):
        st.caption(meta["note"])


def render_block(block: Block, key_prefix: str = "") -> None:
    if block.kind == "metrics":
        if block.title:
            st.markdown(f"**{block.title}**")
        cols = st.columns(len(block.payload))
        for col, metric in zip(cols, block.payload):
            with col:
                st.metric(metric.label, metric.value)
                if metric.help:
                    st.caption(metric.help)

    elif block.kind == "table":
        if block.title:
            st.markdown(f"**{block.title}**")
        st.dataframe(block.payload, width="stretch", hide_index=True)
        if block.meta.get("note"):
            st.caption(block.meta["note"])

    elif block.kind == "chart":
        if block.title:
            st.markdown(f"**{block.title}**")
        render_chart(block)

    elif block.kind == "narrative":
        if block.title:
            st.markdown(f"**{block.title}**")
        st.info(block.payload)

    elif block.kind == "code":
        if block.title:
            st.markdown(f"**{block.title}**")
        st.code(block.payload, language=block.meta.get("language", "python"))

    elif block.kind == "artifact":
        with st.container(border=True):
            st.markdown(block.payload)
        st.download_button(
            f"Download {block.meta['filename']}",
            block.payload, file_name=block.meta["filename"], mime="text/markdown",
            key=f"{key_prefix}-dl-{block.meta['filename']}",
        )


def render_result(result: SkillResult, key_prefix: str = "") -> None:
    st.success(result.headline)
    for i, block in enumerate(result.blocks):
        render_block(block, key_prefix=f"{key_prefix}-{i}")
        st.write("")
