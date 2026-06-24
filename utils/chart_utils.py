"""
Shared chart utilities - chart type selector and universal bar/line builder
"""
import streamlit as st
import plotly.graph_objects as go

CHART_TYPES = ["Bar","Horizontal Bar","Line","Area"]

def chart_type_selector(key, label="Chart type:", default="Bar"):
    return st.radio(label, CHART_TYPES, horizontal=True,
                    index=CHART_TYPES.index(default), key=key)

def build_chart(chart_type, x, y, name=None, color=None, text=None,
                height=300, title="", x_title="", y_title="",
                paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA",
                legend=None, extra_traces=None):
    """Build a Plotly figure based on selected chart type."""
    FONT = dict(family="Plus Jakarta Sans")
    fig = go.Figure()

    if isinstance(y[0], list):
        # Multi-trace (stacked/grouped)
        names   = name or [f"Series {i}" for i in range(len(y))]
        colors  = color or [None]*len(y)
        for i, (yi, ni, ci) in enumerate(zip(y, names, colors)):
            kw = dict(name=ni, marker_color=ci) if ci else dict(name=ni)
            if chart_type == "Bar":
                t = text[i] if text and i < len(text) else None
                fig.add_trace(go.Bar(x=x, y=yi, text=t,
                    textposition='auto', **kw))
            elif chart_type == "Horizontal Bar":
                fig.add_trace(go.Bar(x=yi, y=x, orientation='h', **kw))
            elif chart_type in ["Line","Area"]:
                fill = 'tonexty' if chart_type=="Area" and i>0 else ('tozeroy' if chart_type=="Area" else None)
                fig.add_trace(go.Scatter(x=x, y=yi, mode='lines+markers',
                    fill=fill, **kw))
    else:
        kw = dict(marker_color=color) if color else {}
        txt = text
        if chart_type == "Bar":
            fig.add_trace(go.Bar(x=x, y=y, name=name or "", text=txt,
                textposition='auto', marker_line_width=0, **kw))
        elif chart_type == "Horizontal Bar":
            fig.add_trace(go.Bar(x=y, y=x, orientation='h', name=name or "",
                text=[f"{v:,}" for v in y] if not txt else txt,
                textposition='auto', marker_line_width=0, **kw))
        elif chart_type in ["Line","Area"]:
            fill = 'tozeroy' if chart_type == "Area" else None
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers',
                fill=fill, name=name or "", **kw))

    if extra_traces:
        for t in extra_traces:
            fig.add_trace(t)

    x_ax = dict(gridcolor='#E8E8F0', title=x_title) if chart_type != "Horizontal Bar" else dict(gridcolor='#E8E8F0', title=y_title)
    y_ax = dict(gridcolor='#E8E8F0', title=y_title) if chart_type != "Horizontal Bar" else dict(gridcolor='#E8E8F0', title=x_title)

    fig.update_layout(
        paper_bgcolor=paper_bgcolor, plot_bgcolor=plot_bgcolor, font=FONT,
        margin=dict(t=46,b=20,l=10,r=10), height=height,
        title=dict(text=title, font=dict(size=13)),
        xaxis=x_ax, yaxis=y_ax,
        legend=legend or dict(font=dict(size=10)),
        barmode='group' if isinstance(y[0] if y else None, list) else 'relative',
    )
    return fig
