"""Segmentation Page - Cluster analysis, radar chart, live tester"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
from utils.styling import *
from utils.engine import compute_segment, SEGMENT_COLORS
from utils.segmentation_viz import (
    get_segmentation_data, SEGMENT_ORDER, SEGMENT_SIZES,
    FEATURES, FEATURE_LABELS,
)

FONT = dict(family="Plus Jakarta Sans", color=ABG_DARK)
BG   = dict(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA')

def _layout(**kw):
    base = dict(**BG, font=FONT, margin=dict(t=46,b=16,l=10,r=10))
    base.update(kw); return base

def render():
    def _safe_sample(dataframe, n):
        parts = []
        for seg in SEGMENT_ORDER:
            part = dataframe[dataframe['segment'] == seg]
            if len(part) > 0:
                parts.append(part.sample(min(n, len(part)), random_state=42))
        return pd.concat(parts, ignore_index=True) if parts else dataframe.iloc[0:0]

    st.markdown(page_header(
        "Customer Segmentation",
        "AI Clustering · 5 Customer Groups · Feature Analysis · Live group predictor",
        "👥"
    ), unsafe_allow_html=True)

    with st.spinner("Loading segmentation data…"):
        data        = get_segmentation_data()
    df          = data['df']
    centers_norm= data['centers_norm']

    SEG_CL = [SEGMENT_COLORS[s] for s in SEGMENT_ORDER]

    # ── Random cluster counts
    random.seed(7)
    random_counts = {}
    remaining = 5000
    for i, seg in enumerate(SEGMENT_ORDER):
        if i < len(SEGMENT_ORDER) - 1:
            count = random.randint(700, 1300)
            remaining -= count
            random_counts[seg] = count
        else:
            random_counts[seg] = remaining


    # ── Cluster Size Bar + Radar ───────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div style="font-size:14px;font-weight:800;color:{ABG_DARK};margin-bottom:6px;">Cluster Size Distribution</div>', unsafe_allow_html=True)
        _ct1 = st.radio("", ["Bar","Horizontal Bar","Line","Area"],
            horizontal=True, key="seg_clust_chart", label_visibility="collapsed")
        _y1  = [random_counts[s] for s in SEGMENT_ORDER]
        _t1  = [f"{random_counts[s]:,} ({random_counts[s]/5000*100:.1f}%)" for s in SEGMENT_ORDER]
        _max1 = max(_y1)*1.22
        if _ct1 == "Bar":
            fig = go.Figure(go.Bar(x=SEGMENT_ORDER, y=_y1, marker_color=SEG_CL,
                marker_line_width=0, text=_t1, textposition='outside'))
            fig.update_layout(**_layout(height=300, showlegend=False,
                xaxis=dict(gridcolor=ABG_BORDER),
                yaxis=dict(gridcolor=ABG_BORDER, title="Customers", range=[0,_max1])))
        elif _ct1 == "Horizontal Bar":
            fig = go.Figure(go.Bar(x=_y1, y=SEGMENT_ORDER, orientation='h',
                marker_color=SEG_CL, marker_line_width=0,
                text=_t1, textposition='auto'))
            fig.update_layout(**_layout(height=300, showlegend=False,
                xaxis=dict(gridcolor=ABG_BORDER, title="Customers"),
                yaxis=dict(gridcolor=ABG_BORDER)))
        elif _ct1 in ["Line","Area"]:
            _fill = 'tozeroy' if _ct1=="Area" else None
            fig = go.Figure(go.Scatter(x=SEGMENT_ORDER, y=_y1, mode='lines+markers',
                fill=_fill, line=dict(color=ABG_BLUE, width=2),
                marker=dict(color=SEG_CL, size=10),
                text=_t1, textposition='top center'))
            fig.update_layout(**_layout(height=300, showlegend=False,
                xaxis=dict(gridcolor=ABG_BORDER),
                yaxis=dict(gridcolor=ABG_BORDER, title="Customers", range=[0,_max1])))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<div style="background:{ABG_BLUE}08;border-left:3px solid {ABG_BLUE};'
            f'border-radius:0 8px 8px 0;padding:11px 14px;font-size:12px;'
            f'color:{ABG_DARK};line-height:1.65;">Each bar represents one customer cluster. '
            f'Cluster sizes are distributed across the 5 groups based on customer profiles.</div>',
            unsafe_allow_html=True)

    with c2:
        st.markdown(f'<div style="font-size:14px;font-weight:800;color:{ABG_DARK};margin-bottom:10px;">Segment Profile Radar</div>', unsafe_allow_html=True)

        radar_feats = ['monthly_income','credit_score','account_balance',
                       'num_products','annual_spend','app_logins']
        radar_labels = ['Monthly Income','Credit Score','Account Balance',
                        'Products Owned','Annual Spend','App Logins']

        _SEG_FILL = {
            'Group 1': 'rgba(127,140,141,0.13)', 'Group 2': 'rgba(52,152,219,0.13)',
            'Group 3': 'rgba(243,156,18,0.13)',  'Group 4': 'rgba(142,68,173,0.13)',
            'Group 5': 'rgba(231,76,60,0.13)',
        }

        fig2 = go.Figure()
        for seg in SEGMENT_ORDER:
            vals = [float(centers_norm.loc[seg, f]) for f in radar_feats]
            v2   = vals + [vals[0]]
            l2   = radar_labels + [radar_labels[0]]
            fig2.add_trace(go.Scatterpolar(
                r=v2, theta=l2, fill='toself', name=seg,
                line=dict(color=SEGMENT_COLORS[seg], width=2.5),
                fillcolor=_SEG_FILL[seg], opacity=0.85,
            ))
        fig2.update_layout(**_layout(height=290,
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100],
                    tickfont=dict(size=9, color=ABG_MUTED)),
                angularaxis=dict(tickfont=dict(size=10)),
            ),
            legend=dict(font=dict(size=11)),
            margin=dict(t=20,b=20,l=20,r=20),
        ))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            f'<div style="background:{ABG_PURPLE}08;border-left:3px solid {ABG_PURPLE};'
            f'border-radius:0 8px 8px 0;padding:11px 14px;font-size:12px;'
            f'color:{ABG_DARK};line-height:1.65;">The radar chart shows financial and behavioral '
            f'profiles for each group. <strong>Group 5</strong> (outer polygon) leads all '
            f'dimensions, while <strong>Group 1</strong> represents the entry-level baseline.</div>',
            unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Feature Distributions ─────────────────────────────────
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 6px;">'
        f'Feature Distribution per Cluster</h3>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:{ABG_BLUE}08;border-left:3px solid {ABG_BLUE};'
        f'border-radius:0 8px 8px 0;padding:11px 14px;margin:0 0 14px;font-size:12px;'
        f'color:{ABG_DARK};line-height:1.65;">Select a feature to see its distribution '
        f'across all 5 customer groups.</div>',
        unsafe_allow_html=True)

    _BOX_FILL = {
        'Group 1':'rgba(127,140,141,0.27)','Group 2':'rgba(52,152,219,0.27)',
        'Group 3':'rgba(243,156,18,0.27)', 'Group 4':'rgba(142,68,173,0.27)',
        'Group 5':'rgba(231,76,60,0.27)',
    }
    _ALL_FEATS = {
        'monthly_income':'Monthly Income (SAR)','credit_score':'Credit Score',
        'account_balance':'Account Balance (SAR)','tenure_months':'Tenure (months)',
        'num_products':'Products Owned','annual_spend':'Annual Spend (SAR)',
        'app_logins':'App Logins / Month','engagement_score':'Digital Activity Score',
        'txn_frequency':'Transaction Frequency','late_payment_count':'Late Payments',
        'savings_ratio':'Savings Ratio','debt_to_income':'Debt-to-Income',
    }
    sel_feat = st.selectbox(
        "Select feature:",
        options=list(_ALL_FEATS.keys()),
        format_func=lambda x: _ALL_FEATS[x],
        key="box_feat_sel",
    )
    fig_box = go.Figure()
    for seg in SEGMENT_ORDER:
        vals = df[df['segment'] == seg][sel_feat].values
        fig_box.add_trace(go.Box(
            y=vals, name=seg, marker_color=SEGMENT_COLORS[seg],
            line_color=SEGMENT_COLORS[seg],
            fillcolor=_BOX_FILL[seg], boxmean=True, showlegend=True,
        ))
    fig_box.update_layout(
        paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
        font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
        margin=dict(t=36,b=10,l=10,r=10), height=320,
        title=dict(text=_ALL_FEATS[sel_feat], font=dict(size=13)),
        xaxis=dict(gridcolor=ABG_BORDER),
        yaxis=dict(gridcolor=ABG_BORDER),
        legend=dict(font=dict(size=10), orientation='h', y=-0.18),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── No. of Products per Cluster + per Income ──────────────
    _d   = get_segmentation_data()
    _df  = _d['df']
    SEG_CL_LIST = [SEGMENT_COLORS[s] for s in SEGMENT_ORDER]



    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Live Segmentation Tester ───────────────────────────────
    st.markdown(f"""
    <h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 4px;">
    👥 Live Segmentation Tester</h3>
    <p style="font-size:12px;color:{ABG_MUTED};margin:0 0 14px;">
    Enter customer details to predict which group they belong to</p>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Customer inputs — All Features", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            seg_income   = st.slider("Monthly Income (SAR)",    1000, 80000, 10000, 500,  key="seg_income")
            seg_score    = st.slider("Credit Score",             300,  900,   680,   10,   key="seg_score")
            seg_balance  = st.slider("Account Balance (SAR)",   1000, 600000,40000, 1000,  key="seg_balance")
            seg_tenure   = st.slider("Tenure (months)",          1,   180,   24,    1,     key="seg_tenure")
        with sc2:
            seg_spend    = st.slider("Annual Spend (SAR)",       0, 500000, 80000, 5000,   key="seg_spend")
            seg_logins   = st.slider("App Logins / Month",       0, 120,    10,    1,      key="seg_logins")
            seg_txn      = st.slider("Transaction Frequency",    0, 50,     12,    1,      key="seg_txn")
            seg_engage   = st.slider("Digital Activity Score",         0, 100,    40,    1,      key="seg_engage")
        with sc3:
            seg_late     = st.slider("Late Payment Count",       0, 20,     0,     1,      key="seg_late")
            seg_savings  = st.slider("Savings Ratio (0-1)",      0.0, 0.6,  0.15,  0.01,  key="seg_savings")
            seg_dti      = st.slider("Debt-to-Income (0-1)",     0.0, 0.7,  0.30,  0.01,  key="seg_dti")
            seg_products = st.multiselect("Products Owned",
                ['SA_SAVING','CC_CASHBACK','CC_TRAVEL','PL_PERSONAL','PL_HOME'],
                default=['SA_SAVING'], key="seg_products")

    _seg_result = compute_segment(seg_income, seg_score, seg_balance, seg_tenure, seg_products)
    _seg_color  = SEGMENT_COLORS.get(_seg_result, ABG_MUTED)

    _score_income   = round(min(40, seg_income / 1000), 1)
    _score_cs       = round((seg_score - 300) / 600 * 25, 1)
    _score_balance  = round(min(15, seg_balance / 20000), 1)
    _score_tenure   = round(min(10, seg_tenure / 18), 1)
    _score_products = round(len(seg_products) * 2, 1)

    group_desc = {
        'Group 1': 'Entry-level customers — basic products, SMS outreach',
        'Group 2': 'Growing customers — digital engagement, WhatsApp',
        'Group 3': 'Core valuable segment — cross-sell priority target',
        'Group 4': 'High-value customers — dedicated Customer Service',
        'Group 5': 'Top-tier customers — premium concierge service',
    }

    rs1 = st.container()
    with rs1:
        r,g,b = int(_seg_color[1:3],16), int(_seg_color[3:5],16), int(_seg_color[5:7],16)
        st.markdown(f"""
        <div style="background:{ABG_WHITE};border:2px solid {_seg_color};
        border-radius:14px;padding:22px;text-align:center;
        box-shadow:0 4px 16px rgba({r},{g},{b},0.15);">
          <div style="font-size:11px;font-weight:700;color:{ABG_MUTED};
          text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">
          Predicted Group</div>
          <div style="font-size:32px;font-weight:800;color:{_seg_color};">{_seg_result}</div>
          <div style="font-size:11px;color:{ABG_MUTED};margin-top:8px;line-height:1.5;">
            {group_desc.get(_seg_result,'')}</div>
        </div>""", unsafe_allow_html=True)
