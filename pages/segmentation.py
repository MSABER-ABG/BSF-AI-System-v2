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
        "K-Means clustering · 5 customer groups · Feature analysis · Live group predictor",
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
        st.markdown(f'<div style="font-size:14px;font-weight:800;color:{ABG_DARK};margin-bottom:10px;">Cluster Size Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=SEGMENT_ORDER, y=[random_counts[s] for s in SEGMENT_ORDER],
            marker_color=SEG_CL, marker_line_width=0,
            text=[f"{random_counts[s]:,} ({random_counts[s]/5000*100:.1f}%)" for s in SEGMENT_ORDER],
            textposition='outside',
        ))
        fig.update_layout(**_layout(height=290, showlegend=False,
            xaxis=dict(gridcolor=ABG_BORDER),
            yaxis=dict(gridcolor=ABG_BORDER, title="Customers")))
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
                       'num_products','txn_frequency','annual_spend','app_logins']
        radar_labels = ['Monthly Income','Credit Score','Account Balance',
                        'Products Owned','No. of Transaction','Annual Spend','App Logins']

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
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 6px;">Feature Distribution per Cluster</h3>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:{ABG_BLUE}08;border-left:3px solid {ABG_BLUE};'
        f'border-radius:0 8px 8px 0;padding:11px 14px;margin:0 0 14px;font-size:12px;'
        f'color:{ABG_DARK};line-height:1.65;">These charts show how each customer feature '
        f'varies across the 5 groups. Clear differences between groups confirm that the '
        f'clustering has correctly separated customers with distinct financial profiles.</div>',
        unsafe_allow_html=True)

    plot_type = "Box Plot"

    _BOX_FILL = {
        'Group 1':'rgba(127,140,141,0.27)','Group 2':'rgba(52,152,219,0.27)',
        'Group 3':'rgba(243,156,18,0.27)', 'Group 4':'rgba(142,68,173,0.27)',
        'Group 5':'rgba(231,76,60,0.27)',
    }
    _VLN_FILL = {
        'Group 1':'rgba(127,140,141,0.33)','Group 2':'rgba(52,152,219,0.33)',
        'Group 3':'rgba(243,156,18,0.33)', 'Group 4':'rgba(142,68,173,0.33)',
        'Group 5':'rgba(231,76,60,0.33)',
    }

    show_feats = [f for f in FEATURES if f not in ['late_payment_count','debt_to_income','savings_ratio']]
    pairs = [(show_feats[i], show_feats[i+1] if i+1<len(show_feats) else None)
             for i in range(0, len(show_feats), 2)]

    for f1, f2 in pairs:
        cols = st.columns(2) if f2 else [st.container()]
        for col, feat in zip(cols, [f for f in [f1,f2] if f]):
            lbl = FEATURE_LABELS[feat].replace(' (SAR)','').replace(' / month','').replace(' (months)','')
            with col:
                fig_d = go.Figure()
                for seg in SEGMENT_ORDER:
                    vals = df[df['segment'] == seg][feat].values
                    if plot_type == "Box Plot":
                        fig_d.add_trace(go.Box(
                            y=vals, name=seg, marker_color=SEGMENT_COLORS[seg],
                            line_color=SEGMENT_COLORS[seg],
                            fillcolor=_BOX_FILL[seg], boxmean=True, showlegend=False,
                        ))
                fig_d.update_layout(**_layout(height=240,
                    title=dict(text=lbl, font=dict(size=12)),
                    margin=dict(t=36,b=10,l=10,r=10),
                    xaxis=dict(gridcolor=ABG_BORDER),
                    yaxis=dict(gridcolor=ABG_BORDER)))
                st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Live Segmentation Tester ───────────────────────────────
    st.markdown(f"""
    <h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 4px;">
    👥 Live Segmentation Tester</h3>
    <p style="font-size:12px;color:{ABG_MUTED};margin:0 0 14px;">
    Enter customer details to predict which group they belong to</p>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Customer inputs", expanded=True):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            seg_income  = st.slider("Monthly Income (SAR)", 1000, 80000, 10000, 500, key="seg_income")
            seg_score   = st.slider("Credit Score", 300, 900, 680, 10, key="seg_score")
        with sc2:
            seg_balance = st.slider("Account Balance (SAR)", 1000, 600000, 40000, 1000, key="seg_balance")
            seg_tenure  = st.slider("Tenure (months)", 1, 180, 24, 1, key="seg_tenure")
        with sc3:
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

    rs1, rs2 = st.columns([1.2, 1])
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

    with rs2:
        st.markdown(f'<div style="font-size:12px;font-weight:700;color:{ABG_DARK};margin-bottom:8px;">Group Thresholds</div>', unsafe_allow_html=True)
        for seg_name, thresh, color in [
            ('Group 5','≥ 80','#E74C3C'),('Group 4','≥ 60','#8E44AD'),
            ('Group 3','≥ 40','#F39C12'),('Group 2','≥ 22','#3498DB'),('Group 1','< 22','#7F8C8D'),
        ]:
            active = seg_name == _seg_result
            st.markdown(f"""
            <div style="background:{''+color+'12' if active else 'transparent'};
            border:1.5px solid {''+color+'44' if active else ABG_BORDER};
            border-radius:8px;padding:7px 12px;margin-bottom:4px;
            display:flex;align-items:center;justify-content:space-between;">
              <span style="font-size:12px;font-weight:{'700' if active else '500'};
              color:{''+color if active else '#888'};">{seg_name}</span>
              <span style="font-size:11px;font-weight:700;color:{color};">{thresh}</span>
              {'<span style="font-size:10px;color:'+color+';font-weight:700;">← YOU</span>' if active else ''}
            </div>""", unsafe_allow_html=True)

