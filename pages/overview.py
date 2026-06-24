"""Overview Page - Cards, Charts, Modules, Table, Pipeline"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import random
from utils.styling import *

random.seed(42)

GRP_COLORS = {'Group 1':'#7F8C8D','Group 2':'#3498DB','Group 3':'#F39C12',
              'Group 4':'#8E44AD','Group 5':'#E74C3C'}

def _gen_customers(n=50):
    segs = ['Group 1','Group 2','Group 3','Group 4','Group 5']
    products_list = ['SA_SAVING','CC_CASHBACK','CC_TRAVEL','PL_PERSONAL','PL_HOME']
    channels = ['SMS','WhatsApp','Send Notification','Customer Service','Branch Agent']
    activities = ['Login','Transfer','Purchase','Inquiry','Deposit']
    rows = []
    inc_r = {'Group 1':(3000,7000),'Group 2':(7000,12000),'Group 3':(12000,20000),
             'Group 4':(20000,35000),'Group 5':(35000,80000)}
    for i in range(n):
        seg = random.choice(segs)
        lo, hi = inc_r[seg]
        inc   = round(random.uniform(lo, hi), 0)
        si    = segs.index(seg)
        rows.append({
            'Customer ID':           f'C{10000+i}',
            'Cluster':               seg,
            'Monthly Income (SAR)':  inc,
            'Account Balance (SAR)': round(inc * random.uniform(1.5, 8), 0),
            'Credit Score':          random.randint(540 + si*50, min(900, 680 + si*50)),
            'Tenure (months)':       random.randint(6*(si+1), 12*(si+2)),
            'Annual Spend (SAR)':    round(inc * random.uniform(0.5, 2.5), 0),
            'App Logins/Month':      random.randint(si*4, si*18+3),
            'Transaction Frequency': random.randint(3, 40),
            'Late Payments':         random.randint(0, max(0, 4-si)),
            'Products Owned':        random.randint(1, si+2),
            'Digital Activity Score':random.randint(10+si*12, 30+si*15),
            'Savings Ratio':         round(random.uniform(0.05+si*0.04, 0.15+si*0.06), 2),
            'Debt-to-Income':        round(random.uniform(0.10, 0.50-si*0.05), 2),
            'Recommended Product':   random.choice(products_list),
            'Best Channel':          channels[min(si, len(channels)-1)],
        })
    return pd.DataFrame(rows)

def render():
    st.markdown(page_header(
        "BSF AI Banking Intelligence System",
        "AI-powered platform for customer segmentation, cross-selling, and channel optimization · Developed by ABG",
        "🏦"
    ), unsafe_allow_html=True)

    # ── KPI Cards ──────────────────────────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5)
    for col,(ico,lbl,val,sub,color) in zip([k1,k2,k3,k4,k5],[
        ("👥","Total Customers","5,000","Synthetic BSF profiles",ABG_BLUE),
        ("🎯","No. of Clusters","5","AI-identified groups",ABG_PURPLE),
        ("🏦","BSF Products","5","Available for cross-sell",ABG_GREEN),
        ("📡","Contact Channels","5","SMS to Customer Service",ABG_GOLD),
        ("🤖","AI Modules","4","Seg · NBA · Channel · Chatbot",ABG_RED),
    ]):
        col.markdown(
            f'<div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};'
            f'border-radius:12px;padding:16px;border-top:3px solid {color};">'
            f'<div style="font-size:20px;">{ico}</div>'
            f'<div style="font-size:10px;color:{ABG_MUTED};font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.06em;margin:4px 0 2px;">{lbl}</div>'
            f'<div style="font-size:22px;font-weight:800;color:{color};">{val}</div>'
            f'<div style="font-size:10px;color:{ABG_MUTED};">{sub}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3 Pie Charts ──────────────────────────────────────────
    _ov_chart = st.radio("Chart type for distributions:",
        ["Pie","Bar","Horizontal Bar"], horizontal=True, key="ov_dist_chart")
    pc1, pc2, pc3 = st.columns(3)

    # Data for all 3 charts
    _seg_labels = ['Group 1','Group 2','Group 3','Group 4','Group 5']
    _seg_vals   = [1480,1250,1100,760,410]
    _seg_cols   = list(GRP_COLORS.values())

    _prod_labels = ['Savings Account','Cashback Card','Travel Card','Personal Finance','Home Finance']
    _prod_vals   = [3820,1540,680,920,410]
    _prod_cols   = [ABG_GOLD,ABG_BLUE,ABG_BLUE,ABG_GREEN,ABG_GREEN]

    _ch_labels = ['SMS','WhatsApp','Send Notification','Customer Service','Branch Agent']
    _ch_vals   = [1480,2100,840,330,250]
    _ch_cols   = ['#7F8C8D','#25D366',ABG_BLUE,ABG_PURPLE,ABG_GOLD]

    def _make_chart(labels, vals, colors, title, key_suffix):
        if _ov_chart == "Pie":
            fig = go.Figure(go.Pie(
                labels=labels, values=vals,
                marker=dict(colors=colors, line=dict(color='white', width=2)),
                hole=0.52, textinfo='label+percent',
                textfont=dict(size=10, family="Plus Jakarta Sans"),
            ))
            fig.update_layout(paper_bgcolor=ABG_WHITE,
                font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
                margin=dict(t=46,b=10,l=10,r=10), height=270,
                title=dict(text=title, font=dict(size=13)),
                legend=dict(font=dict(size=9)), annotations=[])
        elif _ov_chart == "Horizontal Bar":
            fig = go.Figure(go.Bar(x=vals, y=labels, orientation='h',
                marker_color=colors, marker_line_width=0,
                text=[f"{v:,}" for v in vals], textposition='auto'))
            fig.update_layout(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
                font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
                margin=dict(t=46,b=10,l=10,r=10), height=270, showlegend=False,
                title=dict(text=title, font=dict(size=13)),
                xaxis=dict(gridcolor='#E8E8F0', title="Count"),
                yaxis=dict(gridcolor='#E8E8F0'))
        else:  # Bar
            fig = go.Figure(go.Bar(x=labels, y=vals,
                marker_color=colors, marker_line_width=0,
                text=[f"{v:,}" for v in vals], textposition='outside'))
            _mx = max(vals)*1.22
            fig.update_layout(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
                font=dict(family="Plus Jakarta Sans", color=ABG_DARK),
                margin=dict(t=46,b=10,l=10,r=10), height=270, showlegend=False,
                title=dict(text=title, font=dict(size=13)),
                xaxis=dict(gridcolor='#E8E8F0', tickangle=-15),
                yaxis=dict(gridcolor='#E8E8F0', title="Count", range=[0,_mx]))
        return fig

    with pc1:
        st.plotly_chart(_make_chart(_seg_labels, _seg_vals, _seg_cols,
            "Customer Cluster Distribution","seg"), use_container_width=True)
    with pc2:
        st.plotly_chart(_make_chart(_prod_labels, _prod_vals, _prod_cols,
            "Product Ownership Distribution","prod"), use_container_width=True)
    with pc3:
        st.plotly_chart(_make_chart(_ch_labels, _ch_vals, _ch_cols,
            "Best Channel Distribution","ch"), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI Modules ────────────────────────────────────────────
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 14px;">🤖 AI Modules</h3>',
        unsafe_allow_html=True)
    modules = [
        ("👥","Module 1 — Customer Segmentation",ABG_BLUE,
         "Groups 5,000 BSF customers into 5 distinct clusters using AI Clustering based on financial behavior, digital engagement, and relationship depth."),
        ("🎯","Module 2 — Cross-Sell Recommendations",ABG_PURPLE,
         "Predicts which BSF product each customer is most likely to purchase next at the individual customer level — not by cluster."),
        ("📡","Module 3 — Best Channel Selection",ABG_GOLD,
         "Determines the optimal communication channel for each individual customer based on their own features, not their group."),
        ("🤖","Module 4 — Chatbot",ABG_GREEN,
         "Conversational AI assistant that answers product queries and guides customers through BSF offerings in real-time."),
    ]
    c1, c2 = st.columns(2)
    for i, (icon, title, color, desc) in enumerate(modules):
        col = c1 if i % 2 == 0 else c2
        col.markdown(
            f'<div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};'
            f'border-radius:14px;padding:20px;border-top:4px solid {color};'
            f'margin-bottom:14px;">'
            f'<div style="font-size:26px;margin-bottom:8px;">{icon}</div>'
            f'<div style="font-size:13px;font-weight:800;color:{ABG_DARK};margin-bottom:8px;">{title}</div>'
            f'<div style="font-size:12px;color:{ABG_MUTED};line-height:1.7;">{desc}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Customer Activity Table with Filter ───────────────────
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 14px;">'
        f'📋 Customer Data — AI Pipeline Output</h3>',
        unsafe_allow_html=True)

    df = _gen_customers(50)

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_cluster = st.multiselect("Filter by Cluster",
            options=['Group 1','Group 2','Group 3','Group 4','Group 5'],
            default=[], key="ov_cluster_filter",
            placeholder="All clusters")
    with fc2:
        sel_product = st.multiselect("Filter by Recommended Product",
            options=sorted(df['Recommended Product'].unique()),
            default=[], key="ov_product_filter",
            placeholder="All products")
    with fc3:
        sel_channel = st.multiselect("Filter by Best Channel",
            options=sorted(df['Best Channel'].unique()),
            default=[], key="ov_channel_filter",
            placeholder="All channels")

    filtered = df.copy()
    if sel_cluster: filtered = filtered[filtered['Cluster'].isin(sel_cluster)]
    if sel_product: filtered = filtered[filtered['Recommended Product'].isin(sel_product)]
    if sel_channel: filtered = filtered[filtered['Best Channel'].isin(sel_channel)]

    st.markdown(
        f'<div style="font-size:11px;color:{ABG_MUTED};margin-bottom:8px;">'
        f'Showing {len(filtered)} of {len(df)} customers</div>',
        unsafe_allow_html=True)

    # Column selector
    all_cols = ['Customer ID','Cluster','Monthly Income (SAR)','Account Balance (SAR)',
                'Credit Score','Tenure (months)','Annual Spend (SAR)','App Logins/Month',
                'Transaction Frequency','Late Payments','Products Owned',
                'Digital Activity Score','Savings Ratio','Debt-to-Income',
                'Recommended Product','Best Channel']
    default_cols = ['Customer ID','Cluster','Monthly Income (SAR)','Credit Score',
                    'Products Owned','Recommended Product','Best Channel']
    sel_cols = st.multiselect("Select columns to display:",
        options=all_cols, default=default_cols, key="ov_col_sel")
    if not sel_cols:
        sel_cols = default_cols

    def color_cluster(val):
        color = GRP_COLORS.get(val, '#000')
        return f'color: {color}; font-weight: 700'

    show_df = filtered[sel_cols]
    if 'Cluster' in sel_cols:
        st.dataframe(show_df.style.applymap(color_cluster, subset=['Cluster']),
            use_container_width=True, height=350)
    else:
        st.dataframe(show_df, use_container_width=True, height=350)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline Flow ─────────────────────────────────────────
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 16px;">🔄 AI Pipeline Flow</h3>',
        unsafe_allow_html=True)
    steps = [
        ("📦","Raw Data","Customer profiles, transactions, and digital behavior collected from BSF systems",ABG_MUTED),
        ("👥","Segmentation","Customers grouped into 5 clusters using AI Clustering based on financial and behavioral features",ABG_BLUE),
        ("🎯","Cross-Sell","AI predicts the best product for each individual customer — not by cluster",ABG_PURPLE),
        ("📡","Best Channel","Selects optimal channel per customer based on individual digital behavior and preferences",ABG_GOLD),
        ("🤖","Chatbot","AI assistant guides customers through BSF products in real-time",ABG_GREEN),
    ]
    arrow_cols = st.columns([1,0.12,1,0.12,1,0.12,1,0.12,1])
    step_cols  = [arrow_cols[0],arrow_cols[2],arrow_cols[4],arrow_cols[6],arrow_cols[8]]
    for col, (ico, title, desc, color) in zip(step_cols, steps):
        col.markdown(
            f'<div style="background:{ABG_WHITE};border:1.5px solid {color}44;'
            f'border-radius:14px;padding:16px;border-top:4px solid {color};min-height:140px;">'
            f'<div style="font-size:24px;margin-bottom:6px;">{ico}</div>'
            f'<div style="font-size:12px;font-weight:800;color:{color};margin-bottom:6px;">{title}</div>'
            f'<div style="font-size:10px;color:{ABG_MUTED};line-height:1.6;">{desc}</div>'
            f'</div>', unsafe_allow_html=True)
    for idx in [1,3,5,7]:
        arrow_cols[idx].markdown(
            f'<div style="display:flex;align-items:center;justify-content:center;'
            f'height:140px;font-size:22px;color:{ABG_BORDER};">→</div>',
            unsafe_allow_html=True)
