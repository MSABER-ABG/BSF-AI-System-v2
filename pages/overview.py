"""Overview Page - Pipeline flow at bottom, data analysis in features table"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
from utils.styling import *

random.seed(42)
np.random.seed(42)

def _hex_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip('#')
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

def _gen_sample_data(n=500):
    segs = ['Group 1','Group 2','Group 3','Group 4','Group 5']
    inc_r = {'Group 1':(3000,7000),'Group 2':(7000,12000),'Group 3':(12000,20000),
              'Group 4':(20000,35000),'Group 5':(35000,80000)}
    rows = []
    for seg in segs:
        lo,hi = inc_r[seg]
        for _ in range(n//5):
            inc = random.uniform(lo,hi)
            rows.append({
                'Group': seg,
                'Monthly Income':     round(inc,0),
                'Account Balance':    round(inc * random.uniform(1.5,8),0),
                'Credit Score':       random.randint(int(550+(segs.index(seg)*50)), min(900,int(680+(segs.index(seg)*50)))),
                'Tenure (months)':    random.randint(6*(segs.index(seg)+1), 12*(segs.index(seg)+2)),
                'App Logins/Month':   random.randint(segs.index(seg)*5, segs.index(seg)*20+5),
                'Products Owned':     random.randint(1, segs.index(seg)+2),
            })
    return pd.DataFrame(rows)

def render():
    st.markdown(page_header(
        "BSF AI Banking Intelligence System",
        "AI-powered platform for customer segmentation, cross-selling, and channel optimization · Developed by ABG",
        "🏦"
    ), unsafe_allow_html=True)

    # ── KPI Strip ──────────────────────────────────────────────
    k1,k2,k3,k4 = st.columns(4)
    for col,(ico,lbl,val,sub,color) in zip([k1,k2,k3,k4],[
        ("👥","Total Customers","5,000","Synthetic BSF profiles",ABG_BLUE),
        ("🏦","BSF Products","5","Available for cross-sell",ABG_PURPLE),
        ("📡","Contact Channels","5","SMS · WhatsApp · Send Notification · Customer Service · Branch",ABG_GOLD),
        ("🤖","AI Modules","4","Seg · Cross-sell · Channel · Chatbot",ABG_GREEN),
    ]):
        col.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
        border-radius:12px;padding:18px 16px;border-top:3px solid {color};
        box-shadow:0 2px 6px rgba(0,0,0,0.05);">
          <div style="font-size:22px;margin-bottom:6px;">{ico}</div>
          <div style="font-size:11px;color:{ABG_MUTED};font-weight:600;
          text-transform:uppercase;letter-spacing:0.06em;">{lbl}</div>
          <div style="font-size:24px;font-weight:800;color:{color};">{val}</div>
          <div style="font-size:11px;color:{ABG_MUTED};margin-top:2px;">{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Features Table + Data Analysis ───────────────────────
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 14px;">📋 Customer Features — Data Analysis</h3>', unsafe_allow_html=True)

    df = _gen_sample_data(500)

    # Stats per feature
    features = [
        ("Monthly Income (SAR)",      "Financial",   "Core driver for segmentation and product eligibility"),
        ("Account Balance (SAR)",     "Financial",   "Reflects savings behavior and financial stability"),
        ("Credit Score (300–900)",    "Financial",   "Determines credit eligibility and risk level"),
        ("Annual Spend (SAR)",        "Financial",   "Total yearly spending through BSF channels"),
        ("Tenure (months)",           "Relationship","Length of customer relationship with BSF"),
        ("Products Owned",            "Relationship","Number of current BSF products in portfolio"),
        ("App Logins / Month",        "Digital",     "Frequency of mobile app usage — digital engagement"),
        ("Transaction Frequency",     "Behavioral",  "Number of monthly transactions across all channels"),
        ("Preferred Channel",         "Digital",     "Customer's preferred contact method"),
        ("Late Payment Count",        "Risk",        "Historical count of late payments — credit risk signal"),
        ("Savings Ratio",             "Financial",   "Proportion of monthly income saved"),
        ("Employment Type",           "Financial",   "Government / Private / Self-employed"),
    ]
    cat_colors = {"Financial":ABG_BLUE,"Relationship":ABG_PURPLE,
                  "Behavioral":ABG_GOLD,"Digital":ABG_GREEN,"Risk":ABG_RED}

    rows_html = ""
    for feat_name, cat, desc in features:
        cc = cat_colors.get(cat, ABG_MUTED)
        rows_html += f"""
        <tr>
          <td style="padding:9px 14px;font-size:12px;font-weight:600;
          color:{ABG_DARK};border-bottom:1px solid {ABG_BORDER};">{feat_name}</td>
          <td style="padding:9px 14px;border-bottom:1px solid {ABG_BORDER};">
            <span style="background:{cc}15;color:{cc};padding:2px 10px;
            border-radius:12px;font-size:10px;font-weight:700;">{cat}</span></td>
          <td style="padding:9px 14px;font-size:12px;color:{ABG_MUTED};
          border-bottom:1px solid {ABG_BORDER};">{desc}</td>
        </tr>"""

    st.markdown(f"""
    <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
    border-radius:12px;overflow:hidden;box-shadow:0 2px 6px rgba(0,0,0,0.05);">
      <table style="width:100%;border-collapse:collapse;">
        <thead><tr style="background:#F8F8FF;">
          <th style="padding:10px 14px;font-size:11px;font-weight:700;color:{ABG_MUTED};text-align:left;text-transform:uppercase;letter-spacing:0.07em;width:30%;">Feature</th>
          <th style="padding:10px 14px;font-size:11px;font-weight:700;color:{ABG_MUTED};text-align:left;text-transform:uppercase;letter-spacing:0.07em;width:16%;">Category</th>
          <th style="padding:10px 14px;font-size:11px;font-weight:700;color:{ABG_MUTED};text-align:left;text-transform:uppercase;letter-spacing:0.07em;">Description</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)

    # ── AI Modules (no Coming Soon, no Key Questions) ─────────
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 14px;">🤖 AI Modules</h3>', unsafe_allow_html=True)
    modules = [
        ("👥","Module 1 — Customer Segmentation",ABG_BLUE,
         "Groups 5,000 BSF customers into 5 distinct clusters based on financial behavior, digital engagement, and relationship depth. Each group receives a tailored banking strategy."),
        ("🎯","Module 2 — Cross-Sell Recommendations",ABG_PURPLE,
         "Predicts which BSF product each customer is most likely to purchase next. Five classifiers — one per product — rank recommendations by buy probability."),
        ("📡","Module 3 — Best Channel Selection",ABG_GOLD,
         "Determines the optimal communication channel for each customer based on their individual data — ensuring the right message reaches the right person through the right channel."),
        ("🤖","Module 4 — Chatbot",ABG_GREEN,
         "Conversational AI assistant that interacts with customers, answers product queries, and guides them through BSF offerings — integrated with segmentation and recommendation outputs."),
    ]
    c1, c2 = st.columns(2)
    for i, (icon, title, color, desc) in enumerate(modules):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
        border-radius:14px;padding:22px;border-top:4px solid {color};
        margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.05);">
          <div style="font-size:30px;margin-bottom:10px;">{icon}</div>
          <div style="font-size:14px;font-weight:800;color:{ABG_DARK};margin-bottom:10px;">{title}</div>
          <div style="font-size:12px;color:{ABG_MUTED};line-height:1.7;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline Flow (at bottom) ─────────────────────────────
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 16px;">🔄 AI Pipeline Flow</h3>', unsafe_allow_html=True)
    steps = [
        ("📦","Raw Data","Customer profiles, transactions, and digital behavior data collected from BSF systems",ABG_MUTED),
        ("👥","Segmentation","Customers are grouped into 5 segments based on income, balance, credit score, and behavior",ABG_BLUE),
        ("🎯","Cross-Sell","AI predicts which BSF product each customer is most likely to buy next",ABG_PURPLE),
        ("📡","Best Channel","Selects the optimal channel (SMS, WhatsApp, Send Notification, Customer Service, Branch)",ABG_GOLD),
        ("🤖","Chatbot","Conversational AI assistant answers queries and guides customers through BSF products",ABG_GREEN),
    ]
    arrow_cols = st.columns([1,0.12,1,0.12,1,0.12,1,0.12,1])
    step_cols  = [arrow_cols[0],arrow_cols[2],arrow_cols[4],arrow_cols[6],arrow_cols[8]]
    for col, (ico, title, desc, color) in zip(step_cols, steps):
        col.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {color}44;
        border-radius:14px;padding:18px;border-top:4px solid {color};
        box-shadow:0 2px 8px rgba(0,0,0,0.06);min-height:150px;">
          <div style="font-size:26px;margin-bottom:6px;">{ico}</div>
          <div style="font-size:12px;font-weight:800;color:{color};margin-bottom:6px;">{title}</div>
          <div style="font-size:10px;color:{ABG_MUTED};line-height:1.6;">{desc}</div>
        </div>""", unsafe_allow_html=True)
    for idx in [1,3,5,7]:
        arrow_cols[idx].markdown(f"""
        <div style="display:flex;align-items:center;justify-content:center;
        height:100%;min-height:150px;padding-top:40px;">
          <div style="font-size:22px;color:{ABG_BORDER};">→</div>
        </div>""", unsafe_allow_html=True)
