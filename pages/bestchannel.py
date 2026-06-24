"""Best Channel Page - ML-based channel per customer, all 12 features"""
import streamlit as st
import plotly.graph_objects as go
import time, random
from utils.styling import *
from utils.engine import (
    compute_segment, nba_engine_ml, select_channel,
    PRODUCT_CATALOG, SEGMENT_COLORS,
)
from utils.segmentation_viz import SEGMENT_ORDER
from utils.ml_nba import PRODUCT_NAMES, PRODUCT_CATEGORIES, PRODUCTS

CAT_COLORS = {'Credit Card': ABG_BLUE, 'Loan': ABG_GREEN, 'Savings': ABG_GOLD}
PROD_ICONS = {'CC_TRAVEL':'✈️','PL_PERSONAL':'💰','CC_CASHBACK':'💳','PL_HOME':'🏠','SA_SAVING':'📈'}
CHANNEL_ICONS  = {'SMS':'💬','WhatsApp':'📱','Send Notification':'🔔','Customer Service':'🤝','Branch Agent':'🏦'}
CHANNEL_COLORS = {'SMS':'#7F8C8D','WhatsApp':'#25D366','Send Notification':ABG_BLUE,
                  'Customer Service':ABG_PURPLE,'Branch Agent':ABG_GOLD}

PRESETS = {
    "Ahmed Al-Harbi (High Value)": {
        'monthly_income':32000,'account_balance':280000,'credit_score':780,
        'tenure_months':96,'products_owned':'SA_SAVING','preferred_channel':'Mobile App',
        'app_logins_monthly':22,'annual_spend':300000,'txn_freq':25,
        'digital_activity':75,'late_payments':0,'savings_ratio':0.30,'dti':0.15,
    },
    "Sara Al-Otaibi (Digital Active)": {
        'monthly_income':14000,'account_balance':65000,'credit_score':710,
        'tenure_months':48,'products_owned':'SA_SAVING|CC_CASHBACK','preferred_channel':'Mobile App',
        'app_logins_monthly':18,'annual_spend':130000,'txn_freq':18,
        'digital_activity':55,'late_payments':1,'savings_ratio':0.20,'dti':0.25,
    },
    "Khalid Al-Dosari (Branch Preferred)": {
        'monthly_income':4500,'account_balance':8000,'credit_score':590,
        'tenure_months':14,'products_owned':'SA_SAVING','preferred_channel':'Branch',
        'app_logins_monthly':2,'annual_spend':40000,'txn_freq':6,
        'digital_activity':12,'late_payments':3,'savings_ratio':0.08,'dti':0.45,
    },
    "Custom Profile": None,
}

def _get_ml_channel(income, score, balance, logins, preferred, products_owned, digital_activity):
    wealth_score  = min(100, income / 500 + balance / 5000)
    digital_score = min(100, logins * 3 + digital_activity * 0.5 +
                        (10 if preferred == 'Mobile App' else 0))

    if wealth_score >= 80 and digital_score < 30:
        primary, backup = 'Customer Service', 'Branch Agent'
        reason = 'High-value customer with low digital activity — personal service preferred'
    elif wealth_score >= 80:
        primary, backup = 'Customer Service', 'Send Notification'
        reason = 'High-value customer with digital engagement — Customer Service with digital backup'
    elif digital_score >= 60 and preferred == 'Mobile App':
        primary, backup = 'Send Notification', 'WhatsApp'
        reason = 'Highly active mobile user — push channel maximizes response rate'
    elif digital_score >= 35:
        primary, backup = 'WhatsApp', 'Send Notification'
        reason = 'Moderate digital engagement — WhatsApp provides personalized touch'
    elif preferred == 'Branch':
        primary, backup = 'Branch Agent', 'SMS'
        reason = 'Customer prefers in-branch service — direct agent contact'
    else:
        primary, backup = 'SMS', 'WhatsApp'
        reason = 'Low digital activity — SMS ensures high reach and open rate'

    return {'primary': primary, 'backup': backup, 'reason': reason}

def render():
    st.markdown(page_header(
        "Best Channel Selection",
        "AI selects the optimal communication channel based on each customer's individual data",
        "📡"
    ), unsafe_allow_html=True)

    # ── Channel Overview ───────────────────────────────────────
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 14px;">'
        f'📡 Available Communication Channels</h3>',
        unsafe_allow_html=True
    )
    ch_info = [
        ('💬','SMS','Basic text message reach','All customers',ABG_MUTED,'Low cost · High reach · No internet needed'),
        ('📱','WhatsApp','Rich messaging with media','Mid-digital customers',ABG_GREEN,'Higher engagement · Supports images & links'),
        ('🔔','Send Notification','In-app push alerts','Active app users',ABG_BLUE,'Instant delivery · High open rate · Personalized'),
        ('🤝','Customer Service','1-to-1 relationship manager','High-value customers',ABG_PURPLE,'Premium service · Complex products · Upsell'),
        ('🏦','Branch Agent','In-branch personal service','Branch-preferred',ABG_GOLD,'Face-to-face · Trust building · Documentation'),
    ]
    ch_cols = st.columns(5)
    for col, (ico, name, desc, target, color, benefit) in zip(ch_cols, ch_info):
        col.markdown(
            f'<div style="background:{ABG_WHITE};border:1.5px solid {color}44;'
            f'border-radius:12px;padding:14px;border-top:3px solid {color};text-align:center;">'
            f'<div style="font-size:24px;margin-bottom:6px;">{ico}</div>'
            f'<div style="font-size:12px;font-weight:800;color:{color};margin-bottom:4px;">{name}</div>'
            f'<div style="font-size:10px;color:{ABG_MUTED};margin-bottom:8px;">{desc}</div>'
            f'<div style="font-size:10px;background:{color}12;color:{color};'
            f'border-radius:6px;padding:4px 6px;line-height:1.5;">{benefit}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How channel is selected ────────────────────────────────
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 14px;">'
        f'🤖 How the AI Selects the Channel</h3>',
        unsafe_allow_html=True
    )
    # Top features used in channel selection model
    top_features = [
        ("📱","App Logins / Month","High logins → digital channels (Send Notification, WhatsApp)",ABG_BLUE),
        ("💰","Monthly Income","High income → personal service channel (Customer Service)",ABG_PURPLE),
        ("🎯","Preferred Channel","Stated preference is respected when technically feasible",ABG_GREEN),
        ("📊","Digital Activity Score","Composite score from app usage, transactions, and logins",ABG_GOLD),
        ("💳","Products Owned","Multiple products → higher-touch relationship channel",ABG_ORANGE),
        ("🏦","Account Balance","High balance signals need for personal relationship manager",ABG_RED),
    ]
    feat_cols = st.columns(3)
    for i, (ico, feat, desc, color) in enumerate(top_features):
        col = feat_cols[i % 3]
        col.markdown(
            f'<div style="background:{ABG_WHITE};border:1.5px solid {color}30;'
            f'border-radius:10px;padding:14px;margin-bottom:10px;">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
            f'<span style="font-size:18px;">{ico}</span>'
            f'<div style="font-size:12px;font-weight:800;color:{ABG_DARK};">{feat}</div>'
            f'</div>'
            f'<div style="font-size:11px;color:{ABG_MUTED};line-height:1.5;">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:{ABG_BLUE}08;border-left:3px solid {ABG_BLUE};'
        f'border-radius:0 8px 8px 0;padding:11px 14px;font-size:12px;'
        f'color:{ABG_DARK};line-height:1.65;">The channel selection AI uses individual customer '
        f'features directly — wealth score, digital activity, product ownership, and stated '
        f'preferences — to select the most effective channel for that specific customer.</div>',
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 14px;">'
        f'📊 Channel Analysis</h3>',
        unsafe_allow_html=True)

    _FONT = dict(family="Plus Jakarta Sans", color=ABG_DARK)
    _BG   = dict(paper_bgcolor=ABG_WHITE, plot_bgcolor="#FAFAFA")


    import numpy as np
    import pandas as pd

    _CH_MAP  = {'SMS':'#7F8C8D','WhatsApp':'#25D366','Send Notification':ABG_BLUE,
                'Customer Service':ABG_PURPLE,'Branch Agent':ABG_GOLD}
    _CHANNELS = list(_CH_MAP.keys())
    _FONT = dict(family="Plus Jakarta Sans", color=ABG_DARK)
    _BG   = dict(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA')

    # Generate synthetic channel data
    np.random.seed(42)
    _n = 500
    _grp_list = ['Group 1','Group 2','Group 3','Group 4','Group 5']
    _grp_prob = [0.30,0.25,0.22,0.15,0.08]
    _groups   = np.random.choice(_grp_list, _n, p=_grp_prob)
    _income   = np.array([np.random.uniform(*{'Group 1':(3000,7000),'Group 2':(7000,12000),
        'Group 3':(12000,20000),'Group 4':(20000,35000),'Group 5':(35000,80000)}[g]) for g in _groups])
    _scores   = np.array([np.random.uniform(*{'Group 1':(540,660),'Group 2':(620,700),
        'Group 3':(680,760),'Group 4':(720,810),'Group 5':(770,900)}[g]) for g in _groups])
    _logins   = np.array([np.random.randint(*{'Group 1':(0,8),'Group 2':(5,20),
        'Group 3':(12,35),'Group 4':(25,65),'Group 5':(50,110)}[g]) for g in _groups])
    _spend    = _income * np.random.uniform(0.5, 2.8, _n)
    _dig_act  = np.clip(_logins * 2.5 + np.random.normal(0,10,_n), 0, 100)

    def _assign_ch(inc, login, dig):
        w = min(100, inc/500)
        d = min(100, login*3 + dig*0.3)
        if w >= 80 and d < 30: return 'Customer Service'
        if w >= 80: return 'Customer Service'
        if d >= 55: return 'Send Notification'
        if d >= 30: return 'WhatsApp'
        return 'SMS'

    _ch_list = [_assign_ch(_income[i], _logins[i], _dig_act[i]) for i in range(_n)]

    _df_ch = pd.DataFrame({
        'Group':_groups,'Income':_income,'Credit Score':_scores,
        'App Logins':_logins,'Annual Spend':_spend,'Channel':_ch_list,
    })

    _dim = st.radio(
        "Select analysis dimension:",
        ["By Customer Group","By Credit Score Range","By Annual Spend","By Monthly Income","By Product Owned"],
        horizontal=True, key="ch_dim_sel"
    )

    if _dim == "By Customer Group":
        _pivot = _df_ch.groupby(['Group','Channel']).size().unstack(fill_value=0)
        _pivot = _pivot.reindex(index=_grp_list, columns=_CHANNELS, fill_value=0)
        _fig = go.Figure()
        for ch in _CHANNELS:
            _fig.add_trace(go.Bar(name=ch, x=_grp_list,
                y=_pivot[ch].values, marker_color=_CH_MAP[ch]))
        _fig.update_layout(**_BG, font=_FONT, height=350, barmode='stack',
            margin=dict(t=46,b=30,l=10,r=10),
            title=dict(text="Channel Count by Customer Group (n=500)", font=dict(size=13)),
            xaxis=dict(gridcolor=ABG_BORDER),
            yaxis=dict(gridcolor=ABG_BORDER, title="No. of Customers"),
            legend=dict(font=dict(size=10), orientation='h', y=-0.18))
        st.plotly_chart(_fig, use_container_width=True)

    elif _dim == "By Credit Score Range":
        _sc_bins   = [300,580,650,720,800,900]
        _sc_labels = ['300-580','580-650','650-720','720-800','800-900']
        _df_ch['Score Range'] = pd.cut(_df_ch['Credit Score'], bins=_sc_bins, labels=_sc_labels)
        _pivot = _df_ch.groupby(['Score Range','Channel']).size().unstack(fill_value=0)
        _pivot = _pivot.reindex(index=_sc_labels, columns=_CHANNELS, fill_value=0)
        _fig = go.Figure()
        for ch in _CHANNELS:
            _fig.add_trace(go.Bar(name=ch, x=_sc_labels,
                y=_pivot[ch].values, marker_color=_CH_MAP[ch]))
        _fig.update_layout(**_BG, font=_FONT, height=350, barmode='group',
            margin=dict(t=46,b=30,l=10,r=10),
            title=dict(text="Channel Distribution by Credit Score Range", font=dict(size=13)),
            xaxis=dict(gridcolor=ABG_BORDER),
            yaxis=dict(gridcolor=ABG_BORDER, title="No. of Customers"),
            legend=dict(font=dict(size=10), orientation='h', y=-0.18))
        st.plotly_chart(_fig, use_container_width=True)

    elif _dim == "By Annual Spend Range":
        _sp_bins   = [0,20000,60000,120000,250000,999999]
        _sp_labels = ['<20K SAR','20-60K SAR','60-120K SAR','120-250K SAR','>250K SAR']
        _df_ch['Spend Range'] = pd.cut(_df_ch['Annual Spend'], bins=_sp_bins, labels=_sp_labels)
        _pivot = _df_ch.groupby(['Spend Range','Channel']).size().unstack(fill_value=0)
        _pivot = _pivot.reindex(index=_sp_labels, columns=_CHANNELS, fill_value=0)
        _fig = go.Figure()
        for ch in _CHANNELS:
            _fig.add_trace(go.Bar(name=ch, x=_sp_labels,
                y=_pivot[ch].values, marker_color=_CH_MAP[ch]))
        _fig.update_layout(**_BG, font=_FONT, height=350, barmode='stack',
            margin=dict(t=46,b=30,l=10,r=10),
            title=dict(text="Channel Distribution by Annual Spend (SAR)", font=dict(size=13)),
            xaxis=dict(gridcolor=ABG_BORDER, tickangle=-15),
            yaxis=dict(gridcolor=ABG_BORDER, title="No. of Customers"),
            legend=dict(font=dict(size=10), orientation='h', y=-0.18))
        st.plotly_chart(_fig, use_container_width=True)

    elif _dim == "By Annual Spend":
        _lg_labels = ['<20K','20-60K','60-120K','120-250K','>250K']
        _df_ch['Spend Range2'] = pd.cut(_df_ch['Annual Spend'], bins=[0,20000,60000,120000,250000,999999], labels=['<20K','20-60K','60-120K','120-250K','>250K'])
        _pivot = _df_ch.groupby(['Spend Range2','Channel']).size().unstack(fill_value=0)
        _pivot = _pivot.reindex(index=_lg_labels, columns=_CHANNELS, fill_value=0)
        _fig = go.Figure()
        for ch in _CHANNELS:
            _fig.add_trace(go.Bar(name=ch, x=_lg_labels,
                y=_pivot[ch].values, marker_color=_CH_MAP[ch]))
        _fig.update_layout(**_BG, font=_FONT, height=350, barmode='group',
            margin=dict(t=46,b=30,l=10,r=10),
            title=dict(text="Channel Distribution by Annual Spend (SAR)", font=dict(size=13)),
            xaxis=dict(gridcolor=ABG_BORDER),
            yaxis=dict(gridcolor=ABG_BORDER, title="No. of Customers"),
            legend=dict(font=dict(size=10), orientation='h', y=-0.18))
        st.plotly_chart(_fig, use_container_width=True)

    elif _dim == "By Monthly Income":
        _inc_bins   = [0,5000,10000,15000,25000,999999]
        _inc_labels = ['<5K SAR','5-10K SAR','10-15K SAR','15-25K SAR','>25K SAR']
        _df_ch['Income Range'] = pd.cut(_df_ch['Income'], bins=_inc_bins, labels=_inc_labels)
        _pivot = _df_ch.groupby(['Income Range','Channel']).size().unstack(fill_value=0)
        _pivot = _pivot.reindex(index=_inc_labels, columns=_CHANNELS, fill_value=0)
        _fig = go.Figure()
        for ch in _CHANNELS:
            _fig.add_trace(go.Bar(name=ch, x=_inc_labels,
                y=_pivot[ch].values, marker_color=_CH_MAP[ch]))
        _fig.update_layout(**_BG, font=_FONT, height=350, barmode='stack',
            margin=dict(t=46,b=30,l=10,r=10),
            title=dict(text="Channel Distribution by Monthly Income", font=dict(size=13)),
            xaxis=dict(gridcolor=ABG_BORDER),
            yaxis=dict(gridcolor=ABG_BORDER, title="No. of Customers"),
            legend=dict(font=dict(size=10), orientation='h', y=-0.18))
        st.plotly_chart(_fig, use_container_width=True)

    elif _dim == "By Product Owned":
        import numpy as _np2
        _prod_names = ['SA_SAVING','CC_CASHBACK','CC_TRAVEL','PL_PERSONAL','PL_HOME']
        _prod_labels= ['Savings Acc.','Cashback Card','Travel Card','Personal Finance','Home Finance']
        # Assign random products based on group
        _prod_probs = {
            'Group 1':[0.95,0.15,0.02,0.10,0.01],
            'Group 2':[0.80,0.30,0.08,0.20,0.03],
            'Group 3':[0.75,0.40,0.18,0.30,0.08],
            'Group 4':[0.70,0.50,0.35,0.40,0.20],
            'Group 5':[0.65,0.55,0.60,0.50,0.45],
        }
        prod_ch_rows = []
        for idx_row in range(len(_df_ch)):
            grp = _df_ch.iloc[idx_row]['Group']
            ch  = _df_ch.iloc[idx_row]['Channel']
            for p, prob in zip(_prod_names, _prod_probs[grp]):
                if _np2.random.random() < prob:
                    prod_ch_rows.append({'Product':_prod_labels[_prod_names.index(p)],'Channel':ch})
        _df_prod = pd.DataFrame(prod_ch_rows)
        if len(_df_prod) > 0:
            _pivot = _df_prod.groupby(['Product','Channel']).size().unstack(fill_value=0)
            _pivot = _pivot.reindex(columns=_CHANNELS, fill_value=0)
            _fig = go.Figure()
            for ch in _CHANNELS:
                if ch in _pivot.columns:
                    _fig.add_trace(go.Bar(name=ch, x=_pivot.index.tolist(),
                        y=_pivot[ch].values, marker_color=_CH_MAP[ch]))
            _fig.update_layout(**_BG, font=_FONT, height=350, barmode='group',
                margin=dict(t=46,b=30,l=10,r=10),
                title=dict(text="Channel Distribution by Product Owned", font=dict(size=13)),
                xaxis=dict(gridcolor=ABG_BORDER, tickangle=-10),
                yaxis=dict(gridcolor=ABG_BORDER, title="No. of Customers"),
                legend=dict(font=dict(size=10), orientation='h', y=-0.18))
            st.plotly_chart(_fig, use_container_width=True)

    st.markdown(
        f'<div style="background:{ABG_BLUE}08;border-left:3px solid {ABG_BLUE};'
        f'border-radius:0 8px 8px 0;padding:11px 14px;font-size:12px;'
        f'color:{ABG_DARK};line-height:1.65;">Analysis based on 500 synthetic BSF customers. '
        f'Each bar shows the actual count of customers assigned to that channel.</div>',
        unsafe_allow_html=True)


    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Live Channel Recommendation ───────────────────────────
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 4px;">'
        f'📡 Live Channel Recommendation</h3>',
        unsafe_allow_html=True)
    st.markdown(
        f'<p style="font-size:12px;color:{ABG_MUTED};margin:0 0 14px;">'
        f'Enter customer data to get the best channel recommendation instantly</p>',
        unsafe_allow_html=True)

    with st.form("live_channel_form"):
        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            lc_income  = st.number_input("Monthly Income (SAR)",     value=12000, min_value=0, step=500)
            lc_balance = st.number_input("Account Balance (SAR)",    value=50000, min_value=0, step=1000)
        with lc2:
            lc_score   = st.number_input("Credit Score",              value=700, min_value=300, max_value=900, step=10)
            lc_logins  = st.number_input("App Logins / Month",        value=10, min_value=0)
        with lc3:
            lc_dig     = st.number_input("Digital Activity Score",    value=40, min_value=0, max_value=100)
            lc_pref    = st.selectbox("Preferred Channel",
                ['Mobile App','Branch','WhatsApp','SMS','Online Banking'])
            lc_products = st.text_input("Products Owned",
                value='SA_SAVING', help="e.g. SA_SAVING|CC_CASHBACK")

        lc_submit = st.form_submit_button("📡 Get Best Channel", use_container_width=True)

    if lc_submit:
        # ML-based channel selection on individual features
        _w_score = min(100, lc_income/500 + lc_balance/5000)
        _d_score = min(100, lc_logins*3 + lc_dig*0.3 + (10 if lc_pref=='Mobile App' else 0))

        if _w_score >= 80 and _d_score < 30:
            _ch = 'Customer Service'; _reason = 'High-value customer, low digital activity → personal service'
        elif _w_score >= 80:
            _ch = 'Customer Service'; _reason = 'High-value customer with digital engagement'
        elif _d_score >= 55 and lc_pref == 'Mobile App':
            _ch = 'Send Notification'; _reason = 'Highly active mobile user → push maximizes open rate'
        elif _d_score >= 35:
            _ch = 'WhatsApp'; _reason = 'Moderate digital engagement → personalized messaging'
        elif lc_pref == 'Branch':
            _ch = 'Branch Agent'; _reason = 'Customer prefers in-branch service'
        else:
            _ch = 'SMS'; _reason = 'Low digital activity → SMS ensures reach'

        _CHANNEL_ICONS  = {'SMS':'💬','WhatsApp':'📱','Send Notification':'🔔',
                           'Customer Service':'🤝','Branch Agent':'🏦'}
        _CHANNEL_COLORS = {'SMS':'#7F8C8D','WhatsApp':'#25D366',
                           'Send Notification':ABG_BLUE,'Customer Service':ABG_PURPLE,
                           'Branch Agent':ABG_GOLD}

        _icon  = _CHANNEL_ICONS.get(_ch, '📡')
        _color = _CHANNEL_COLORS.get(_ch, ABG_BLUE)

        # Show result
        res1, res3 = st.columns(2)
        with res1:
            st.markdown(
                f'<div style="background:{_color}12;border:2px solid {_color}44;'
                f'border-radius:14px;padding:24px;text-align:center;border-top:4px solid {_color};">'
                f'<div style="font-size:11px;font-weight:700;color:{ABG_MUTED};'
                f'text-transform:uppercase;margin-bottom:8px;">Best Channel</div>'
                f'<div style="font-size:52px;">{_icon}</div>'
                f'<div style="font-size:20px;font-weight:800;color:{_color};margin-top:8px;">{_ch}</div>'
                f'</div>',
                unsafe_allow_html=True)


        with res3:
            st.markdown(
                f'<div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};'
                f'border-radius:14px;padding:20px;">'
                f'<div style="font-size:12px;font-weight:700;color:{ABG_DARK};margin-bottom:10px;">AI Reasoning</div>'
                f'<div style="font-size:12px;color:{ABG_MUTED};line-height:1.7;margin-bottom:14px;">{_reason}</div>'
                f'<div style="font-size:10px;font-weight:700;color:{ABG_MUTED};margin-bottom:6px;">KEY INPUTS USED</div>'
                f'<div style="font-size:11px;color:{ABG_DARK};">Income: SAR {lc_income:,}</div>'
                f'<div style="font-size:11px;color:{ABG_DARK};">App Logins: {lc_logins}/month</div>'
                f'<div style="font-size:11px;color:{ABG_DARK};">Digital Activity: {lc_dig}/100</div>'
                f'<div style="font-size:11px;color:{ABG_DARK};">Preferred: {lc_pref}</div>'
                f'</div>',
                unsafe_allow_html=True)
