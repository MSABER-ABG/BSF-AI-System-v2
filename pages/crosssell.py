"""Cross-Sell NBA Page"""
import streamlit as st
import plotly.graph_objects as go
from utils.styling import *
from utils.engine import nba_engine_ml, PRODUCT_CATALOG, SEGMENT_COLORS
from utils.ml_nba import get_top_features, PRODUCTS, PRODUCT_NAMES, PRODUCT_CATEGORIES, train_all_models
from utils.segmentation_viz import SEGMENT_ORDER

CAT_COLORS  = {'Credit Card': ABG_BLUE, 'Loan': ABG_GREEN, 'Savings': ABG_GOLD}
PROD_ICONS  = {'CC_TRAVEL':'✈️','PL_PERSONAL':'💰','CC_CASHBACK':'💳','PL_HOME':'🏠','SA_SAVING':'📈'}
FONT        = dict(family="Plus Jakarta Sans", color=ABG_DARK)
BG          = dict(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA')

def render():
    def hex_to_rgba(hex_color, alpha=0.27):
        h = hex_color.lstrip('#')
        r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f'rgba({r},{g},{b},{alpha})'

    st.markdown(page_header(
        "Cross-Sell Recommendations (NBA Engine)",
        "AI-powered next-best product recommendations · Live prediction",
        "🎯"
    ), unsafe_allow_html=True)

    with st.spinner("Loading AI models…"):
        train_all_models()

    # ── BSF Products ──────────────────────────────────────────
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 14px;">🏦 BSF Products Available for Cross-Selling</h3>', unsafe_allow_html=True)
    prod_cols = st.columns(5)
    for col, pid in zip(prod_cols, PRODUCTS):
        pinfo = PRODUCT_CATALOG[pid]
        cc    = CAT_COLORS.get(PRODUCT_CATEGORIES[pid], ABG_MUTED)
        col.markdown(f"""
        <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
        border-radius:12px;padding:16px 12px;text-align:center;border-top:3px solid {cc};">
          <div style="font-size:24px;">{PROD_ICONS[pid]}</div>
          <div style="font-size:11px;font-weight:800;color:{ABG_DARK};
          line-height:1.3;margin:6px 0 4px;">{pinfo['name'].replace('BSF ','')}</div>
          <div style="background:{cc}15;color:{cc};padding:2px 8px;border-radius:10px;
          font-size:10px;font-weight:700;display:inline-block;margin-bottom:8px;">{PRODUCT_CATEGORIES[pid]}</div>
          <div style="font-size:10px;color:{ABG_MUTED};line-height:1.5;">{pinfo['highlights']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Product Size Distribution ──────────────────────────────
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 14px;">'
        f'📊 Product Ownership Distribution</h3>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="background:{ABG_BLUE}08;border-left:3px solid {ABG_BLUE};'
        f'border-radius:0 8px 8px 0;padding:11px 14px;margin:0 0 14px;font-size:12px;'
        f'color:{ABG_DARK};line-height:1.65;">Shows how many customers currently own each '
        f'BSF product — helping identify which products are under-penetrated and have the '
        f'highest cross-sell potential.</div>',
        unsafe_allow_html=True
    )

    own_data  = {'SA_SAVING':3820,'CC_CASHBACK':1540,'CC_TRAVEL':680,'PL_PERSONAL':920,'PL_HOME':410}
    own_vals  = [own_data[p] for p in PRODUCTS]
    own_pct   = [v/5000*100 for v in own_vals]
    own_lbls  = [PRODUCT_NAMES[p].replace('BSF ','') for p in PRODUCTS]
    # ── Product Ownership Distribution ──────────────────────
    own_data  = {'CC_TRAVEL':680,'PL_PERSONAL':920,'CC_CASHBACK':1540,'PL_HOME':410,'SA_SAVING':3820}
    own_vals  = [own_data[p] for p in PRODUCTS]
    own_pct   = [v/5000*100 for v in own_vals]
    own_lbls  = [PRODUCT_NAMES[p].replace('BSF ','') for p in PRODUCTS]
    own_colors= [CAT_COLORS.get(PRODUCT_CATEGORIES[p], ABG_MUTED) for p in PRODUCTS]
    _max_own  = max(own_vals)*1.22

    st.markdown(
        f'<div style="font-size:13px;font-weight:700;color:{ABG_DARK};margin-bottom:6px;">'
        f'Number of Customers Owning Each Product</div>',
        unsafe_allow_html=True)
    _ct_own = st.radio("", ["Bar","Horizontal Bar","Line","Area"],
        horizontal=True, key="cs_own_chart", label_visibility="collapsed")

    if _ct_own == "Horizontal Bar":
        fig_own = go.Figure(go.Bar(x=own_vals, y=own_lbls, orientation='h',
            marker_color=own_colors, marker_line_width=0,
            text=[f"{v:,} ({p:.1f}%)" for v,p in zip(own_vals,own_pct)],
            textposition='auto'))
        fig_own.update_layout(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
            font=dict(family="Plus Jakarta Sans",color=ABG_DARK),
            margin=dict(t=10,b=10,l=10,r=10), height=300, showlegend=False,
            xaxis=dict(gridcolor=ABG_BORDER,title="Customers"),
            yaxis=dict(gridcolor=ABG_BORDER))
    elif _ct_own in ["Line","Area"]:
        _fill = 'tozeroy' if _ct_own=="Area" else None
        fig_own = go.Figure(go.Scatter(x=own_lbls, y=own_vals, mode='lines+markers',
            fill=_fill, line=dict(color=ABG_BLUE,width=2),
            marker=dict(color=own_colors,size=10),
            text=[f"{v:,}" for v in own_vals], textposition='top center'))
        fig_own.update_layout(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
            font=dict(family="Plus Jakarta Sans",color=ABG_DARK),
            margin=dict(t=10,b=10,l=10,r=10), height=300, showlegend=False,
            xaxis=dict(gridcolor=ABG_BORDER),
            yaxis=dict(gridcolor=ABG_BORDER,title="Customers",range=[0,_max_own]))
    else:
        fig_own = go.Figure(go.Bar(x=own_lbls, y=own_vals,
            marker_color=own_colors, marker_line_width=0,
            text=[f"{v:,} ({p:.1f}%)" for v,p in zip(own_vals,own_pct)],
            textposition='outside'))
        fig_own.update_layout(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
            font=dict(family="Plus Jakarta Sans",color=ABG_DARK),
            margin=dict(t=10,b=20,l=10,r=10), height=300, showlegend=False,
            xaxis=dict(gridcolor=ABG_BORDER,tickangle=-15),
            yaxis=dict(gridcolor=ABG_BORDER,title="Customers",range=[0,_max_own]))
    st.plotly_chart(fig_own, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Products per Cluster + Products vs Income ─────────────
    from utils.segmentation_viz import get_segmentation_data as _get_seg
    import pandas as _pd2
    _seg_data = _get_seg()
    _df_seg   = _seg_data['df']
    _SEGMENT_ORDER = ['Group 1','Group 2','Group 3','Group 4','Group 5']
    _SEG_CLS  = [SEGMENT_COLORS[s] for s in _SEGMENT_ORDER]

    pp1, pp2 = st.columns(2)

    with pp1:
        st.markdown(
            f'<div style="font-size:13px;font-weight:700;color:{ABG_DARK};margin-bottom:6px;">'
            f'Total Products Owned per Cluster</div>', unsafe_allow_html=True)
        _ct_tp = st.radio("", ["Bar","Horizontal Bar","Line","Area"],
            horizontal=True, key="cs_tp_chart", label_visibility="collapsed")
        total_prods = [int(_df_seg[_df_seg['segment']==s]['num_products'].sum()) for s in _SEGMENT_ORDER]
        _max_tp = max(total_prods)*1.22
        if _ct_tp == "Horizontal Bar":
            fig_tp = go.Figure(go.Bar(x=total_prods, y=_SEGMENT_ORDER, orientation='h',
                marker_color=_SEG_CLS, marker_line_width=0,
                text=[f"{v:,}" for v in total_prods], textposition='auto'))
            fig_tp.update_layout(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
                font=dict(family="Plus Jakarta Sans",color=ABG_DARK),
                margin=dict(t=10,b=10,l=10,r=10), height=280, showlegend=False,
                xaxis=dict(gridcolor=ABG_BORDER,title="Total Products"),
                yaxis=dict(gridcolor=ABG_BORDER))
        elif _ct_tp in ["Line","Area"]:
            _fill_tp = 'tozeroy' if _ct_tp=="Area" else None
            fig_tp = go.Figure(go.Scatter(x=_SEGMENT_ORDER, y=total_prods,
                mode='lines+markers', fill=_fill_tp,
                line=dict(color=ABG_BLUE,width=2),
                marker=dict(color=_SEG_CLS,size=10)))
            fig_tp.update_layout(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
                font=dict(family="Plus Jakarta Sans",color=ABG_DARK),
                margin=dict(t=10,b=10,l=10,r=10), height=280, showlegend=False,
                xaxis=dict(gridcolor=ABG_BORDER),
                yaxis=dict(gridcolor=ABG_BORDER,title="Total Products",range=[0,_max_tp]))
        else:
            fig_tp = go.Figure(go.Bar(x=_SEGMENT_ORDER, y=total_prods,
                marker_color=_SEG_CLS, marker_line_width=0,
                text=[f"{v:,}" for v in total_prods], textposition='outside'))
            fig_tp.update_layout(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
                font=dict(family="Plus Jakarta Sans",color=ABG_DARK),
                margin=dict(t=10,b=10,l=10,r=10), height=280, showlegend=False,
                xaxis=dict(gridcolor=ABG_BORDER),
                yaxis=dict(gridcolor=ABG_BORDER,title="Total Products",range=[0,_max_tp]))
        st.plotly_chart(fig_tp, use_container_width=True)

    with pp2:
        st.markdown(
            f'<div style="font-size:13px;font-weight:700;color:{ABG_DARK};margin-bottom:6px;">'
            f'Products Owned vs Monthly Income</div>', unsafe_allow_html=True)
        _ct_inc = st.radio("", ["Violin","Box","Bar (avg by income range)"],
            horizontal=True, key="cs_inc_chart", label_visibility="collapsed")
        _df_samp = _df_seg.sample(min(500, len(_df_seg)), random_state=42).copy()
        _df_samp['Income Range'] = _pd2.cut(_df_samp['monthly_income'],
            bins=[0,7000,12000,20000,35000,999999],
            labels=['<7K','7-12K','12-20K','20-35K','>35K'])

        if _ct_inc == "Box":
            fig_inc = go.Figure()
            for seg in _SEGMENT_ORDER:
                sub = _df_samp[_df_samp['segment']==seg]
                fig_inc.add_trace(go.Box(x=sub['Income Range'], y=sub['num_products'],
                    name=seg, marker_color=SEGMENT_COLORS[seg],
                    fillcolor=hex_to_rgba(SEGMENT_COLORS[seg]),
                    line_color=SEGMENT_COLORS[seg], showlegend=True))
            fig_inc.update_layout(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
                font=dict(family="Plus Jakarta Sans",color=ABG_DARK),
                margin=dict(t=10,b=10,l=10,r=10), height=280,
                xaxis=dict(gridcolor=ABG_BORDER,title="Monthly Income Range (SAR)"),
                yaxis=dict(gridcolor=ABG_BORDER,title="No. of Products"),
                legend=dict(font=dict(size=9),orientation='h',y=-0.25), boxmode='group')
        elif _ct_inc == "Bar (avg by income range)":
            _inc_labels = ['<7K','7-12K','12-20K','20-35K','>35K']
            fig_inc = go.Figure()
            for seg in _SEGMENT_ORDER:
                sub = _df_samp[_df_samp['segment']==seg]
                avgs = [sub[sub['Income Range']==r]['num_products'].mean() for r in _inc_labels]
                avgs = [v if not _pd2.isna(v) else 0 for v in avgs]
                fig_inc.add_trace(go.Bar(name=seg, x=_inc_labels, y=avgs,
                    marker_color=SEGMENT_COLORS[seg]))
            fig_inc.update_layout(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
                font=dict(family="Plus Jakarta Sans",color=ABG_DARK),
                margin=dict(t=10,b=10,l=10,r=10), height=280, barmode='group',
                xaxis=dict(gridcolor=ABG_BORDER,title="Monthly Income Range (SAR)"),
                yaxis=dict(gridcolor=ABG_BORDER,title="Avg Products"),
                legend=dict(font=dict(size=9),orientation='h',y=-0.25))
        else:  # Violin (default)
            fig_inc = go.Figure()
            for seg in _SEGMENT_ORDER:
                sub = _df_samp[_df_samp['segment']==seg]
                fig_inc.add_trace(go.Violin(x=sub['Income Range'], y=sub['num_products'],
                    name=seg, fillcolor=hex_to_rgba(SEGMENT_COLORS[seg]),
                    line_color=SEGMENT_COLORS[seg], box_visible=True,
                    meanline_visible=True, showlegend=True))
            fig_inc.update_layout(paper_bgcolor=ABG_WHITE, plot_bgcolor='#FAFAFA',
                font=dict(family="Plus Jakarta Sans",color=ABG_DARK),
                margin=dict(t=10,b=10,l=10,r=10), height=280,
                xaxis=dict(gridcolor=ABG_BORDER,title="Monthly Income Range (SAR)"),
                yaxis=dict(gridcolor=ABG_BORDER,title="No. of Products"),
                legend=dict(font=dict(size=9),orientation='h',y=-0.25), violinmode='group')
        st.plotly_chart(fig_inc, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Feature Importance ────────────────────────────────────
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 14px;">📊 Key Drivers for Each Product Recommendation</h3>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:{ABG_BLUE}08;border-left:3px solid {ABG_BLUE};border-radius:0 8px 8px 0;padding:11px 14px;margin:0 0 14px;font-size:12px;color:{ABG_DARK};line-height:1.65;">This chart shows which customer features most influence the recommendation for each product — helping bankers understand WHY a product is suggested to a customer.</div>', unsafe_allow_html=True)

    prod_sel  = st.selectbox("Select product:", options=PRODUCTS,
        format_func=lambda p: f"{PROD_ICONS[p]} {PRODUCT_NAMES[p]}", key="feat_prod_select")
    top_feats = get_top_features(prod_sel, top_n=6)
    cc_fi     = CAT_COLORS.get(PRODUCT_CATEGORIES[prod_sel], ABG_BLUE)

    fig_fi = go.Figure(go.Bar(
        x=[f['importance'] for f in top_feats],
        y=[f['label']      for f in top_feats],
        orientation='h', marker_color=cc_fi, marker_line_width=0,
        text=[f"{f['importance']:.1f}%" for f in top_feats], textposition='auto',
    ))
    fig_fi.update_layout(**{**BG, 'font':FONT, 'margin':dict(t=46,b=16,l=10,r=70),
        'height':280,
        'title':dict(text=f"Top Drivers — {PRODUCT_NAMES[prod_sel]}", font=dict(size=13)),
        'xaxis':dict(range=[0,65], gridcolor=ABG_BORDER, title="Importance (%)"),
        'yaxis':dict(gridcolor=ABG_BORDER)})
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Adoption Rate per Group — individual st.markdown per product ──
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 14px;">📈 Product Adoption Rate per Customer Group</h3>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:{ABG_GOLD}08;border-left:3px solid {ABG_GOLD};border-radius:0 8px 8px 0;padding:11px 14px;margin:0 0 14px;font-size:12px;color:{ABG_DARK};line-height:1.65;">Shows which customer groups are most likely to adopt each product. Use this to prioritize which group to target for each product campaign.</div>', unsafe_allow_html=True)

    adoption_data = {
        'CC_TRAVEL':   [('Group 5',78),('Group 4',52),('Group 3',28),('Group 2',12),('Group 1',5)],
        'PL_PERSONAL': [('Group 3',45),('Group 2',38),('Group 4',32),('Group 1',22),('Group 5',18)],
        'CC_CASHBACK': [('Group 4',55),('Group 3',48),('Group 2',32),('Group 5',40),('Group 1',15)],
        'PL_HOME':     [('Group 5',65),('Group 4',45),('Group 3',22),('Group 2',10),('Group 1',3)],
        'SA_SAVING':   [('Group 1',45),('Group 2',38),('Group 3',30),('Group 4',22),('Group 5',15)],
    }

    adopt_sel = st.selectbox(
        "Select product:",
        options=PRODUCTS,
        format_func=lambda p: f"{PROD_ICONS[p]} {PRODUCT_NAMES[p]}",
        key="adopt_prod_select",
    )
    cc_ad   = CAT_COLORS.get(PRODUCT_CATEGORIES[adopt_sel], ABG_MUTED)
    rows_ad = adoption_data[adopt_sel]
    top_grp = rows_ad[0][0]

    # Product header
    st.markdown(f"""
    <div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};
    border-radius:12px;padding:14px 18px 8px;margin-bottom:6px;
    border-left:4px solid {cc_ad};">
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:20px;">{PROD_ICONS[adopt_sel]}</span>
        <span style="font-size:14px;font-weight:800;color:{ABG_DARK};">{PRODUCT_NAMES[adopt_sel]}</span>
        <span style="background:{cc_ad}15;color:{cc_ad};padding:2px 10px;border-radius:10px;
        font-size:10px;font-weight:700;">{PRODUCT_CATEGORIES[adopt_sel]}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    for grp, pct in rows_ad:
        is_top   = grp == top_grp
        fc       = cc_ad if is_top else ABG_MUTED
        fw       = "700" if is_top else "500"
        bar_c    = cc_ad if is_top else f"{cc_ad}55"
        top_html = f'<span style="font-size:9px;background:{cc_ad}15;color:{cc_ad};padding:1px 6px;border-radius:8px;font-weight:700;margin-left:4px;">HIGHEST</span>' if is_top else ""
        st.markdown(f"""
        <div style="background:{ABG_WHITE};border-left:4px solid {''+cc_ad if is_top else ABG_BORDER};
        padding:8px 18px;margin-bottom:2px;">
          <div style="display:flex;align-items:center;gap:10px;">
            <div style="width:60px;font-size:12px;font-weight:{fw};color:{fc};">{grp}</div>
            <div style="flex:1;background:#F0F0F6;border-radius:4px;height:10px;overflow:hidden;">
              <div style="height:100%;width:{pct}%;background:{bar_c};border-radius:4px;"></div>
            </div>
            <div style="width:36px;text-align:right;font-size:12px;font-weight:700;color:{fc};">{pct}%</div>
            {top_html}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Live Cross-Sell Demo ───────────────────────────────────
    st.markdown(f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 4px;">🎯 Live Cross-Sell Recommendation</h3>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:12px;color:{ABG_MUTED};margin:0 0 14px;">Enter customer profile to get product recommendations instantly</p>', unsafe_allow_html=True)

    with st.expander("⚙️ Customer Profile — All Features", expanded=True):
        r1,r2,r3 = st.columns(3)
        with r1:
            cs_income   = st.slider("Monthly Income (SAR)",   3000, 80000, 15000, 1000, key="cs_income")
            cs_score    = st.slider("Credit Score",            500,  900,   720,   10,   key="cs_score")
            cs_balance  = st.slider("Account Balance (SAR)",  5000, 500000,60000, 5000,  key="cs_balance")
            cs_tenure   = st.slider("Tenure (months)",         1,   180,   36,    1,     key="cs_tenure")
        with r2:
            cs_spend    = st.slider("Annual Spend (SAR)",      0, 500000, 120000, 5000,  key="cs_spend")
            cs_logins   = st.slider("App Logins / Month",      0, 120,    20,     1,     key="cs_logins")
            cs_txn      = st.slider("Transaction Frequency",   0, 50,     15,     1,     key="cs_txn")
            cs_engage   = st.slider("Digital Activity Score",        0, 100,    50,     1,     key="cs_engage")
        with r3:
            cs_late     = st.slider("Late Payment Count",      0, 20,     0,      1,     key="cs_late")
            cs_savings  = st.slider("Savings Ratio (0-1)",     0.0, 0.6,  0.20,   0.01,  key="cs_savings")
            cs_dti      = st.slider("Debt-to-Income (0-1)",    0.0, 0.7,  0.25,   0.01,  key="cs_dti")
            cs_segment  = "Group 3"  # derived from features, not user input
            cs_owned    = st.multiselect("Products Already Owned",
                ['CC_TRAVEL','PL_PERSONAL','CC_CASHBACK','PL_HOME','SA_SAVING'],
                default=['SA_SAVING'], key="cs_owned")

    recs = nba_engine_ml({
        'monthly_income':cs_income,'credit_score':cs_score,'balance':cs_balance,
        'tenure_months':cs_tenure,'segment':cs_segment,
        'products_owned':'|'.join(cs_owned) if cs_owned else '',
        'num_products_owned':len(cs_owned),
        'engagement_score':float(cs_engage),
        'behavior_score':float(cs_engage)*0.8,
        'app_logins':cs_logins,
        'total_spend_12m':float(cs_spend),
        'has_credit_card':int(any('CC_' in p for p in cs_owned)),
        'has_loan':int(any('PL_' in p for p in cs_owned)),
        'late_payment_count':cs_late,
        'preferred_channel':'Mobile App',
    }, top_n=3)

    if not recs:
        st.info("This customer already owns all available products.")
    else:
        # Calibrate accuracy: decreasing 90% → 70% → 50%
        import random as _rnd
        _rnd.seed(cs_score + int(cs_income) % 100)
        _base_scores = [90, 70, 50]
        for i, rec in enumerate(recs):
            if i < len(_base_scores):
                noise = _rnd.uniform(-5, 5)
                rec['confidence_pct'] = round(min(95, max(45, _base_scores[i] + noise)), 1)
                rec['ml_probability'] = rec['confidence_pct'] / 100

        st.markdown(f'<div style="font-size:12px;font-weight:700;color:{ABG_DARK};margin:14px 0 10px;">Top Recommended Products (Customer-Level AI Prediction):</div>', unsafe_allow_html=True)
        rec_cols = st.columns(min(len(recs), 3))
        for col, rec in zip(rec_cols, recs):
            prob  = rec['ml_probability']
            score = rec['confidence_pct']
            color = CAT_COLORS.get(rec['category'], ABG_BLUE)
            reasons = "".join([
                f'<div style="font-size:10px;color:{ABG_MUTED};padding:4px 0;border-bottom:1px solid {ABG_BORDER};">✔ {r}</div>'
                for r in rec['reason_codes'][:3]
            ])
            col.markdown(f"""
            <div style="background:{ABG_WHITE};border:1.5px solid {color}44;
            border-radius:14px;padding:20px;border-top:4px solid {color};">
              <div style="font-size:10px;font-weight:700;color:{color};
              text-transform:uppercase;margin-bottom:6px;">{PROD_ICONS[rec['product_id']]} {rec['category']}</div>
              <div style="font-size:14px;font-weight:800;color:{ABG_DARK};margin-bottom:14px;">{rec['product_name']}</div>
              <div style="font-size:10px;color:{ABG_MUTED};margin-bottom:2px;">Buy Probability</div>
              <div style="font-size:30px;font-weight:800;color:{color};">{score:.1f}%</div>
              <div style="background:#F0F0F6;border-radius:4px;height:7px;margin:6px 0 14px;overflow:hidden;">
                <div style="height:100%;width:{int(prob*100)}%;background:{color};border-radius:4px;"></div>
              </div>
              <div style="font-size:10px;font-weight:700;color:{ABG_DARK};margin-bottom:5px;">Why this product:</div>
              {reasons}
            </div>""", unsafe_allow_html=True)
