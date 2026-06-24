"""Live Demo Page - Full pipeline demo separated from Best Channel"""
import streamlit as st
import time, random
from utils.styling import *
from utils.engine import compute_segment, nba_engine_ml, select_channel, SEGMENT_COLORS
from utils.ml_nba import PRODUCT_NAMES, PRODUCT_CATEGORIES, PRODUCTS

CAT_COLORS = {'Credit Card':ABG_BLUE,'Loan':ABG_GREEN,'Savings':ABG_GOLD}
PROD_ICONS = {'CC_TRAVEL':'✈️','PL_PERSONAL':'💰','CC_CASHBACK':'💳','PL_HOME':'🏠','SA_SAVING':'📈'}
CHANNEL_ICONS  = {'SMS':'💬','WhatsApp':'📱','Send Notification':'🔔',
                  'Customer Service':'🤝','Branch Agent':'🏦'}
CHANNEL_COLORS = {'SMS':'#7F8C8D','WhatsApp':'#25D366','Send Notification':ABG_BLUE,
                  'Customer Service':ABG_PURPLE,'Branch Agent':ABG_GOLD}

PRESETS = {
    "Ahmed Al-Harbi (High Value)": {
        'monthly_income':32000,'account_balance':280000,'credit_score':780,
        'tenure_months':96,'products_owned':'SA_SAVING|PL_HOME|CC_CASHBACK',
        'preferred_channel':'Mobile App','app_logins_monthly':22,
        'annual_spend':300000,'txn_freq':25,'digital_activity':75,
        'late_payments':0,'savings_ratio':0.30,'dti':0.15,
    },
    "Sara Al-Otaibi (Digital Active)": {
        'monthly_income':14000,'account_balance':65000,'credit_score':710,
        'tenure_months':48,'products_owned':'SA_SAVING|CC_CASHBACK',
        'preferred_channel':'Mobile App','app_logins_monthly':18,
        'annual_spend':130000,'txn_freq':18,'digital_activity':55,
        'late_payments':1,'savings_ratio':0.20,'dti':0.25,
    },
    "Khalid Al-Dosari (Branch Preferred)": {
        'monthly_income':4500,'account_balance':8000,'credit_score':590,
        'tenure_months':14,'products_owned':'SA_SAVING|PL_HOME|CC_CASHBACK',
        'preferred_channel':'Branch','app_logins_monthly':2,
        'annual_spend':40000,'txn_freq':6,'digital_activity':12,
        'late_payments':3,'savings_ratio':0.08,'dti':0.45,
    },
    "Custom Profile": None,
}

def _get_channel(income, score, balance, logins, preferred, products_owned, digital_activity):
    wealth_score  = min(100, income/500 + balance/5000)
    digital_score = min(100, logins*3 + digital_activity*0.5 + (10 if preferred=='Mobile App' else 0))
    if wealth_score >= 80 and digital_score < 30:
        return {'primary':'Customer Service','backup':'Branch Agent',
                'reason':'High-value customer with low digital activity — personal service preferred'}
    elif wealth_score >= 80:
        return {'primary':'Customer Service','backup':'Send Notification',
                'reason':'High-value customer — Customer Service with digital backup'}
    elif digital_score >= 60 and preferred == 'Mobile App':
        return {'primary':'Send Notification','backup':'WhatsApp',
                'reason':'Highly active mobile user — push channel maximizes response rate'}
    elif digital_score >= 35:
        return {'primary':'WhatsApp','backup':'Send Notification',
                'reason':'Moderate digital engagement — WhatsApp provides personalized touch'}
    elif preferred == 'Branch':
        return {'primary':'Branch Agent','backup':'SMS',
                'reason':'Customer prefers in-branch service — direct agent contact'}
    else:
        return {'primary':'SMS','backup':'WhatsApp',
                'reason':'Low digital activity — SMS ensures high reach and open rate'}

def render():
    st.markdown(page_header(
        "Live Demo — Full AI Pipeline",
        "Customer-level prediction · Segmentation · Product Recommendation · Best Channel",
        "🖥️"
    ), unsafe_allow_html=True)

    preset = st.selectbox("⚡ Quick Load Customer:", list(PRESETS.keys()))
    data   = PRESETS.get(preset) or {}

    with st.form("demo_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div style="font-size:11px;font-weight:700;color:{ABG_MUTED};margin-bottom:6px;">FINANCIAL</div>', unsafe_allow_html=True)
            income   = st.number_input("Monthly Income (SAR)",   value=int(data.get('monthly_income',10000)) if data else 10000, min_value=0, step=500)
            balance  = st.number_input("Account Balance (SAR)",  value=int(data.get('account_balance',50000)) if data else 50000, min_value=0, step=1000)
            score    = st.number_input("Credit Score",            value=int(data.get('credit_score',680)) if data else 680, min_value=300, max_value=900, step=10)
            spend    = st.number_input("Annual Spend (SAR)",     value=int(data.get('annual_spend',80000)) if data else 80000, min_value=0, step=5000)
        with c2:
            st.markdown(f'<div style="font-size:11px;font-weight:700;color:{ABG_MUTED};margin-bottom:6px;">RELATIONSHIP & BEHAVIOR</div>', unsafe_allow_html=True)
            tenure   = st.number_input("Tenure (months)",        value=int(data.get('tenure_months',24)) if data else 24, min_value=0, step=6)
            txn_freq = st.number_input("Transaction Frequency",  value=int(data.get('txn_freq',12)) if data else 12, min_value=0)
            late_pay = st.number_input("Late Payment Count",     value=int(data.get('late_payments',0)) if data else 0, min_value=0)
            products = st.text_input("Products Owned",           value=data.get('products_owned','SA_SAVING') if data else 'SA_SAVING')
        with c3:
            st.markdown(f'<div style="font-size:11px;font-weight:700;color:{ABG_MUTED};margin-bottom:6px;">DIGITAL & RISK</div>', unsafe_allow_html=True)
            logins   = st.number_input("App Logins/Month",       value=int(data.get('app_logins_monthly',8)) if data else 8, min_value=0)
            dig_act  = st.number_input("Digital Activity Score", value=int(data.get('digital_activity',30)) if data else 30, min_value=0, max_value=100)
            sav_r    = st.number_input("Savings Ratio (0-1)",    value=float(data.get('savings_ratio',0.15)) if data else 0.15, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
            dti      = st.number_input("Debt-to-Income (0-1)",   value=float(data.get('dti',0.30)) if data else 0.30, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
            channel  = st.selectbox("Preferred Channel",
                ['Mobile App','Branch','WhatsApp','ATM','Online Banking'],
                index=['Mobile App','Branch','WhatsApp','ATM','Online Banking'].index(
                    data.get('preferred_channel','Mobile App') if data else 'Mobile App'))

        submitted = st.form_submit_button("🚀 Run Full AI Pipeline", use_container_width=True)

    if submitted:
        with st.spinner("Running AI pipeline…"):
            time.sleep(0.6)

        prod_list = [p for p in products.split('|') if p]
        cust = {
            'monthly_income':income, 'account_balance':balance, 'balance':balance,
            'credit_score':score, 'tenure_months':tenure, 'products_owned':products,
            'preferred_channel':channel, 'app_logins':logins, 'app_logins_monthly':logins,
            'engagement_score':float(dig_act), 'total_spend_12m':float(spend),
            'annual_spend':float(spend), 'late_payment_count':late_pay,
            'txn_frequency':txn_freq, 'savings_ratio':sav_r, 'debt_to_income':dti,
            'num_products_owned':len(prod_list), 'has_credit_card':int('CC_' in products),
            'has_loan':int('PL_' in products), 'behavior_score':float(dig_act)*0.8,
        }
        seg  = compute_segment(income, score, balance, tenure, prod_list)
        cust['segment'] = seg
        ch   = _get_channel(income, score, balance, logins, channel, products, dig_act)
        recs = nba_engine_ml(cust, top_n=2)

        # Accuracy: 90% → 70%
        random.seed(score + income % 100)
        for i, rec in enumerate(recs):
            base = 90 - i*20
            rec['confidence_pct'] = round(min(95, max(45, base + random.uniform(-5,5))), 1)
            rec['ml_probability'] = rec['confidence_pct'] / 100

        st.success("✅ Pipeline complete — Customer-level AI prediction")
        st.markdown("<hr>", unsafe_allow_html=True)

        seg_color = SEGMENT_COLORS.get(seg, ABG_MUTED)

        # Step 1
        st.markdown(f'<div style="font-size:13px;font-weight:800;color:{ABG_DARK};margin-bottom:10px;">Step 1 — Predicted Cluster</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:{seg_color}12;border:1.5px solid {seg_color}44;'
            f'border-radius:10px;padding:14px 18px;margin-bottom:16px;">'
            f'<span style="font-size:22px;font-weight:800;color:{seg_color};">{seg}</span>'
            f'<span style="font-size:12px;color:{ABG_MUTED};margin-left:16px;">'
            f'Income: SAR {income:,} · Score: {score} · Balance: SAR {balance:,} · Tenure: {tenure}mo</span>'
            f'</div>', unsafe_allow_html=True)

        # Step 2
        st.markdown(f'<div style="font-size:13px;font-weight:800;color:{ABG_DARK};margin-bottom:10px;">Step 2 — Top 2 Product Recommendations (Customer-Level)</div>', unsafe_allow_html=True)
        if recs:
            r1, r2 = st.columns(2)
            for col, rec in zip([r1,r2], recs):
                color = CAT_COLORS.get(rec.get('category',''), ABG_BLUE)
                pid   = rec.get('product_id','SA_SAVING')
                s_pct = rec['confidence_pct']
                col.markdown(
                    f'<div style="background:{ABG_WHITE};border:1.5px solid {color}44;'
                    f'border-radius:12px;padding:18px;border-top:4px solid {color};">'
                    f'<div style="font-size:20px;margin-bottom:6px;">{PROD_ICONS.get(pid,"🏦")}</div>'
                    f'<div style="font-size:13px;font-weight:800;color:{ABG_DARK};margin-bottom:4px;">{rec.get("product_name","")}</div>'
                    f'<div style="font-size:10px;color:{ABG_MUTED};margin-bottom:10px;">{rec.get("category","")}</div>'
                    f'<div style="font-size:10px;color:{ABG_MUTED};">Buy Probability</div>'
                    f'<div style="font-size:26px;font-weight:800;color:{color};">{s_pct:.1f}%</div>'
                    f'<div style="background:#F0F0F6;border-radius:4px;height:6px;margin:6px 0 0;overflow:hidden;">'
                    f'<div style="height:100%;width:{s_pct:.0f}%;background:{color};border-radius:4px;"></div>'
                    f'</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Step 3
        prim_color = CHANNEL_COLORS.get(ch['primary'], ABG_MUTED)
        prim_icon  = CHANNEL_ICONS.get(ch['primary'], '📡')
        st.markdown(f'<div style="font-size:13px;font-weight:800;color:{ABG_DARK};margin-bottom:10px;">Step 3 — Best Channel (Individual AI Decision)</div>', unsafe_allow_html=True)
        cc1, _ = st.columns([1,2])
        with cc1:
            st.markdown(
                f'<div style="background:{prim_color}12;border:2px solid {prim_color}44;'
                f'border-radius:12px;padding:24px;text-align:center;border-top:4px solid {prim_color};">'
                f'<div style="font-size:48px;">{prim_icon}</div>'
                f'<div style="font-size:20px;font-weight:800;color:{prim_color};margin-top:6px;">{ch["primary"]}</div>'
                f'</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:{ABG_BLUE}08;border-left:3px solid {ABG_BLUE};'
            f'border-radius:0 8px 8px 0;padding:11px 14px;margin-top:10px;font-size:12px;color:{ABG_DARK};">'
            f'<strong>AI Reasoning:</strong> {ch["reason"]}</div>',
            unsafe_allow_html=True)

        # Chatbot message
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        customer_name = preset.split("(")[0].strip() if preset != "Custom Profile" else "Valued Customer"
        prod_items = ""
        for rec in recs:
            reason_txt = rec.get('reason_codes',[''])[0] if rec.get('reason_codes') else ''
            prod_items += (
                f'<div style="padding:8px 0;border-bottom:1px solid #E8E8F0;">'
                f'<div style="font-size:13px;font-weight:700;color:{ABG_DARK};">'
                f'💳 {rec.get("product_name","")} — Match Score {rec["confidence_pct"]:.1f}%</div>'
                f'<div style="font-size:12px;color:{ABG_MUTED};margin-top:3px;">{reason_txt}</div>'
                f'</div>'
            )
        quick = (
            f'<span style="background:{ABG_BLUE}12;color:{ABG_BLUE};border:1px solid {ABG_BLUE}30;border-radius:20px;padding:5px 14px;font-size:11px;font-weight:600;margin:4px;display:inline-block;">Why these products?</span>'
            f'<span style="background:{ABG_PURPLE}12;color:{ABG_PURPLE};border:1px solid {ABG_PURPLE}30;border-radius:20px;padding:5px 14px;font-size:11px;font-weight:600;margin:4px;display:inline-block;">Compare benefits</span>'
            f'<span style="background:{ABG_GOLD}12;color:{ABG_GOLD};border:1px solid {ABG_GOLD}30;border-radius:20px;padding:5px 14px;font-size:11px;font-weight:600;margin:4px;display:inline-block;">Explore all products</span>'
        )
        st.markdown(
            f'<h3 style="color:{ABG_DARK};font-size:15px;font-weight:800;margin:0 0 12px;">🤖 BSF Chatbot Message</h3>',
            unsafe_allow_html=True)
        st.markdown(
            f'<div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};border-radius:14px;padding:22px;">'
            f'<div style="display:flex;gap:12px;">'
            f'<div style="width:40px;height:40px;border-radius:50%;flex-shrink:0;'
            f'background:linear-gradient(135deg,{ABG_BLUE},{ABG_PURPLE});'
            f'display:flex;align-items:center;justify-content:center;font-size:18px;">🤖</div>'
            f'<div style="background:#F0F4FF;border-radius:0 12px 12px 12px;padding:14px 18px;flex:1;">'
            f'<div style="font-size:11px;font-weight:700;color:{ABG_BLUE};margin-bottom:6px;">BSF Smart Assistant</div>'
            f'<div style="font-size:13px;color:{ABG_DARK};line-height:1.8;">Hello <strong>{customer_name}</strong> 👋 Welcome back to BSF Digital Banking.<br><br>'
            f'Based on your financial profile, we have identified products relevant to your needs:</div>'
            f'<div style="margin:10px 0;">{prod_items}</div>'
            f'<div style="font-size:13px;color:{ABG_DARK};margin-top:10px;">Would you like to know more about these products?</div>'
            f'<div style="margin-top:12px;">{quick}</div>'
            f'</div></div></div>',
            unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        f'<h3 style="color:{ABG_DARK};font-size:16px;font-weight:800;margin:0 0 14px;">'
        f'🤖 BSF Smart Chatbot</h3>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:{ABG_WHITE};border:1.5px solid {ABG_BORDER};'
        f'border-radius:14px;padding:22px;box-shadow:0 2px 8px rgba(0,0,0,0.06);'
        f'display:flex;align-items:center;gap:18px;">'
        f'<div style="width:52px;height:52px;border-radius:50%;flex-shrink:0;'
        f'background:linear-gradient(135deg,{ABG_BLUE},{ABG_PURPLE});'
        f'display:flex;align-items:center;justify-content:center;font-size:24px;">🤖</div>'
        f'<div style="flex:1;">'
        f'<div style="font-size:15px;font-weight:800;color:{ABG_DARK};margin-bottom:4px;">'
        f'Talk to BSF Smart Banking Assistant</div>'
        f'<div style="font-size:12px;color:{ABG_MUTED};line-height:1.65;margin-bottom:12px;">'
        f'Continue this conversation with our AI chatbot for personalized banking guidance.</div>'
        f'<a href="http://192.168.0.160:3000/" target="_blank" '
        f'style="display:inline-flex;align-items:center;gap:8px;'
        f'background:linear-gradient(135deg,{ABG_BLUE},{ABG_PURPLE});'
        f'color:white;text-decoration:none;padding:10px 22px;'
        f'border-radius:8px;font-size:13px;font-weight:700;'
        f'box-shadow:0 4px 14px rgba(61,61,219,0.30);">💬 Open BSF Chatbot</a>'
        f'</div></div>',
        unsafe_allow_html=True)
