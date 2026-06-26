"""
Market Risk & Portfolio Optimisation Engine
============================================
Author : Darun A — IISER Tirupati · 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, jarque_bera, skew, kurtosis

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Market Risk Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Global font stack for a clean, system-native look */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    /* Refined typography for better hierarchy */
    h1, h2, h3, .mode-card h2 {
        font-family: "SF Pro Display", "Helvetica Neue", Helvetica, Arial, sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    .stApp { background-color: #0d1117; color: #e6edf3; }
    [data-testid="stSidebar"] { display: none; }

    /* Mode selection cards */
    .mode-card {
        background: #161b22;
        border: 2px solid #30363d;
        border-radius: 14px;
        padding: 36px 28px;
        text-align: center;
        cursor: pointer;
        transition: border-color 0.2s;
        height: 220px;
    }
    .mode-card:hover { border-color: #1f6feb; }
    .mode-card h2 { color: #e6edf3; font-size: 1.5rem; margin-bottom: 10px; }
    .mode-card p  { color: #8b949e; font-size: 0.92rem; line-height: 1.6; }
    .mode-icon    { font-size: 2.8rem; margin-bottom: 14px; }

    /* Section card */
    .section-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 18px;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px;
    }

    /* Alert boxes */
    .risk-alert {
        background: #3d1a1a; border-left: 4px solid #f85149;
        padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 8px 0;
    }
    .safe-box {
        background: #1a3d2b; border-left: 4px solid #3fb950;
        padding: 12px 16px; border-radius: 0 6px 6px 0; margin: 8px 0;
    }
    .info-box {
        background: #1a2840; border-left: 4px solid #1f6feb;
        padding: 10px 14px; border-radius: 0 6px 6px 0;
        margin: 4px 0; font-size: 0.85rem; color: #8b949e;
    }

    /* Divider */
    hr { border-color: #30363d; }

    /* Back button styling */
    .back-btn { color: #8b949e; font-size: 0.85rem; cursor: pointer; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        background: #161b22; color: #8b949e;
        border-radius: 6px 6px 0 0; padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #1f6feb !important; color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"

# ──────────────────────────────────────────────
# AVAILABLE TICKERS
# ──────────────────────────────────────────────
TICKERS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'ITC.NS', 'HINDUNILVR.NS', 'AXISBANK.NS', 'BAJFINANCE.NS',
    'BHARTIARTL.NS', 'WIPRO.NS', 'SUNPHARMA.NS', 'ADANIENT.NS',
    'APOLLOHOSP.NS', 'BRITANNIA.NS'
]

# ──────────────────────────────────────────────
# DATA FETCHING
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching market data...")
def get_data(tickers_tuple, start, end):
    raw = yf.download(list(tickers_tuple), start=start, end=end, progress=False)['Close']
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name=list(tickers_tuple)[0])
    return raw.dropna(axis=1, how='all')

# ──────────────────────────────────────────────
# RISK FUNCTIONS
# ──────────────────────────────────────────────
def historical_var(returns, confidence):
    return float(np.percentile(returns, (1 - confidence) * 100))

def parametric_var(returns, confidence):
    mu    = float(np.mean(returns))
    sigma = float(np.std(returns))
    return float(mu + norm.ppf(1 - confidence) * sigma)

def monte_carlo_var(returns, confidence, n_sim=10_000, seed=42):
    rng   = np.random.default_rng(seed)
    mu    = float(np.mean(returns))
    sigma = float(np.std(returns))
    sims  = rng.normal(mu, sigma, n_sim)
    return float(np.percentile(sims, (1 - confidence) * 100))

def expected_shortfall(returns, confidence):
    threshold = historical_var(returns, confidence)
    tail      = returns[returns <= threshold]
    return float(tail.mean()) if len(tail) > 0 else threshold

def jb_verdict(returns):
    _, p = jarque_bera(returns)
    if p < 0.05:
        return "Normality Rejected — Fat Tail Risk Detected (p < 0.05)", "risk-alert"
    return "Returns appear normally distributed (p >= 0.05)", "safe-box"

def dark_fig(w=12, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    return fig, ax

# ══════════════════════════════════════════════
# PAGE 0 — HOME
# ══════════════════════════════════════════════
if st.session_state.page == "home":

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Market Risk & Portfolio Optimisation Engine")
    st.markdown("Quantitative risk analytics — VaR, Expected Shortfall, Markowitz Optimisation")
    st.markdown("---")
    st.markdown("### Select a mode to get started")
    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("""
        <div class="mode-card">
            <h2>Single Stock Analysis</h2>
            <p>Analyse the risk of one stock at a time.<br>
            Compute Historical, Parametric & Monte Carlo VaR,
            Expected Shortfall, fat-tail diagnostics,
            and return distribution visualisations.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter Single Stock Analysis", use_container_width=True, key="btn_single"):
            st.session_state.page = "single"
            st.rerun()

    with col_r:
        st.markdown("""
        <div class="mode-card">
            <h2>Portfolio Optimisation</h2>
            <p>Build and optimise a multi-stock portfolio.<br>
            Explore the Efficient Frontier across 50,000 simulations,
            find the maximum Sharpe portfolio,
            and compute diversified portfolio VaR.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter Portfolio Optimisation", use_container_width=True, key="btn_port"):
            st.session_state.page = "portfolio"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built by Darun A — BS Economic & Statistical Sciences — IISER Tirupati — 2026")

# ══════════════════════════════════════════════
# PAGE 1 — SINGLE STOCK ANALYSIS
# ══════════════════════════════════════════════
elif st.session_state.page == "single":

    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("## Single Stock Risk Analysis")
    st.markdown("---")

    inp1, inp2, inp3 = st.columns(3)
    with inp1:
        st.markdown("**Select Stock**")
        stock = st.selectbox("", TICKERS, index=0, label_visibility="collapsed", key="ss_stock")
    with inp2:
        st.markdown("**Confidence Level (%)**")
        confidence_pct = st.slider("", 90.0, 99.0, 95.0, step=0.5, label_visibility="collapsed", key="ss_conf")
        confidence = confidence_pct / 100
    with inp3:
        st.markdown("**VaR Methods**")
        methods = st.multiselect("", ["Historical", "Parametric", "Monte Carlo"], default=["Historical", "Parametric", "Monte Carlo"], label_visibility="collapsed", key="ss_methods")

    date_col1, date_col2 = st.columns(2)
    with date_col1:
        st.markdown("**Start Date**")
        start_date = st.date_input("", value=pd.to_datetime("2016-01-01"), label_visibility="collapsed", key="ss_start")
    with date_col2:
        st.markdown("**End Date**")
        end_date = st.date_input("", value=pd.to_datetime("2026-06-25"), label_visibility="collapsed", key="ss_end")

    run_single = st.button("Run Analysis", type="primary", use_container_width=True, key="run_single")

    if run_single:
        st.markdown("---")
        data = get_data((stock,), str(start_date), str(end_date))
        
        if stock not in data.columns or data[stock].isna().all():
            st.error(f"No valid data found for {stock}. Please check the date range or select a different stock.")
        else:
            log_ret = np.log(data / data.shift(1)).dropna()
            rets    = log_ret[stock].values
            n_days = len(rets)
            st.caption(f"{stock} — {n_days} trading days — {start_date} to {end_date}")

            st.markdown("### Statistical Health")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Trading Days", f"{n_days:,}")
            h2.metric("Skewness", f"{float(skew(rets)):.4f}")
            h3.metric("Excess Kurtosis", f"{float(kurtosis(rets)):.4f}")
            h4.metric("Daily Std Dev", f"{float(np.std(rets)):.4f}")
            
            jb_text, jb_class = jb_verdict(rets)
            st.markdown(f'<div class="{jb_class}">{jb_text}</div>', unsafe_allow_html=True)
            st.markdown("---")

            st.markdown("### Risk Metrics")
            results = {}
            if "Historical" in methods: results["Historical VaR"] = historical_var(rets, confidence)
            if "Parametric" in methods: results["Parametric VaR"] = parametric_var(rets, confidence)
            if "Monte Carlo" in methods: results["Monte Carlo VaR"] = monte_carlo_var(rets, confidence)
            results["Expected Shortfall"] = expected_shortfall(rets, confidence)

            metric_cols = st.columns(len(results))
            for col, (label, val) in zip(metric_cols, results.items()):
                col.metric(label, f"{val:.2%}")

            st.markdown("---")
            st.markdown("### Return Distribution")
            fig, ax = dark_fig(12, 4)
            sns.histplot(rets, bins=60, kde=True, ax=ax, color='#1f6feb', alpha=0.6)
            st.pyplot(fig)
            plt.close(fig)

# ══════════════════════════════════════════════
# PAGE 2 — PORTFOLIO OPTIMISATION
# ══════════════════════════════════════════════
elif st.session_state.page == "portfolio":
    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("## Portfolio Optimisation")
    st.markdown("---")
    selected_stocks = st.multiselect("Select Stocks (at least 2)", TICKERS, default=TICKERS[:5])

    if len(selected_stocks) < 2:
        st.info("Please select at least 2 stocks.")
    else:
        run_port = st.button("Run Portfolio Optimisation", type="primary")
        if run_port:
            data = get_data(tuple(selected_stocks), "2016-01-01", "2026-06-25")
            log_returns = np.log(data / data.shift(1)).dropna()
            
            st.markdown("### Correlation Matrix")
            fig_c