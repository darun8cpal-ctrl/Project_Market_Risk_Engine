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
# SESSION STATE — tracks which page the user is on
# ──────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"   # home | single | portfolio

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
# PAGE 0 — HOME (mode selection)
# ══════════════════════════════════════════════
if st.session_state.page == "home":

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## Market Risk & Portfolio Optimisation Engine")
    st.markdown("*Quantitative risk analytics — VaR · Expected Shortfall · Markowitz Optimisation*")
    st.markdown("---")
    st.markdown("### Select a mode to get started")
    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon"></div>
            <h2>Single Stock Analysis</h2>
            <p>Analyse the risk of one stock at a time.<br>
            Compute Historical, Parametric & Monte Carlo VaR,
            Expected Shortfall, fat-tail diagnostics,
            and return distribution visualisations.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter Single Stock Analysis →", use_container_width=True, key="btn_single"):
            st.session_state.page = "single"
            st.rerun()

    with col_r:
        st.markdown("""
        <div class="mode-card">
            <div class="mode-icon"></div>
            <h2>Portfolio Optimisation</h2>
            <p>Build and optimise a multi-stock portfolio.<br>
            Explore the Efficient Frontier across 50,000 simulations,
            find the maximum Sharpe portfolio,
            and compute diversified portfolio VaR.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter Portfolio Optimisation →", use_container_width=True, key="btn_port"):
            st.session_state.page = "portfolio"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built by Darun A · BS Economic & Statistical Sciences · IISER Tirupati · 2026")

# ══════════════════════════════════════════════
# PAGE 1 — SINGLE STOCK ANALYSIS
# ══════════════════════════════════════════════
elif st.session_state.page == "single":

    # Back button
    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("## Single Stock Risk Analysis")
    st.markdown("---")

    # ── INPUTS ──────────────────────────────
    st.markdown("### Configure Your Analysis")
    st.markdown("<br>", unsafe_allow_html=True)

    inp1, inp2, inp3 = st.columns(3)

    with inp1:
        st.markdown("**Select Stock**")
        stock = st.selectbox("", TICKERS, index=0, label_visibility="collapsed", key="ss_stock")

    with inp2:
        st.markdown("**Confidence Level (%)**")
        confidence_pct = st.slider("", 90.0, 99.0, 95.0, step=0.5,
                                   label_visibility="collapsed", key="ss_conf")
        confidence = confidence_pct / 100

    with inp3:
        st.markdown("**VaR Methods**")
        methods = st.multiselect("", ["Historical", "Parametric", "Monte Carlo"],
                                 default=["Historical", "Parametric", "Monte Carlo"],
                                 label_visibility="collapsed", key="ss_methods")

    st.markdown("<br>", unsafe_allow_html=True)

    date_col1, date_col2 = st.columns(2)
    with date_col1:
        st.markdown("**Start Date**")
        start_date = st.date_input("", value=pd.to_datetime("2016-01-01"),
                                   label_visibility="collapsed", key="ss_start")
    with date_col2:
        st.markdown("**End Date**")
        end_date = st.date_input("", value=pd.to_datetime("2026-06-25"),
                                 label_visibility="collapsed", key="ss_end")

    st.markdown("<br>", unsafe_allow_html=True)
    run_single = st.button("▶ Run Analysis", type="primary", use_container_width=True, key="run_single")

    if run_single:
        st.markdown("---")

        # Fetch data
        data = get_data((stock,), str(start_date), str(end_date))
        log_ret = np.log(data / data.shift(1)).dropna()
        rets    = log_ret[stock].values

        n_days = len(rets)
        st.caption(f"{stock} · {n_days} trading days · {str(start_date)} → {str(end_date)}")

        # ── STATISTICAL HEALTH ───────────────
        st.markdown("### Statistical Health")
        st.markdown('<div class="info-box">These diagnostics reveal the shape of return distribution and whether standard risk models are reliable for this stock.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        sk_val   = float(skew(rets))
        kurt_val = float(kurtosis(rets))
        jb_text, jb_class = jb_verdict(rets)

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Trading Days", f"{n_days:,}")
        h2.metric("Skewness", f"{sk_val:.4f}")
        h3.metric("Excess Kurtosis", f"{kurt_val:.4f}")
        h4.metric("Daily Std Dev", f"{float(np.std(rets)):.4f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Definitions
        st.markdown("""
        <div class="info-box">
        <b>Skewness</b> — Measures asymmetry of returns.
        Negative skew = more extreme losses than gains (tail risk on the left).
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Excess Kurtosis</b> — Measures how fat the tails are versus a normal distribution.
        Values above 0 mean extreme events happen more often than normal models predict.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Jarque-Bera Test</b> — A statistical test that checks whether returns follow a normal distribution.
        If rejected, parametric VaR underestimates true risk.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="{jb_class}">{jb_text}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── RISK METRICS ─────────────────────
        st.markdown("### Risk Metrics")
        st.markdown('<div class="info-box">All metrics computed at the selected confidence level. A VaR of −2.33% means: on 95% of trading days, loss will not exceed 2.33%.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        results = {}
        if "Historical"  in methods:
            results["Historical VaR"]  = historical_var(rets, confidence)
        if "Parametric"  in methods:
            results["Parametric VaR"]  = parametric_var(rets, confidence)
        if "Monte Carlo" in methods:
            results["Monte Carlo VaR"] = monte_carlo_var(rets, confidence)
        results["Expected Shortfall"] = expected_shortfall(rets, confidence)

        metric_cols = st.columns(len(results))
        for col, (label, val) in zip(metric_cols, results.items()):
            col.metric(label, f"{val:.2%}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Definitions
        definitions = {
            "Historical VaR":  "Uses actual past returns sorted by magnitude — no distributional assumption. Most reliable for fat-tailed assets.",
            "Parametric VaR":  "Assumes returns follow a normal distribution and computes loss analytically using mean and std dev. Fast but underestimates tail risk.",
            "Monte Carlo VaR": "Simulates 10,000 possible future returns randomly using historical mean and volatility. Captures more scenarios than history alone.",
            "Expected Shortfall": "Average loss on the days that breach VaR — answers 'when things go wrong, how bad is it on average?' Always worse than VaR.",
        }
        for label in results:
            if label in definitions:
                st.markdown(f'<div class="info-box"> <b>{label}</b> — {definitions[label]}</div>',
                            unsafe_allow_html=True)

        st.markdown("---")

        # ── DISTRIBUTION PLOT ────────────────
        st.markdown("### Return Distribution")
        st.markdown('<div class="info-box">Histogram of daily log returns. The dashed lines mark risk thresholds. The curve shows what a perfect normal distribution would look like — notice the real data has a taller peak and fatter tails.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        fig, ax = dark_fig(12, 4)
        sns.histplot(rets, bins=60, kde=True, ax=ax,
                     color='#1f6feb', alpha=0.6,
                     line_kws={'linewidth': 2})

        x = np.linspace(rets.min(), rets.max(), 300)
        normal_y = norm.pdf(x, rets.mean(), rets.std()) * len(rets) * (rets.max() - rets.min()) / 60
        ax.plot(x, normal_y, color='#e6edf3', linewidth=1.5,
                linestyle='--', label='Normal curve', alpha=0.7)

        line_colors = {
            'Historical VaR':  '#f85149',
            'Parametric VaR':  '#ff7b72',
            'Monte Carlo VaR': '#ffa657',
            'Expected Shortfall': '#d2a8ff'
        }
        for label, val in results.items():
            ax.axvline(val, color=line_colors.get(label, 'white'),
                       linestyle='--', linewidth=1.5,
                       label=f"{label}: {val:.2%}")

        ax.legend(facecolor='#161b22', edgecolor='#30363d',
                  labelcolor='#e6edf3', fontsize=9)
        ax.set_xlabel("Log Return", color='#8b949e')
        ax.set_ylabel("Count", color='#8b949e')
        ax.set_title(f"{stock} — Return Distribution at {confidence_pct:.1f}% Confidence",
                     color='#e6edf3', pad=12)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")

        # ── VAR BREACH SCATTER ───────────────
        st.markdown("### VaR Breach Scatter")
        st.markdown('<div class="info-box">Each dot is one trading day. Red dots = days where loss exceeded VaR. At 95% confidence, roughly 5% of days should appear red.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        h_var = historical_var(rets, confidence)
        dot_colors = ['#f85149' if r < h_var else '#1f6feb' for r in rets]

        fig2, ax2 = dark_fig(12, 4)
        ax2.scatter(range(len(rets)), rets, c=dot_colors, alpha=0.5, s=5)
        ax2.axhline(h_var, color='#f85149', linestyle='--', linewidth=1.5,
                    label=f"Historical VaR ({confidence_pct:.1f}%): {h_var:.2%}")
        ax2.legend(facecolor='#161b22', edgecolor='#30363d',
                   labelcolor='#e6edf3', fontsize=9)
        ax2.set_xlabel("Trading Day", color='#8b949e')
        ax2.set_ylabel("Log Return",  color='#8b949e')
        ax2.set_title(f"{stock} — Daily Returns · Red = VaR Breach",
                      color='#e6edf3', pad=12)
        st.pyplot(fig2)
        plt.close(fig2)

        n_breaches  = int(np.sum(np.array(rets) < h_var))
        breach_rate = n_breaches / len(rets) * 100
        expected_br = (1 - confidence) * 100

        b1, b2, b3 = st.columns(3)
        b1.metric("VaR Breaches",   f"{n_breaches} days")
        b2.metric("Actual Breach %", f"{breach_rate:.2f}%")
        b3.metric("Expected Breach %", f"{expected_br:.1f}%",
                  delta=f"{breach_rate - expected_br:+.2f}%",
                  delta_color="inverse" if breach_rate > expected_br * 1.2 else "normal")

# ══════════════════════════════════════════════
# PAGE 2 — PORTFOLIO OPTIMISATION
# ══════════════════════════════════════════════
elif st.session_state.page == "portfolio":

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("## Portfolio Optimisation")
    st.markdown("---")

    # ── INPUTS ──────────────────────────────
    st.markdown("### Configure Your Portfolio")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("**Select Stocks** *(choose at least 2)*")
    selected_stocks = st.multiselect(
        "", TICKERS,
        default=TICKERS[:5],
        label_visibility="collapsed",
        key="port_stocks"
    )

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        st.markdown("**Confidence Level (%)**")
        p_conf_pct = st.slider("", 90.0, 99.0, 95.0, step=0.5,
                               label_visibility="collapsed", key="port_conf")
        p_confidence = p_conf_pct / 100

    with pc2:
        st.markdown("**Number of Simulations**")
        n_sim = st.select_slider("", [10_000, 25_000, 50_000, 100_000],
                                 value=50_000,
                                 label_visibility="collapsed", key="port_nsim")

    with pc3:
        st.markdown("**Date Range**")
        p_start = st.date_input("Start", value=pd.to_datetime("2016-01-01"), key="port_start")
        p_end   = st.date_input("End",   value=pd.to_datetime("2026-06-25"), key="port_end")

    st.markdown("<br>", unsafe_allow_html=True)
    run_port = st.button("▶ Run Portfolio Optimisation", type="primary",
                         use_container_width=True, key="run_port")

    if len(selected_stocks) < 2:
        st.info("Please select at least 2 stocks to run portfolio analysis.")

    if run_port and len(selected_stocks) >= 2:
        st.markdown("---")

        # Fetch data
        data = get_data(tuple(selected_stocks), str(p_start), str(p_end))
        log_returns = np.log(data / data.shift(1)).dropna()
        n_stocks    = len(log_returns.columns)
        n_days      = len(log_returns)

        st.caption(f"{n_stocks} stocks · {n_days} trading days · {str(p_start)} → {str(p_end)}")

        # ── CORRELATION HEATMAP ──────────────
        st.markdown("### Correlation Matrix")
        st.markdown("""
        <div class="info-box">
        <b>Correlation</b> — Measures how two stocks move together, scaled from −1 to +1.
        Values near +1 mean they move in sync (less diversification benefit).
        Values near 0 mean they move independently (better diversification).
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        fig_c, ax_c = plt.subplots(figsize=(10, 6))
        fig_c.patch.set_facecolor('#0d1117')
        ax_c.set_facecolor('#0d1117')
        sns.heatmap(log_returns.corr(), annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, ax=ax_c,
                    annot_kws={'size': 8},
                    linewidths=0.5, linecolor='#30363d')
        ax_c.tick_params(colors='#e6edf3', labelsize=8)
        ax_c.set_title("Stock Correlation Matrix", color='#e6edf3', pad=12)
        st.pyplot(fig_c)
        plt.close(fig_c)

        st.markdown("---")

        # ── EFFICIENT FRONTIER ───────────────
        st.markdown("### Efficient Frontier")
        st.markdown("""
        <div class="info-box">
        <b>Efficient Frontier</b> — Each dot represents one possible portfolio (a combination of stock weights).
        The upper-left boundary is the frontier: portfolios that maximise return for a given level of risk.
        Colour shows the Sharpe Ratio — brighter = better risk-adjusted return.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner(f"Simulating {n_sim:,} portfolios..."):
            cov_matrix = log_returns.cov() * 252
            ann_ret    = (log_returns.mean() * 252) - (0.5 * log_returns.var() * 252)

            np.random.seed(42)
            w_mat     = np.random.dirichlet(np.ones(n_stocks), n_sim)
            p_rets    = w_mat @ ann_ret.values
            p_vols    = np.sqrt(np.einsum('ij,jk,ik->i', w_mat, cov_matrix.values, w_mat))
            sharpes   = p_rets / p_vols
            best_idx  = int(np.argmax(sharpes))

        fig_ef, ax_ef = plt.subplots(figsize=(12, 6))
        fig_ef.patch.set_facecolor('#0d1117')
        ax_ef.set_facecolor('#0d1117')

        sc = ax_ef.scatter(p_vols, p_rets, c=sharpes,
                           cmap='viridis', alpha=0.4, s=6)
        ax_ef.scatter(p_vols[best_idx], p_rets[best_idx],
                      color='#f85149', marker='*', s=600,
                      zorder=10, label='Optimal Portfolio (Max Sharpe)')

        cbar = plt.colorbar(sc, ax=ax_ef)
        cbar.set_label('Sharpe Ratio', color='#8b949e')
        cbar.ax.yaxis.set_tick_params(color='#8b949e')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e')

        ax_ef.set_xlabel("Portfolio Volatility (Risk)", color='#8b949e')
        ax_ef.set_ylabel("Portfolio Return",           color='#8b949e')
        ax_ef.tick_params(colors='#8b949e')
        ax_ef.set_title(f"Efficient Frontier — {n_sim:,} Simulated Portfolios",
                        color='#e6edf3', pad=12)
        ax_ef.legend(facecolor='#161b22', edgecolor='#30363d',
                     labelcolor='#e6edf3', fontsize=10)
        for spine in ax_ef.spines.values():
            spine.set_edgecolor('#30363d')

        st.pyplot(fig_ef)
        plt.close(fig_ef)

        st.markdown("---")

        # ── OPTIMAL PORTFOLIO ────────────────
        st.markdown("### Optimal Portfolio")
        st.markdown("""
        <div class="info-box">
        <b>Sharpe Ratio</b> — Return earned per unit of risk taken. Higher is better.
        The optimal portfolio maximises this ratio across all simulated combinations.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        oc1, oc2, oc3 = st.columns(3)
        oc1.metric("Annual Return",     f"{p_rets[best_idx]:.2%}")
        oc2.metric("Annual Volatility", f"{p_vols[best_idx]:.2%}")
        oc3.metric("Sharpe Ratio",      f"{sharpes[best_idx]:.4f}")

        st.markdown("---")

        # ── WEIGHTS ─────────────────────────
        st.markdown("### Optimal Allocation")
        st.markdown("""
        <div class="info-box">
         <b>Portfolio Weights</b> — The percentage of total investment allocated to each stock.
        Markowitz underweights highly correlated stocks and overweights independent, high-return stocks.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        opt_w = w_mat[best_idx]
        w_df  = pd.DataFrame({
            'Stock':    log_returns.columns,
            'Weight':   opt_w,
            'Weight %': (opt_w * 100).round(2)
        }).sort_values('Weight', ascending=False).reset_index(drop=True)

        wc1, wc2 = st.columns([1, 1])

        with wc1:
            fig_pie, ax_pie = plt.subplots(figsize=(7, 7))
            fig_pie.patch.set_facecolor('#0d1117')
            ax_pie.set_facecolor('#0d1117')
            wedges, texts, autotexts = ax_pie.pie(
                w_df['Weight'],
                labels=w_df['Stock'],
                autopct='%1.1f%%',
                pctdistance=0.82,
                colors=plt.cm.tab20.colors[:len(w_df)],
                wedgeprops={'linewidth': 1.5, 'edgecolor': '#0d1117'}
            )
            for t in texts + autotexts:
                t.set_color('#e6edf3')
                t.set_fontsize(8)
            ax_pie.set_title("Portfolio Allocation", color='#e6edf3', pad=12)
            st.pyplot(fig_pie)
            plt.close(fig_pie)

        with wc2:
            st.dataframe(w_df[['Stock', 'Weight %']], use_container_width=True, hide_index=True)

        st.markdown("---")

        # ── PORTFOLIO VAR ────────────────────
        st.markdown("### Portfolio Risk Metrics")
        st.markdown("""
        <div class="info-box">
         <b>Portfolio VaR</b> — The same VaR concept from single stock analysis, 
        but now applied to the combined optimal portfolio of all selected stocks.
        Diversification should make this lower than individual stock VaRs.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        port_daily = (log_returns * opt_w).sum(axis=1).values

        p_hvar  = historical_var(port_daily, p_confidence)
        p_pvar  = parametric_var(port_daily, p_confidence)
        p_mcvar = monte_carlo_var(port_daily, p_confidence)
        p_es    = expected_shortfall(port_daily, p_confidence)

        rv1, rv2, rv3, rv4 = st.columns(4)
        rv1.metric("Historical VaR",     f"{p_hvar:.2%}")
        rv2.metric("Parametric VaR",     f"{p_pvar:.2%}")
        rv3.metric("Monte Carlo VaR",    f"{p_mcvar:.2%}")
        rv4.metric("Expected Shortfall", f"{p_es:.2%}")

        # Diversification benefit
        avg_ind_var = float(np.mean([historical_var(log_returns[s].values, p_confidence)
                                     for s in log_returns.columns]))
        div_benefit = avg_ind_var - p_hvar

        st.markdown("<br>", unsafe_allow_html=True)
        if div_benefit < 0:
            st.markdown(
                f'<div class="safe-box"> <b>Diversification Benefit:</b> '
                f'Portfolio VaR ({p_hvar:.2%}) is <b>{abs(div_benefit):.2%} lower</b> '
                f'than the average individual stock VaR ({avg_ind_var:.2%}). '
                f'Your portfolio is well diversified.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="risk-alert"> Portfolio VaR ({p_hvar:.2%}) vs '
                f'average individual VaR ({avg_ind_var:.2%}). '
                f'Consider adding less correlated stocks.</div>',
                unsafe_allow_html=True
            )

        st.markdown("""
        <div class="info-box">
         <b>Diversification Benefit</b> — When combining stocks with low correlation,
        the portfolio VaR is lower than the average of individual stock VaRs.
        This reduction in risk without sacrificing return is the core insight of Markowitz portfolio theory.
        </div>
        """, unsafe_allow_html=True)