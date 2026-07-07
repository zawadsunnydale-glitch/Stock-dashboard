import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px

# Establish production terminal page layout
st.set_page_config(
    page_title="AlphaNexus Matrix | Multi-Asset Intelligence Terminal", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Advanced CSS styling injection for premium glassmorphism layouts and color structures
st.markdown("""
    <style>
    .main { background-color: #050811; color: #E2E8F0; }
    
    /* Glowing metric tags */
    div[data-testid="stMetricValue"] { 
        font-size: 26px !important; 
        font-weight: 800 !important; 
        color: #00FFCC !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
    }
    
    /* Institutional insights box panels */
    .ai-card {
        background-color: #0d1326;
        border-left: 5px solid #00FFCC;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    .ai-card-fundamental {
        background-color: #0d1326;
        border-left: 5px solid #9B5DE5;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    
    /* Interactive metric hovering animation matrix */
    div[data-testid="metric-container"] {
        background-color: #0a0f1d;
        border: 1px solid #1e293b;
        padding: 15px;
        border-radius: 10px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        border-color: #00FFCC;
        box-shadow: 0 4px 25px rgba(0, 255, 204, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌌 ALPHANEXUS TOTAL INTELLIGENCE TERMINAL")
st.caption("Cross-Paradigm Engine • Live Quantitative Backtesting & Institutional Deep-Dive Fundamental Core")
st.markdown("---")

# ==========================================
# SIDEBAR RADAR PARAMETERS
# ==========================================
st.sidebar.header("🎛️ TERMINAL PARAMETERS")

ticker_choice = st.sidebar.selectbox(
    "Select Target Asset Cluster", 
    ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL"],
    index=0
)
ticker_input = ticker_choice.strip()
period = st.sidebar.selectbox("Historical Training Window", ["1y", "2y", "5y"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Feature Space Offsets")
rsi_window = st.sidebar.slider("RSI Spectrum Window", min_value=5, max_value=30, value=14)
ma_short = st.sidebar.slider("Short EMA Signal Line", min_value=5, max_value=50, value=12)
ma_long = st.sidebar.slider("Long EMA Base Line", min_value=10, max_value=100, value=26)

# ==========================================
# FIXED DYNAMIC PIPELINE EXTRACTION MATRIX
# ==========================================
@st.cache_data(ttl=600)
def fetch_quant_data(ticker, prd, rsi_w, ma_s, ma_l):
    # Quant pricing data - download cleanly using group_by to force format handling
    raw_df = yf.download(ticker, period=prd, interval="1d", group_by='ticker')
    if raw_df.empty:
        return None
    
    df = raw_df.copy()
    
    # Advanced extraction to flatten modern yfinance MultiIndex variations completely
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.levels[0]:
            df = df[ticker]
        else:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
    # Explicit backup fallback if column headers are strings instead of objects
    df.columns = [str(c) for c in df.columns]
    
    # Core Data Calculations
    df['Returns'] = df['Close'].pct_change()
    df['EMA_Short'] = df['Close'].ewm(span=ma_s, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=ma_l, adjust=False).mean()
    df['MACD'] = df['EMA_Short'] - df['EMA_Long']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_w).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_w).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Target'] = np.where(df['Returns'].shift(-1) > 0, 1, 0)
    df.dropna(inplace=True)
    return df

with st.spinner("Compiling cross-paradigm databases and auditing financial log streams..."):
    df = fetch_quant_data(ticker_input, period, rsi_window, ma_short, ma_long)
    ticker_obj = yf.Ticker(ticker_input)

if df is None or len(df) < 10:
    st.error("Terminal initialization failure. Target asset node rejected connection or returned empty rows.")
    st.info("💡 Pro-Tip: Make sure your app settings point exactly to 'stockdashboardapp.py' inside your Streamlit Cloud deploy console.")
else:
    # Train Predictive Quant Classifier Model
    features = ['MACD', 'RSI', 'Close']
    X = df[features].values
    y = df['Target'].values
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    backtest_df = df.iloc[split_idx:].copy()
    backtest_df['Predicted_Target'] = model.predict(X_test)
    backtest_df['Prediction_Probability'] = model.predict_proba(X_test)[:, 1]
    
    backtest_df['Strategy_Returns'] = backtest_df['Predicted_Target'] * backtest_df['Returns']
    backtest_df['Cumulative_Strategy_Return'] = (1 + backtest_df['Strategy_Returns']).cumprod() - 1
    backtest_df['Cumulative_Buy_Hold_Return'] = (1 + backtest_df['Returns']).cumprod() - 1

    # Macro Return Data Metrics
    final_strat = backtest_df['Cumulative_Strategy_Return'].iloc[-1] * 100
    final_bh = backtest_df['Cumulative_Buy_Hold_Return'].iloc[-1] * 100
    alpha_metric = final_strat - final_bh
    correct_signals = (backtest_df['Predicted_Target'] == backtest_df['Target']).sum()
    win_rate = (correct_signals / len(backtest_df)) * 100

    # Populate Live System Performance on Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 SYSTEM COGNITION OVERVIEW")
    st.sidebar.metric(label="Model Win Probability", value=f"{win_rate:.1f}%")
    st.sidebar.metric(label="Excess Alpha Variance", value=f"{alpha_metric:+.2f}%", delta=f"Index: {final_bh:.1f}%")

    # ==========================================
    # WORKSPACE OUTPUT PANELS (CORE INTERFACE TABS)
    # ==========================================
    tab_market, tab_fundamentals, tab_dividends, tab_analytics = st.tabs([
        "⚡ EXCHANGE EXECUTION RADAR", 
        "🏦 INSTITUTIONAL FUNDAMENTAL CORE",
        "💎 DIVIDEND MILESTONE MATRIX",
        "📈 EQUITY VECTOR ANALYTICS"
    ])
    
    # ------------------------------------------
    # TAB 1: TECH RADAR TERMINAL
    # ------------------------------------------
    with tab_market:
        col_chart, col_feed = st.columns([2, 1])
        with col_chart:
            st.subheader(f"Interactive Candlestick Flux Matrix • {ticker_input}")
            candle_display = backtest_df.tail(60)
            fig_candle = go.Figure()
            fig_candle.add_trace(go.Candlestick(
                x=candle_display.index, open=candle_display['Open'], high=candle_display['High'],
                low=candle_display['Low'], close=candle_display['Close'], name="Price Framework",
                increasing_line_color='#00FFCC', decreasing_line_color='#FF007F'
            ))
            buy_signals = candle_display[candle_display['Predicted_Target'] == 1]
            fig_candle.add_trace(go.Scatter(
                x=buy_signals.index, y=buy_signals['Low'] * 0.988, mode='markers', name='System LONG Entry',
                marker=dict(symbol='triangle-up', size=13, color='#00FFCC', line=dict(color='#FFFFFF', width=1))
            ))
            fig_candle.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, height=380, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_candle, use_container_width=True)
            
        with col_feed:
            st.subheader("📋 Active Pipeline Stream")
            stream_view = backtest_df[['Close', 'Predicted_Target', 'Prediction_Probability']].tail(8).copy()
            stream_view['Signal Outflow'] = stream_view['Predicted_Target'].apply(lambda x: "🟢 ALLOCATE LONG" if x == 1 else "⚪ POSITION FLAT")
            stream_view['Model Conviction'] = stream_view['Prediction_Probability'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(stream_view[['Close', 'Signal Outflow', 'Model Conviction']].iloc[::-1], use_container_width=True)

        st.markdown("---")
        recent_3m = backtest_df.tail(60)
        recent_strat_perf = (recent_3m['Strategy_Returns'] + 1).prod() - 1
        recent_bh_perf = (recent_3m['Returns'] + 1).prod() - 1
        latest_signal = backtest_df['Predicted_Target'].iloc[-1]
        latest_prob = backtest_df['Prediction_Probability'].iloc[-1] * 100
        
        if latest_signal == 1 and latest_prob >= 55.0:
            action_plan = "🚀 ACTIVE ALLOCATION CRITERIA MATCHED: BUY / LONG"
            strategy_advice = f"The Random Forest model outputs an elevated **{latest_prob:.1f}% conviction level**. Technical indicator configurations imply structural alpha velocity is opening a positive window for immediate spot accumulation."
        else:
            action_plan = "🛑 PROTECT RESERVES: LIQUID CASH POSITION"
            strategy_advice = f"Predictive matrices register defensive headwinds or directionless baseline chop. The alpha generation model commands an absolute transition into secure cash hedges to prevent drawdown volatility."

        st.markdown(f"""
        <div class="ai-card">
            <h4 style="color: #00FFCC; margin-top: 0;">🔮 Automated Trade Execution Brief: <span style="color: #FFF;">{action_plan}</span></h4>
            <p style="font-size: 15px; line-height: 1.6; margin-bottom: 0;">{strategy_advice}</p>
        </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------
    # TAB 2: INSTITUTIONAL FUNDAMENTAL CORE
    # ------------------------------------------
    with tab_fundamentals:
        st.subheader(f"🏢 Multi-Statement Ledger Audit Architecture: {ticker_input}")
        
        try:
            cashflow = ticker_obj.cashflow
            balance = ticker_obj.balance_sheet
            income = ticker_obj.financials
            
            # Re-index sheets dynamically with string conversion to ensure access matching
            cashflow.index = [str(x) for x in cashflow.index]
            balance.index = [str(x) for x in balance.index]
            income.index = [str(x) for x in income.index]
            
            current_assets = balance.loc['CurrentAssets'].iloc[0] if 'CurrentAssets' in balance.index else 1
            current_liab = balance.loc['CurrentLiabilities'].iloc[0] if 'CurrentLiabilities' in balance.index else 1
            inventory = balance.loc['Inventory'].iloc[0] if 'Inventory' in balance.index else 0
            total_debt = balance.loc['TotalDebt'].iloc[0] if 'TotalDebt' in balance.index else 1
            total_equity = balance.loc['StockholdersEquity'].iloc[0] if 'StockholdersEquity' in balance.index else 1
            
            current_ratio = current_assets / current_liab
            quick_ratio = (current_assets - inventory) / current_liab
            debt_to_equity = total_debt / total_equity
            
            c_liq1, c_liq2, c_liq3 = st.columns(3)
            with c_liq1:
                st.metric(label="Current Ratio (Short-Term Liquidity)", value=f"{current_ratio:.2f}x")
            with c_liq2:
                st.metric(label="Quick Ratio (Acid-Test Matrix)", value=f"{quick_ratio:.2f}x")
            with c_liq3:
                st.metric(label="Debt-to-Equity Leverage Ratio", value=f"{debt_to_equity:.2f}x")

            st.markdown("---")
            st.subheader("📊 Primary Operational Vector Scaling (Historical Multi-Period Analysis)")
            
            target_metrics = {}
            possible_keys = [
                'FreeCashFlow', 'RepurchaseOfCapitalStock', 'RepaymentOfDebt', 'IssuanceOfDebt',
                'CapitalExpenditure', 'OperatingCashFlow', 'NetIncomeFromContinuingOperations',
                'DepreciationAndAmortization', 'ChangeInWorkingCapital'
            ]
            
            for key in possible_keys:
                if key in cashflow.index:
                    target_metrics[key] = np.ravel(cashflow.loc[key].values) / 1e9
                elif key in income.index:
                    target_metrics[key] = np.ravel(income.loc[key].values) / 1e9
                    
            if target_metrics:
                years_labels = [d.strftime('%Y') for d in cashflow.columns]
                # Sync array lengths uniformly
                min_len = min([len(v) for v in target_metrics.values()] + [len(years_labels)])
                years_labels = years_labels[:min_len]
                sync_metrics = {k: v[:min_len] for k, v in target_metrics.items()}
                
                chart_df = pd.DataFrame(sync_metrics, index=years_labels).reset_index().rename(columns={'index': 'Year'})
                melted_df = chart_df.melt(id_vars='Year', var_name='Ledger Metric', value_name='Amount ($ Billions)')
                
                fig_fund = px.bar(melted_df, x='Year', y='Amount ($ Billions)', color='Ledger Metric', bmode='group', template='plotly_dark')
                fig_fund.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=380)
                st.plotly_chart(fig_fund, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔍 Production Statement Ledger Data Stream Subsections")
            st.dataframe(cashflow.head(20), use_container_width=True)

            st.markdown("---")
            f_fcf = target_metrics.get('FreeCashFlow', [0])[0]
            if f_fcf > 0 and debt_to_equity < 1.2:
                f_health = "⭐ PREMIUM HEALTH CATEGORY: CAPITAL SUPREMACY"
                f_analysis = f"Corporate fundamental analysis establishes outstanding free cash allocations ({f_fcf:.2f}B latest). Debt ratios ({debt_to_equity:.2f}x) verify low balance-sheet distress thresholds."
            else:
                f_health = "⚠️ MODERATE TO HIGH RISK FINANCIAL OVERHEAD DETECTED"
                f_analysis = f"The target corporate frame is navigating structural balance adaptations. Leverage structures evaluate at {debt_to_equity:.2f}x."
                
            st.markdown(f"""
            <div class="ai-card-fundamental">
                <h4 style="color: #9B5DE5; margin-top: 0;">🏛️ Fundamental Risk Analysis: <span style="color: #FFF;">{f_health}</span></h4>
                <p style="font-size: 15px; line-height: 1.6; margin-bottom: 0;">{f_analysis}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as ex:
            st.warning(f"Fundamental structural indices currently sorting baseline tables for this ticker asset layer: {ex}")

    # ------------------------------------------
    # TAB 3: DIVIDEND MILESTONE MATRIX
    # ------------------------------------------
    with tab_dividends:
        st.subheader(f"💎 Yield Distributions & Shareholder Governance Metrics")
        try:
            info = ticker_obj.info
            div_rate = info.get('dividendRate', 0.0) if info.get('dividendRate') is not None else 0.0
            div_yield = (info.get('dividendYield', 0.0) * 100) if info.get('dividendYield') is not None else 0.0
            payout_ratio = (info.get('payoutRatio', 0.0) * 100) if info.get('payoutRatio') is not None else 0.0
            
            c_div1, c_div2, c_div3 = st.columns(3)
            with c_div1:
                st.metric(label="Trailing Dividend Rate", value=f"${div_rate:.2f} / Share")
            with c_div2:
                st.metric(label="Calculated Forward Yield", value=f"{div_yield:.2f}%")
            with c_div3:
                st.metric(label="Capital Payout Ratio", value=f"{payout_ratio:.2f}%")
        except:
            st.caption("Dividend metrics loading dynamically...")

        st.markdown("---")
        try:
            div_history = ticker_obj.dividends
            if not div_history.empty:
                st.subheader("⏳ Historical Distribution Timeline Registers")
                div_clean = div_history.to_frame().sort_index(ascending=False).head(10)
                st.dataframe(div_clean, use_container_width=True)
            else:
                st.caption("Asset ticker maintains non-distributing or zero regular cash dividend operations.")
        except:
            st.caption("Historical ledger parsing skipped.")

    # ------------------------------------------
    # TAB 4: DEEP METRICS EQUITY BACKTEST ANALYTICS
    # ------------------------------------------
    with tab_analytics:
        st.subheader("🛠️ Strategy Equity Growth Variance Engine")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Cumulative_Strategy_Return']*100, mode='lines', name='AI Predictive Alpha Model', line=dict(color='#00FFCC', width=3)))
        fig_equity.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Cumulative_Buy_Hold_Return']*100, mode='lines', name='Passive Benchmark Index', line=dict(color='#FF007F', width=1.5, dash='longdashdot')))
        fig_equity.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Simulation Dates", yaxis_title="Growth Vector Percentage (%)", height=400, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        st.plotly_chart(fig_equity, use_container_width=True)
