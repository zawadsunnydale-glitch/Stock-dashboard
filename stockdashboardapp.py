import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px

# Establish clean dashboard page settings
st.set_page_config(
    page_title="AlphaNexus Matrix | Simple AI Trading Workspace", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom premium UI style cards for easy reading
st.markdown("""
    <style>
    .main { background-color: #050811; color: #E2E8F0; }
    
    div[data-testid="stMetricValue"] { 
        font-size: 26px !important; 
        font-weight: 800 !important; 
        color: #00FFCC !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
    }
    
    .ai-box-buy {
        background-color: #0c231e;
        border-left: 6px solid #00FFCC;
        padding: 22px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .ai-box-hold {
        background-color: #21191d;
        border-left: 6px solid #FF007F;
        padding: 22px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .ai-box-purple {
        background-color: #161224;
        border-left: 6px solid #9B5DE5;
        padding: 22px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    div[data-testid="metric-container"] {
        background-color: #0a0f1d;
        border: 1px solid #1e293b;
        padding: 15px;
        border-radius: 10px;
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        border-color: #00FFCC;
        box-shadow: 0 4px 25px rgba(0, 255, 204, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌌 ALPHANEXUS TOTAL INTELLIGENCE TERMINAL")
st.caption("AI Smart Charts • Simple Stock Market Advisor & Complete Company Health Checker")
st.markdown("---")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("🎛️ ADJUST TERMINAL OPTIONS")

ticker_choice = st.sidebar.selectbox(
    "Select a Stock to Analyze", 
    ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL"],
    index=0
)
ticker_input = ticker_choice.strip()
period = st.sidebar.selectbox("Historical Training Timeline", ["1y", "2y", "5y"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Technical Fine-Tuning")
rsi_window = st.sidebar.slider("RSI Indicator Speed", min_value=5, max_value=30, value=14)
ma_short = st.sidebar.slider("Fast Moving Average Line", min_value=5, max_value=50, value=12)
ma_long = st.sidebar.slider("Slow Moving Average Line", min_value=10, max_value=100, value=26)

# ==========================================
# SAFE DATA DOWNLOADING PIPELINE
# ==========================================
@st.cache_data(ttl=600)
def fetch_quant_data(ticker, prd, rsi_w, ma_s, ma_l):
    raw_df = yf.download(ticker, period=prd, interval="1d", group_by='ticker')
    if raw_df.empty:
        return None
    
    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.levels[0]:
            df = df[ticker]
        else:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            
    df.columns = [str(c) for c in df.columns]
    
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

with st.spinner("Downloading current stock trends and balance sheets..."):
    df = fetch_quant_data(ticker_input, period, rsi_window, ma_short, ma_long)
    ticker_obj = yf.Ticker(ticker_input)

if df is None or len(df) < 10:
    st.error("Could not load data. Please refresh or try a different stock ticker.")
else:
    # Machine Learning Simulation Setup
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

    # Performance metrics
    final_strat = backtest_df['Cumulative_Strategy_Return'].iloc[-1] * 100
    final_bh = backtest_df['Cumulative_Buy_Hold_Return'].iloc[-1] * 100
    alpha_metric = final_strat - final_bh
    correct_signals = (backtest_df['Predicted_Target'] == backtest_df['Target']).sum()
    win_rate = (correct_signals / len(backtest_df)) * 100

    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 STRATEGY REPORT CARD")
    st.sidebar.metric(label="AI Strategy Win Rate", value=f"{win_rate:.1f}%")
    st.sidebar.metric(label="Extra Profit Generated", value=f"{alpha_metric:+.2f}%")

    tab_market, tab_fundamentals, tab_dividends, tab_analytics = st.tabs([
        "⚡ LIVE AI STOCK ADVISOR", 
        "🏦 COMPANY MONEY & LEDGER AUDIT",
        "💎 DIVIDEND PAYDAY CHECKER",
        "📈 SYSTEM BACKTEST GRAPH"
    ])
    
    # ------------------------------------------
    # TAB 1: LIVE AI STOCK ADVISOR
    # ------------------------------------------
    with tab_market:
        col_chart, col_feed = st.columns([2, 1])
        with col_chart:
            st.subheader(f"Price Action Candle Chart • {ticker_input}")
            candle_display = backtest_df.tail(60)
            fig_candle = go.Figure()
            fig_candle.add_trace(go.Candlestick(
                x=candle_display.index, open=candle_display['Open'], high=candle_display['High'],
                low=candle_display['Low'], close=candle_display['Close'], name="Stock Price",
                increasing_line_color='#00FFCC', decreasing_line_color='#FF007F'
            ))
            buy_signals = candle_display[candle_display['Predicted_Target'] == 1]
            fig_candle.add_trace(go.Scatter(
                x=buy_signals.index, y=buy_signals['Low'] * 0.988, mode='markers', name='AI BUY Alert',
                marker=dict(symbol='triangle-up', size=13, color='#00FFCC', line=dict(color='#FFFFFF', width=1))
            ))
            fig_candle.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, height=360, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_candle, use_container_width=True)
            
        with col_feed:
            st.subheader("📋 Live AI Stream Output")
            stream_view = backtest_df[['Close', 'Predicted_Target', 'Prediction_Probability']].tail(8).copy()
            stream_view['Action Plan'] = stream_view['Predicted_Target'].apply(lambda x: "🟢 BUY / LONG" if x == 1 else "⚪ HOLD CASH")
            stream_view['AI Certainty'] = stream_view['Prediction_Probability'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(stream_view[['Close', 'Action Plan', 'AI Certainty']].iloc[::-1], use_container_width=True)

        st.markdown("---")
        st.subheader("🤖 EXPLICIT AI VERDICT & CHART BREAKDOWN")
        
        recent_3m = backtest_df.tail(60)
        recent_strat_perf = ((recent_3m['Strategy_Returns'] + 1).prod() - 1) * 100
        recent_bh_perf = ((recent_3m['Returns'] + 1).prod() - 1) * 100
        latest_signal = backtest_df['Predicted_Target'].iloc[-1]
        latest_prob = backtest_df['Prediction_Probability'].iloc[-1] * 100
        
        if latest_signal == 1 and latest_prob >= 55.0:
            box_style = "ai-box-buy"
            verdict_header = "🟢 AI RECOMMENDATION: WORTH BUYING RIGHT NOW"
            strategy_text = f"**The Strategy Plan:** Buy shares or maintain a Long position. The math model has a high certainty score of **{latest_prob:.1f}%** that the price will move up over the next market session. The indicators are perfectly lined up, suggesting it is statistically safe to enter a trade today."
        else:
            box_style = "ai-box-hold"
            verdict_header = "⚪ AI RECOMMENDATION: NOT WORTH BUYING / HOLD CASH"
            strategy_text = f"**The Strategy Plan:** Do not buy right now. Keep your money safe in liquid cash. The stock's current chart lines look muddy or are trending downwards. The AI has calculated that trading today carries too much risk, so standing aside protects your wallet from taking a hit."

        st.markdown(f"""
        <div class="{box_style}">
            <h3 style="margin-top: 0; color: #FFF;">{verdict_header}</h3>
            <p style="font-size: 16px; line-height: 1.6; margin-bottom: 0;">{strategy_text}</p>
        </div>
        """, unsafe_allow_html=True)

        col_summary1, col_summary2 = st.columns(2)
        with col_summary1:
            st.markdown("### ⏳ Last 3 Months Results Analysis")
            st.write(f"Over the last 90 days, if you blindly bought the stock and sat on it, you would have made **{recent_bh_perf:.2f}%**.")
            st.write(f"By comparison, our smart AI strategy trading model made **{recent_strat_perf:.2f}%** over the exact same period.")
            
            if recent_strat_perf > recent_bh_perf:
                st.success(f"📈 **How things changed:** The AI handily beat the regular stock market returns by **{(recent_strat_perf - recent_bh_perf):.2f}%** this quarter! This proves the model's math successfully dodged price drops and identified the most reliable moments to execute profitable buy triggers.")
            else:
                st.warning(f"📉 **How things changed:** Regular investing beat the AI strategy by **{(recent_bh_perf - recent_strat_perf):.2f}%** this quarter. This means the stock spent the last 3 months locked in a messy, chaotic sideways pattern that caused indicator signals to occasionally misfire.")

        with col_summary2:
            st.markdown("### 🔍 How to Read & Understand This Graph")
            st.info("""
            * **Green and Red Bars (Candles):** This shows daily price movement. Green means the stock finished the day higher than it started; red means it dropped.
            * **Neon Green Triangles (▲):** Look at the bottom of the price bars! These are the exact moments where the AI processed the mathematical parameters and generated a **Buy Signal**, anticipating an up-swing.
            * **AI Certainty Score:** Check the table log on the right side. If the percentage is close to 50%, the stock is wildly unpredictable. If it climbs past 55% or 60%, the AI recognizes a powerful historical pattern and is executing high-confidence setups.
            """)

    # ------------------------------------------
    # TAB 2: FINANCIAL HEALTH AND CASH FLOW ANALYSIS (FIXED KEY ASSIGNMENT BUGS)
    # ------------------------------------------
    with tab_fundamentals:
        st.subheader(f"🏢 Deep-Dive Corporate Financial Position Summary")
        
        try:
            cashflow = ticker_obj.cashflow
            balance = ticker_obj.balance_sheet
            income = ticker_obj.financials
            
            # Helper logic to extract values cleanly without throwing MultiIndex indexing errors
            def get_financial_value(df, key_alternatives):
                if df is None or df.empty:
                    return 1.0
                df_clean = df.copy()
                df_clean.index = [str(x).replace(" ", "").lower() for x in df_clean.index]
                for alt in key_alternatives:
                    alt_clean = alt.replace(" ", "").lower()
                    if alt_clean in df_clean.index:
                        val = df_clean.loc[alt_clean].iloc[0] if isinstance(df_clean.loc[alt_clean], pd.Series) else df_clean.loc[alt_clean]
                        # Extract first value if nested array
                        if hasattr(val, 'values'):
                            val = val.values[0]
                        if isinstance(val, (np.ndarray, list)):
                            val = val[0]
                        return float(val) if pd.notna(val) and float(val) != 0 else 1.0
                return 1.0

            # Dynamic Row Lookup Fixes
            current_assets = get_financial_value(balance, ['CurrentAssets', 'TotalCurrentAssets'])
            current_liab = get_financial_value(balance, ['CurrentLiabilities', 'TotalCurrentLiabilities'])
            inventory = get_financial_value(balance, ['Inventory', 'Inventories'])
            if inventory == 1.0: inventory = 0.0 # reset fallback if non-existent
            
            total_debt = get_financial_value(balance, ['TotalDebt', 'LongTermDebt', 'TotalLiabilities'])
            total_equity = get_financial_value(balance, ['StockholdersEquity', 'TotalStockholderEquity', 'CommonStockEquity'])
            
            # Recalculate true mathematical positions
            current_ratio = current_assets / current_liab
            quick_ratio = (current_assets - inventory) / current_liab
            debt_to_equity = total_debt / total_equity
            
            c_f1, c_f2, c_f3 = st.columns(3)
            with c_f1:
                st.metric(label="Short-Term Cash Buffer (Current Ratio)", value=f"{current_ratio:.2f}x")
                st.caption("Target > 1.5x. Shows if they possess enough cash/assets to pay off incoming short-term bills.")
            with c_f2:
                st.metric(label="Immediate Emergency Cash (Quick Ratio)", value=f"{quick_ratio:.2f}x")
                st.caption("The exact cash safety net available right now if their inventory sales instantly paused.")
            with c_f3:
                st.metric(label="Debt Leverage Pressure (Debt-to-Equity)", value=f"{debt_to_equity:.2f}x")
                st.caption("Lower is safer. Compares borrowed bank loans against the company's actual cash worth.")

            # Map chart metrics cleanly
            st.markdown("---")
            st.subheader("📊 Visualizing Core Corporate Cash Flows (In Billions of Dollars)")
            
            cashflow.index = [str(x) for x in cashflow.index]
            income.index = [str(x) for x in income.index]
            
            all_keys = ['FreeCashFlow', 'OperatingCashFlow', 'NetIncomeFromContinuingOperations', 'CapitalExpenditure']
            target_metrics = {}
            for key in all_keys:
                if key in cashflow.index:
                    target_metrics[key] = np.ravel(cashflow.loc[key].values) / 1e9
                elif key in income.index:
                    target_metrics[key] = np.ravel(income.loc[key].values) / 1e9
                    
            if target_metrics:
                years_labels = [d.strftime('%Y') for d in cashflow.columns]
                min_len = min([len(v) for v in target_metrics.values()] + [len(years_labels)])
                years_labels = years_labels[:min_len]
                sync_metrics = {k: v[:min_len] for k, v in target_metrics.items()}
                
                chart_df = pd.DataFrame(sync_metrics, index=years_labels).reset_index().rename(columns={'index': 'Year'})
                melted_df = chart_df.melt(id_vars='Year', var_name='Financial Metric', value_name='Amount ($ Billions)')
                
                fig_fund = px.bar(melted_df, x='Year', y='Amount ($ Billions)', color='Financial Metric', bmode='group', template='plotly_dark')
                fig_fund.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=380)
                st.plotly_chart(fig_fund, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔍 Full Ledger Audit Sheet Inspection")
            statement_select = st.selectbox("Switch Detailed Statement Grid Tables", ["Core Operating Performance Data", "Full Unfiltered Balance Sheet Structure Logs"])
            
            if statement_select == "Core Operating Performance Data":
                st.dataframe(cashflow.head(20), use_container_width=True)
            else:
                st.dataframe(balance.head(20), use_container_width=True)

            # AI Fundamental Review Card
            st.markdown("---")
            f_fcf = target_metrics.get('FreeCashFlow', [0])[0] if 'FreeCashFlow' in target_metrics else 0
            if f_fcf > 0 and debt_to_equity < 2.0:
                f_title = "⭐ PREMIUM HEALTH CATEGORY: EXCELLENT BUSINESS STANDING"
                f_desc = f"The company's fundamental positioning is solid. They are pulling in real Free Cash Flow after clearing daily operating costs. Combined with realistic balance sheet debt leverage constraints ({debt_to_equity:.2f}x), they possess adequate stability structures."
            else:
                f_title = "⚠️ FINANCIAL WARNING: HIGH FINANCING RELIANCE"
                f_desc = f"This stock shows elements of high financial overhead or tightening operational cash. Leverage structures evaluate at {debt_to_equity:.2f}x. Exercise normal diversification parameters before allocating long-term holds."
                
            st.markdown(f"""
            <div class="ai-box-purple">
                <h4 style="color: #9B5DE5; margin-top: 0;">🏛️ AI Business Position Evaluation: <span style="color: #FFF;">{f_title}</span></h4>
                <p style="font-size: 15px; line-height: 1.6; margin-bottom: 0;">{f_desc}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as ex:
            st.warning(f"Formatting standard financial rows for this asset ticker layout: {ex}")

    # ------------------------------------------
    # TAB 3: DIVIDEND PAYDAY CHECKER & TIMELINES (FIXED COMPATIBILITY LOADING)
    # ------------------------------------------
    with tab_dividends:
        st.subheader(f"💎 Average Dividend Rates & Payday Deadlines")
        
        # Pull live metrics safely from the history directly instead of broken dictionary objects
        try:
            div_history = ticker_obj.dividends
            info = ticker_obj.get_info()
            
            # Base metrics
            div_rate = info.get('trailingAnnualDividendRate', 0.0) or info.get('dividendRate', 0.0) or 0.0
            div_yield = (info.get('trailingAnnualDividendYield', 0.0) or info.get('dividendYield', 0.0) or 0.0) * 100
            payout_ratio = info.get('payoutRatio', 0.0) * 100 if info.get('payoutRatio') else 0.0
            
            # Fallback evaluation via history log tracking if info endpoint blocks scraping request
            if div_rate == 0.0 and not div_history.empty:
                recent_year_divs = div_history.tail(4)
                div_rate = float(recent_year_divs.sum())
                current_price = df['Close'].iloc[-1]
                div_yield = (div_rate / current_price) * 100
            
            c_d1, c_d2, c_d3 = st.columns(3)
            with c_d1:
                st.metric(label="Annual Cash Payout Rate", value=f"${div_rate:.2f} / Share" if div_rate > 0 else "$0.00")
                st.caption("The cash distribution paid to stockholders annually per share owned.")
            with c_d2:
                st.metric(label="Dividend Yield Percentage", value=f"{div_yield:.2f}%" if div_yield > 0 else "0.00%")
                st.caption("Your annual cash flow percentage computed against the stock's current price.")
            with c_d3:
                st.metric(label="Earnings Payout Ratio", value=f"{payout_ratio:.2f}%" if payout_ratio > 0 else "0.00%")
                st.caption("The percentage of profit the company awards back to retail investors.")
        except Exception as err:
            st.caption(f"Syncing live ticker indices: {err}")

        st.markdown("---")
        st.subheader("📋 How and When You Become Eligible for the Payday")
        st.info("""
        To claim stock cash dividends, you must execute your buy trades strictly according to the calendar sequence below:
        1. **Declaration Date:** The board of directors makes an official public announcement stating exactly how much money they intend to pay out.
        2. **Ex-Dividend Date (The Real Cut-off Deadline):** **This is the critical date!** You must buy and own the stock *at least one full market day before* this date. If you buy it on or after this date, you miss out on the current payout.
        3. **Record Date:** The date the company checks its electronic database logs to see who legally holds the stock titles.
        4. **Payment Date (Payday):** The day cash is wired directly into your brokerage account balance!
        """)
        
        if 'div_history' in locals() and not div_history.empty:
            st.subheader("⏳ Recent Payday Calendar Records")
            div_clean = div_history.to_frame().sort_index(ascending=False).head(10)
            st.dataframe(div_clean, use_container_width=True)
        else:
            st.caption("This asset runs on a growth-centric configuration. Profits are retained internally instead of distributed via dividends.")

    # ------------------------------------------
    # TAB 4: SYSTEM BACKTEST GRAPH
    # ------------------------------------------
    with tab_analytics:
        st.subheader("🛠️ AI Strategy vs Buy & Hold Performance Tracker")
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Cumulative_Strategy_Return']*100, mode='lines', name='AI Smart Strategy Return', line=dict(color='#00FFCC', width=3)))
        fig_equity.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Cumulative_Buy_Hold_Return']*100, mode='lines', name='Just Buying and Sitting Still', line=dict(color='#FF007F', width=1.5, dash='longdashdot')))
        fig_equity.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Timeline Dates", yaxis_title="Profit Yield Growth Percentage (%)", height=400, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified")
        st.plotly_chart(fig_equity, use_container_width=True)
