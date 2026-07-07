import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px

# Establish production terminal page layout
st.set_page_config(
    page_title="AlphaDash | Real-Time Quantitative Analytics Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling injection
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    div[data-testid="stMetricValue"] { font-size: 26px; font-weight: 700; color: #00FFB2; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ QUANTITATIVE TERMINAL ENGINE (LIVE)")
st.caption("Powered by Yahoo Finance API & On-The-Fly Random Forest Predictive Framework")
st.markdown("---")

# ==========================================
# SIDEBAR CONTROLS (Dynamic User Inputs)
# ==========================================
st.sidebar.header("🎛️ TERMINAL PARAMETERS")

# Ticker Input matches the yfinance-dash structure
ticker_input = st.sidebar.text_input("Securities (e.g., AAPL, MSFT, TSLA, SPY)", value="AAPL").upper().strip()

# Period & Interval Configuration
period = st.sidebar.selectbox("Historical Training Window", ["1y", "2y", "5y", "max"], index=1)
interval = "1d"

st.sidebar.markdown("---")
st.sidebar.subheader("📈 ML Feature Parameters")
rsi_window = st.sidebar.slider("RSI Lookback Period", min_value=5, max_value=30, value=14)
ma_short = st.sidebar.slider("Short Moving Average (EMA)", min_value=5, max_value=50, value=12)
ma_long = st.sidebar.slider("Long Moving Average (EMA)", min_value=10, max_value=100, value=26)

# ==========================================
# DATA INGESTION & PIPELINE ENGINE
# ==========================================
@st.cache_data(ttl=3600)  # Cache data for 1 hour to optimize execution speed
def fetch_and_build_pipeline(ticker, prd, rsi_w, ma_s, ma_l):
    # Live data extraction
    raw_df = yf.download(ticker, period=prd, interval="1d")
    if raw_df.empty:
        return None
    
    df = raw_df.copy()
    
    # Flatten MultiIndex columns if returned by yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
    # Technical Indicator Calculation Engine
    df['Returns'] = df['Close'].pct_change()
    
    # Exponential Moving Averages
    df['EMA_Short'] = df['Close'].ewm(span=ma_s, adjust=False).mean()
    df['EMA_Long'] = df['Close'].ewm(span=ma_l, adjust=False).mean()
    df['MACD'] = df['EMA_Short'] - df['EMA_Long']
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_w).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_w).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ML Blueprint Targets
    df['Target'] = np.where(df['Returns'].shift(-1) > 0, 1, 0)
    
    df.dropna(inplace=True)
    return df

# Trigger engine
if not ticker_input:
    st.warning("Please enter a valid ticker symbol in the sidebar panel.")
else:
    with st.spinner(f"Querying global exchanges for {ticker_input} and executing model matrix..."):
        df = fetch_and_build_pipeline(ticker_input, period, rsi_window, ma_short, ma_long)
        
    if df is None or len(df) < 50:
        st.error(f"Failed to extract sufficient ticker data for '{ticker_input}'. Verify symbol availability on Yahoo Finance.")
    else:
        # ==========================================
        # LIVE MACHINE LEARNING EXECUTION MODEL
        # ==========================================
        features = ['MACD', 'RSI', 'Close']
        X = df[features].values
        y = df['Target'].values
        
        # Chronological train/test split (80% Train, 20% Backtest Verification)
        split_idx = int(len(df) * 0.8)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Fit Random Forest live on selected asset patterns
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Generate simulation tracking dataset
        backtest_df = df.iloc[split_idx:].copy()
        backtest_df['Predicted_Target'] = model.predict(X_test)
        backtest_df['Prediction_Probability'] = model.predict_proba(X_test)[:, 1]
        
        # Calculate algorithmic returns structure
        backtest_df['Strategy_Returns'] = backtest_df['Predicted_Target'] * backtest_df['Returns']
        backtest_df['Cumulative_Strategy_Return'] = (1 + backtest_df['Strategy_Returns']).cumprod() - 1
        backtest_df['Cumulative_Buy_Hold_Return'] = (1 + backtest_df['Returns']).cumprod() - 1

        # ==========================================
        # RENDER RISK CORE MATRIX & STATS PANEL
        # ==========================================
        final_strat = backtest_df['Cumulative_Strategy_Return'].iloc[-1] * 100
        final_bh = backtest_df['Cumulative_Buy_Hold_Return'].iloc[-1] * 100
        alpha_metric = final_strat - final_bh
        
        # Win-rate tracking calculation
        correct_signals = (backtest_df['Predicted_Target'] == backtest_df['Target']).sum()
        win_rate = (correct_signals / len(backtest_df)) * 100

        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 SECTOR COGNITIVE METRICS")
        st.sidebar.metric(label="Model Directional Accuracy", value=f"{win_rate:.1f}%")
        st.sidebar.metric(label="System Alpha Outperformance", value=f"{alpha_metric:+.2f}%", delta=f"Vs Market: {final_bh:.1f}%")
        st.sidebar.metric(label="Total Generated Yield", value=f"{final_strat:.1f}%")

        # ==========================================
        # INTERACTIVE INTERFACE (TABS & WINDOWS)
        # ==========================================
        tab_market, tab_analytics, tab_raw = st.tabs([
            "🌐 Live Exchange Market Terminal", 
            "📊 Quant Model Performance Deep-Dive",
            "📋 Extracted Feature Stream"
        ])
        
        with tab_market:
            col_chart, col_feed = st.columns([2, 1])
            
            with col_chart:
                st.subheader(f"Interactive Candlestick Stream: {ticker_input}")
                candle_display = backtest_df.tail(90) # Display last 90 trading frames
                
                fig_candle = go.Figure()
                fig_candle.add_trace(go.Candlestick(
                    x=candle_display.index,
                    open=candle_display['Open'], high=candle_display['High'],
                    low=candle_display['Low'], close=candle_display['Close'],
                    name="Price Framework"
                ))
                
                # Overlay live execution triangle markers
                buy_signals = candle_display[candle_display['Predicted_Target'] == 1]
                fig_candle.add_trace(go.Scatter(
                    x=buy_signals.index, y=buy_signals['Low'] * 0.985,
                    mode='markers', name='System LONG Entry',
                    marker=dict(symbol='triangle-up', size=11, color='#00FFB2')
                ))
                
                fig_candle.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=450, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_candle, use_container_width=True)
                
            with col_feed:
                st.subheader("📋 Execution Order Flow")
                stream_view = backtest_df[['Close', 'Predicted_Target', 'Prediction_Probability']].tail(8).copy()
                stream_view['Signal Frame'] = stream_view['Predicted_Target'].apply(lambda x: "🚀 ALLOCATE LONG" if x == 1 else "🛑 HOLD CASH")
                stream_view['Confidence Matrix'] = stream_view['Prediction_Probability'].apply(lambda x: f"{x*100:.1f}%")
                
                st.dataframe(stream_view[['Close', 'Signal Frame', 'Confidence Matrix']].iloc[::-1], use_container_width=True)
                
        with tab_analytics:
            st.subheader("Algorithmic Capital Growth Simulation")
            
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Cumulative_Strategy_Return']*100, mode='lines', name='AI Quant Strategy Vector', line=dict(color='#00FFB2', width=2.5)))
            fig_equity.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Cumulative_Buy_Hold_Return']*100, mode='lines', name='Underlying Benchmark Index', line=dict(color='#FF4B4B', width=1.5, dash='dot')))
            
            fig_equity.update_layout(
                template="plotly_dark",
                xaxis_title="Simulation Dates",
                yaxis_title="Percent Returns Portfolio Growth (%)",
                height=400,
                margin=dict(l=10, r=10, t=10, b=10),
                hovermode="x unified"
            )
            st.plotly_chart(fig_equity, use_container_width=True)
            
        with tab_raw:
            st.subheader("Calculated Historical Technical Features Matrix")
            st.write("Below is the underlying processed feature structure passed into the Artificial Intelligence classifier network.")
            st.dataframe(df[['Close', 'EMA_Short', 'EMA_Long', 'MACD', 'RSI', 'Target']].iloc[::-1], use_container_width=True)
