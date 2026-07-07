import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px

# Establish production terminal page layout
st.set_page_config(
    page_title="AlphaNexus | Quantum Trading Terminal", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Premium CSS for high-tech terminal theme, custom animations, and neon color palettes
st.markdown("""
    <style>
    /* Dark Cyberpunk Theme background */
    .main { background-color: #060913; color: #E2E8F0; }
    
    /* Neon glow effect for metric blocks */
    div[data-testid="stMetricValue"] { 
        font-size: 28px !important; 
        font-weight: 800 !important; 
        color: #00FFCC !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
    }
    
    /* Animated Hover Effect for Sidebar items and cards */
    div[data-testid="metric-container"] {
        background-color: #0f1424;
        border: 1px solid #1e293b;
        padding: 15px;
        border-radius: 10px;
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        border-color: #00FFCC;
        box-shadow: 0 4px 20px rgba(0, 255, 204, 0.15);
    }
    
    /* Sleek scrollbar styling */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #00FFCC; }
    </style>
""", unsafe_allow_html=True)

st.title("🌌 ALPHANEXUS QUANT TERMINAL")
st.caption("AI-Driven Predictive Workspace Engine • Real-Time Core Stream Processing")
st.markdown("---")

# ==========================================
# SIDEBAR CONTROLS (Dropdown Ticker Selector)
# ==========================================
st.sidebar.header("🎛️ TERMINAL PARAMETERS")

# CHANGED: Replaced text input with a beautiful preset dropdown list for auto-updating clicks
ticker_choice = st.sidebar.selectbox(
    "Select Target Asset", 
    ["AAPL (Apple)", "MSFT (Microsoft)", "TSLA (Tesla)", "NVDA (NVIDIA)", "SPY (S&P 500 Index)", "BTC-USD (Bitcoin)", "AMZN (Amazon)", "GOOGL (Google)"],
    index=0
)
# Extract just the ticker symbol from the chosen option
ticker_input = ticker_choice.split(" ")[0]

# Interval Configuration
period = st.sidebar.selectbox("Historical Training Window", ["1y", "2y", "5y"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Feature Engineering Lookbacks")
rsi_window = st.sidebar.slider("RSI Spectrum Window", min_value=5, max_value=30, value=14)
ma_short = st.sidebar.slider("Short EMA Signal Line", min_value=5, max_value=50, value=12)
ma_long = st.sidebar.slider("Long EMA Base Line", min_value=10, max_value=100, value=26)

# ==========================================
# DATA INGESTION & PIPELINE ENGINE
# ==========================================
@st.cache_data(ttl=1800)  # Cache for 30 minutes for blazing fast automatic updates
def fetch_and_build_pipeline(ticker, prd, rsi_w, ma_s, ma_l):
    raw_df = yf.download(ticker, period=prd, interval="1d")
    if raw_df.empty:
        return None
    
    df = raw_df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
        
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

# Trigger processing automatically based on user click choice
with st.spinner("Synchronizing with cloud exchange liquidation logs..."):
    df = fetch_and_build_pipeline(ticker_input, period, rsi_window, ma_short, ma_long)
    
if df is None or len(df) < 50:
    st.error("Matrix compilation error. Incomplete financial dataset retrieved.")
else:
    # Train Predictive AI
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

    # Real-Time Statistics Calculations
    final_strat = backtest_df['Cumulative_Strategy_Return'].iloc[-1] * 100
    final_bh = backtest_df['Cumulative_Buy_Hold_Return'].iloc[-1] * 100
    alpha_metric = final_strat - final_bh
    correct_signals = (backtest_df['Predicted_Target'] == backtest_df['Target']).sum()
    win_rate = (correct_signals / len(backtest_df)) * 100

    # Render Side Matrix Stats Panel
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 STRATEGY LIVE RECAP")
    st.sidebar.metric(label="Model Win Probability", value=f"{win_rate:.1f}%")
    st.sidebar.metric(label="Calculated Excess Alpha", value=f"{alpha_metric:+.2f}%", delta=f"Index Bench: {final_bh:.1f}%")
    st.sidebar.metric(label="Strategy Compounded Yield", value=f"{final_strat:.1f}%")

    # ==========================================
    # WORKSPACE OUTPUT LAYOUT (HIGH-TECH TABS)
    # ==========================================
    tab_market, tab_analytics = st.tabs(["⚡ EXCHANGE RADAR TERMINAL", "📈 STRATEGY GROWTH ENGINE"])
    
    with tab_market:
        col_chart, col_feed = st.columns([2, 1])
        
        with col_chart:
            st.subheader(f"📊 Live Candlestick Node Flow • {ticker_input}")
            candle_display = backtest_df.tail(60) # Keep it zoomed in and ultra crisp
            
            fig_candle = go.Figure()
            # Custom neon styled Candlesticks
            fig_candle.add_trace(go.Candlestick(
                x=candle_display.index,
                open=candle_display['Open'], high=candle_display['High'],
                low=candle_display['Low'], close=candle_display['Close'],
                name="Asset Canvas",
                increasing_line_color='#00FFCC', decreasing_line_color='#FF007F'
            ))
            
            # Superimpose holographic triangle signal buy arrows
            buy_signals = candle_display[candle_display['Predicted_Target'] == 1]
            fig_candle.add_trace(go.Scatter(
                x=buy_signals.index, y=buy_signals['Low'] * 0.988,
                mode='markers', name='AI LONG Signal Trigger',
                marker=dict(symbol='triangle-up', size=13, color='#00FFCC', line=dict(color='#FFFFFF', width=1))
            ))
            
            fig_candle.update_layout(
                template="plotly_dark", 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_rangeslider_visible=False, height=450, margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_candle, use_container_width=True)
            
        with col_feed:
            st.subheader("📋 Core Signal Stream Log")
            stream_view = backtest_df[['Close', 'Predicted_Target', 'Prediction_Probability']].tail(8).copy()
            stream_view['Signal Outflow'] = stream_view['Predicted_Target'].apply(lambda x: "🟢 ALLOCATE LONG" if x == 1 else "⚪ POSITION FLAT")
            stream_view['Model Conviction'] = stream_view['Prediction_Probability'].apply(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(stream_view[['Close', 'Signal Outflow', 'Model Conviction']].iloc[::-1], use_container_width=True)
            
    with tab_analytics:
        st.subheader("🛠️ Algorithmic Growth Metrics & Alpha Variance")
        
        fig_equity = go.Figure()
        # High contrast professional neon colors
        fig_equity.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Cumulative_Strategy_Return']*100, mode='lines', name='AI Predictive Alpha Model', line=dict(color='#00FFCC', width=3)))
        fig_equity.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Cumulative_Buy_Hold_Return']*100, mode='lines', name='Passive Benchmark Index', line=dict(color='#FF007F', width=1.5, dash='longdashdot')))
        
        fig_equity.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Simulation Dates", yaxis_title="Growth Vector Percentage (%)",
            height=400, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified"
        )
        st.plotly_chart(fig_equity, use_container_width=True)
