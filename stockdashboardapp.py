import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Stock Return Prediction Dashboard")

# Load the data you uploaded to GitHub
df = pd.read_csv("simulation_results.csv", parse_dates=True)
if "Date" in df.columns:
    df.set_index("Date", inplace=True)

st.subheader("Simulated Equity Curve")
st.write("This chart shows how your AI strategy performed compared to just buying and holding.")

# Create a clean line chart using your spreadsheet columns
fig = px.line(df, y=["Cumulative_Strategy_Return", "Cumulative_Buy_Hold_Return"],
              labels={"value": "Return", "variable": "Strategy"})
st.plotly_chart(fig)
