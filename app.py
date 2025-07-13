import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

# Page configuration
st.set_page_config(page_title="📈 Stock Market Dashboard", layout="wide")

# Title
st.title("📈 Stock Market Dashboard")

# Sidebar Inputs
st.sidebar.header("Search Stock")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL").upper()
period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=3)
interval = st.sidebar.selectbox("Select Interval", ["1d", "1wk", "1mo"], index=0)

# Fetch data
if ticker:
    try:
        data = yf.download(ticker, period=period, interval=interval)

        if data.empty:
            st.error(f"❌ No data found for ticker '{ticker}' with selected options.")
        else:
            st.subheader(f"📊 Raw Data for {ticker}")
            st.dataframe(data.tail(), use_container_width=True)

            # Candlestick Chart if OHLC is present
            required_cols = {"Open", "High", "Low", "Close"}
            if required_cols.issubset(data.columns):
                st.subheader(f"📉 {ticker} Candlestick Chart")
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=data.index,
                            open=data["Open"],
                            high=data["High"],
                            low=data["Low"],
                            close=data["Close"],
                            increasing_line_color='green',
                            decreasing_line_color='red',
                            name='OHLC'
                        )
                    ]
                )
                fig.update_layout(
                    title=f"{ticker} Stock Price",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    xaxis_rangeslider_visible=False,
                    template="plotly_white",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("📉 Candlestick chart not available. Showing Close price instead.")
                st.line_chart(data["Close"])

            # Volume Chart
            if "Volume" in data.columns:
                st.subheader("📦 Trading Volume")
                st.line_chart(data["Volume"])

            # Summary Statistics
            st.subheader("📌 Summary Statistics")
            st.dataframe(data.describe(), use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
