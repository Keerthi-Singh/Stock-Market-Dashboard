import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(page_title="📈 Stock Market Trend Forecasting Dashboard", layout="wide")

# Title
st.title("📈 Stock Market Trend Forecasting Dashboard")
st.markdown("*Financial Analytics Pipeline with Predictive Modeling & Trend Analysis*")

# Sidebar Inputs
st.sidebar.header("📊 Stock Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "AAPL").upper()
period = st.sidebar.selectbox("Select Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Select Interval", ["1d", "1wk"], index=0)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Analysis Settings")
short_ma = st.sidebar.slider("Short Moving Average (days)", 5, 50, 20)
long_ma = st.sidebar.slider("Long Moving Average (days)", 20, 200, 50)
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 5, 60, 30)

if ticker:
    try:
        # --- Automated Stock Data Extraction ---
        data = yf.download(ticker, period=period, interval=interval)

        if data.empty:
            st.error(f"❌ No data found for ticker '{ticker}' with selected options.")
            st.stop()

        # Flatten multi-level columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # ============================================================
        # DATA CLEANING & PREPROCESSING
        # ============================================================
        st.header("🧹 Data Cleaning & Preprocessing")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            missing = data.isnull().sum().sum()
            st.metric("Missing Values (before)", int(missing))
        with col3:
            duplicates = data.index.duplicated().sum()
            st.metric("Duplicate Rows", int(duplicates))

        # Data cleaning
        data = data.dropna()
        data = data[~data.index.duplicated(keep='first')]
        data = data.sort_index()

        # Statistical transformations
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Pct_Change'] = data['Close'].pct_change() * 100
        data = data.dropna()

        st.success(f"✅ Data cleaned successfully. Final dataset: {len(data)} records.")

        # ============================================================
        # FEATURE ENGINEERING
        # ============================================================
        st.header("🔧 Feature Engineering")

        # Moving Averages
        data['SMA_Short'] = data['Close'].rolling(window=short_ma).mean()
        data['SMA_Long'] = data['Close'].rolling(window=long_ma).mean()
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Volatility Indicators
        data['Rolling_Std'] = data['Close'].rolling(window=20).std()
        data['Bollinger_Upper'] = data['SMA_Short'] + (data['Rolling_Std'] * 2)
        data['Bollinger_Lower'] = data['SMA_Short'] - (data['Rolling_Std'] * 2)

        # RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # ATR (Average True Range)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data['ATR'] = true_range.rolling(window=14).mean()

        # Momentum & Rate of Change
        data['Momentum'] = data['Close'] - data['Close'].shift(10)
        data['ROC'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100

        # Volume features
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']

        # Lag features for ML
        for lag in [1, 3, 5, 7]:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Return_Lag_{lag}'] = data['Log_Return'].shift(lag)

        # Day of week, month
        data['Day_of_Week'] = data.index.dayofweek
        data['Month'] = data.index.month

        engineered_features = ['SMA_Short', 'SMA_Long', 'EMA_12', 'EMA_26', 'MACD',
                               'Rolling_Std', 'Bollinger_Upper', 'Bollinger_Lower',
                               'RSI', 'ATR', 'Momentum', 'ROC', 'Volume_Ratio']
        st.dataframe(data[engineered_features].tail(10), use_container_width=True)

        # ============================================================
        # EXPLORATORY DATA ANALYSIS
        # ============================================================
        st.header("📊 Exploratory Data Analysis")

        tab1, tab2, tab3 = st.tabs(["Price Overview", "Distribution Analysis", "Correlation"])

        with tab1:
            fig = go.Figure(data=[
                go.Candlestick(
                    x=data.index, open=data["Open"], high=data["High"],
                    low=data["Low"], close=data["Close"],
                    increasing_line_color='green', decreasing_line_color='red', name='OHLC'
                )
            ])
            fig.update_layout(title=f"{ticker} OHLC Price Chart", xaxis_title="Date",
                              yaxis_title="Price (USD)", xaxis_rangeslider_visible=False,
                              template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(data, x='Log_Return', nbins=50,
                                   title="Distribution of Log Returns",
                                   labels={'Log_Return': 'Log Return'})
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.histogram(data, x='Pct_Change', nbins=50,
                                   title="Distribution of Daily % Change",
                                   labels={'Pct_Change': '% Change'})
                fig.update_layout(template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("📌 Summary Statistics")
            st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Volume', 'Log_Return']].describe(),
                         use_container_width=True)

        with tab3:
            corr_cols = ['Close', 'Volume', 'Log_Return', 'RSI', 'MACD', 'ATR', 'Momentum', 'ROC']
            corr_data = data[corr_cols].dropna().corr()
            fig = px.imshow(corr_data, text_auto=".2f", title="Feature Correlation Heatmap",
                            color_continuous_scale="RdBu_r", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # TREND ANALYSIS & TECHNICAL INDICATORS
        # ============================================================
        st.header("📈 Trend Analysis & Technical Indicators")

        tab4, tab5, tab6 = st.tabs(["Moving Averages & Bollinger", "RSI & MACD", "Volatility"])

        with tab4:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close Price',
                                     line=dict(color='black', width=1)))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_Short'],
                                     name=f'SMA {short_ma}', line=dict(color='blue', width=1.5)))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_Long'],
                                     name=f'SMA {long_ma}', line=dict(color='orange', width=1.5)))
            fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Upper'],
                                     name='Bollinger Upper', line=dict(color='gray', dash='dash')))
            fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger_Lower'],
                                     name='Bollinger Lower', line=dict(color='gray', dash='dash'),
                                     fill='tonexty', fillcolor='rgba(128,128,128,0.1)'))
            fig.update_layout(title=f"{ticker} - Moving Averages & Bollinger Bands",
                              template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)

            if len(data['SMA_Short'].dropna()) > 0 and len(data['SMA_Long'].dropna()) > 0:
                last_short = data['SMA_Short'].iloc[-1]
                last_long = data['SMA_Long'].iloc[-1]
                if last_short > last_long:
                    st.success(f"🟢 **Bullish Signal**: Short MA ({short_ma}) is above Long MA ({long_ma})")
                else:
                    st.warning(f"🔴 **Bearish Signal**: Short MA ({short_ma}) is below Long MA ({long_ma})")

        with tab5:
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI',
                                         line=dict(color='purple')))
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                fig.update_layout(title="RSI (14-day)", template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD',
                                         line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal Line',
                                         line=dict(color='red')))
                fig.add_trace(go.Bar(x=data.index, y=data['MACD'] - data['MACD_Signal'],
                                     name='Histogram', marker_color='gray', opacity=0.5))
                fig.update_layout(title="MACD", template="plotly_white", height=400)
                st.plotly_chart(fig, use_container_width=True)

        with tab6:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(data, y='Rolling_Std', title="20-Day Rolling Volatility (Std Dev)",
                              template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.line(data, y='ATR', title="Average True Range (14-day)",
                              template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)

        # ============================================================
        # PREDICTIVE MODELING (Scikit-learn)
        # ============================================================
        st.header("🤖 Predictive Modeling & Trend Forecasting")

        feature_cols = ['SMA_Short', 'SMA_Long', 'EMA_12', 'EMA_26', 'MACD', 'RSI',
                        'ATR', 'Momentum', 'ROC', 'Volume_Ratio', 'Rolling_Std',
                        'Close_Lag_1', 'Close_Lag_3', 'Close_Lag_5', 'Close_Lag_7',
                        'Return_Lag_1', 'Return_Lag_3', 'Day_of_Week', 'Month']

        data['Target'] = data['Close'].shift(-1)
        ml_data = data[feature_cols + ['Target']].dropna()

        if len(ml_data) > 50:
            X = ml_data[feature_cols].values
            y = ml_data['Target'].values

            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)

            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Linear Regression")
                st.metric("MAE", f"${mean_absolute_error(y_test, lr_pred):.2f}")
                st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, lr_pred)):.2f}")
                st.metric("R² Score", f"{r2_score(y_test, lr_pred):.4f}")

            with col2:
                st.subheader("Random Forest")
                st.metric("MAE", f"${mean_absolute_error(y_test, rf_pred):.2f}")
                st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")
                st.metric("R² Score", f"{r2_score(y_test, rf_pred):.4f}")

            test_dates = ml_data.index[split_idx:]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=test_dates, y=y_test, name='Actual',
                                     line=dict(color='black', width=2)))
            fig.add_trace(go.Scatter(x=test_dates, y=lr_pred, name='Linear Regression',
                                     line=dict(color='blue', dash='dash')))
            fig.add_trace(go.Scatter(x=test_dates, y=rf_pred, name='Random Forest',
                                     line=dict(color='green', dash='dash')))
            fig.update_layout(title="Predicted vs Actual Stock Price",
                              template="plotly_white", height=500,
                              xaxis_title="Date", yaxis_title="Price (USD)")
            st.plotly_chart(fig, use_container_width=True)

            # Feature Importance
            st.subheader("🏆 Feature Importance (Random Forest)")
            importance = pd.Series(rf_model.feature_importances_, index=feature_cols)
            importance = importance.sort_values(ascending=True).tail(10)
            fig = px.bar(x=importance.values, y=importance.index, orientation='h',
                         title="Top 10 Most Important Features",
                         labels={'x': 'Importance', 'y': 'Feature'})
            fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig, use_container_width=True)

            # ============================================================
            # SCENARIO PROJECTIONS / FORECAST
            # ============================================================
            st.header("🔮 Forecast & Scenario Projections")

            last_features = scaler.transform(X[-1:])
            last_price = data['Close'].iloc[-1]

            rf_next = rf_model.predict(last_features)[0]
            lr_next = lr_model.predict(last_features)[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${last_price:.2f}")
            with col2:
                change_lr = ((lr_next - last_price) / last_price) * 100
                st.metric("LR Forecast (Next Day)", f"${lr_next:.2f}", f"{change_lr:+.2f}%")
            with col3:
                change_rf = ((rf_next - last_price) / last_price) * 100
                st.metric("RF Forecast (Next Day)", f"${rf_next:.2f}", f"{change_rf:+.2f}%")

            # Monte Carlo Simulation
            st.subheader(f"📅 {forecast_days}-Day Scenario Projection")
            daily_returns = data['Log_Return'].dropna()
            mu = daily_returns.mean()
            sigma = daily_returns.std()

            np.random.seed(42)
            n_simulations = 100
            simulations = np.zeros((forecast_days, n_simulations))

            for i in range(n_simulations):
                random_returns = np.random.normal(mu, sigma, forecast_days)
                price_path = last_price * np.exp(np.cumsum(random_returns))
                simulations[:, i] = price_path

            future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1),
                                         periods=forecast_days, freq='B')

            fig = go.Figure()
            for i in range(min(50, n_simulations)):
                fig.add_trace(go.Scatter(x=future_dates, y=simulations[:, i],
                                         mode='lines', line=dict(width=0.5, color='lightblue'),
                                         showlegend=False, opacity=0.3))

            p5 = np.percentile(simulations, 5, axis=1)
            p50 = np.percentile(simulations, 50, axis=1)
            p95 = np.percentile(simulations, 95, axis=1)

            fig.add_trace(go.Scatter(x=future_dates, y=p50, name='Median',
                                     line=dict(color='blue', width=2)))
            fig.add_trace(go.Scatter(x=future_dates, y=p95, name='95th Percentile',
                                     line=dict(color='green', dash='dash')))
            fig.add_trace(go.Scatter(x=future_dates, y=p5, name='5th Percentile',
                                     line=dict(color='red', dash='dash')))

            fig.update_layout(title=f"Monte Carlo Simulation - {forecast_days} Day Projection ({n_simulations} scenarios)",
                              xaxis_title="Date", yaxis_title="Projected Price (USD)",
                              template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🐻 Bearish (5th %ile)", f"${p5[-1]:.2f}",
                          f"{((p5[-1]-last_price)/last_price)*100:+.1f}%")
            with col2:
                st.metric("📊 Base Case (Median)", f"${p50[-1]:.2f}",
                          f"{((p50[-1]-last_price)/last_price)*100:+.1f}%")
            with col3:
                st.metric("🐂 Bullish (95th %ile)", f"${p95[-1]:.2f}",
                          f"{((p95[-1]-last_price)/last_price)*100:+.1f}%")

        else:
            st.warning("⚠️ Not enough data for predictive modeling. Try a longer period.")

        # ============================================================
        # INVESTMENT INSIGHTS SUMMARY
        # ============================================================
        st.header("💡 Actionable Investment Insights")

        insights = []
        last_rsi = data['RSI'].iloc[-1]
        last_macd = data['MACD'].iloc[-1]
        last_macd_signal = data['MACD_Signal'].iloc[-1]
        last_vol_ratio = data['Volume_Ratio'].iloc[-1]

        if last_rsi > 70:
            insights.append("⚠️ **Overbought**: RSI is above 70 — potential reversal or pullback.")
        elif last_rsi < 30:
            insights.append("🟢 **Oversold**: RSI is below 30 — potential buying opportunity.")
        else:
            insights.append(f"📊 **Neutral RSI**: RSI at {last_rsi:.1f} — no extreme conditions.")

        if last_macd > last_macd_signal:
            insights.append("🟢 **MACD Bullish Crossover**: MACD is above the signal line.")
        else:
            insights.append("🔴 **MACD Bearish Crossover**: MACD is below the signal line.")

        if last_vol_ratio > 1.5:
            insights.append("📦 **High Volume**: Trading volume is significantly above average.")
        elif last_vol_ratio < 0.5:
            insights.append("📉 **Low Volume**: Trading volume is below average.")

        if len(data['SMA_Short'].dropna()) > 0 and len(data['SMA_Long'].dropna()) > 0:
            if data['SMA_Short'].iloc[-1] > data['SMA_Long'].iloc[-1]:
                insights.append(f"📈 **Uptrend**: Short-term MA ({short_ma}) is above Long-term MA ({long_ma}).")
            else:
                insights.append(f"📉 **Downtrend**: Short-term MA ({short_ma}) is below Long-term MA ({long_ma}).")

        for insight in insights:
            st.markdown(insight)

        # Volume Analysis
        st.header("📦 Volume Analysis")
        fig = go.Figure()
        colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red'
                  for i in range(len(data))]
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], marker_color=colors, name='Volume'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Volume_SMA'], name='Volume SMA (20)',
                                 line=dict(color='blue', width=2)))
        fig.update_layout(title="Trading Volume with Moving Average",
                          template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Footer
        st.markdown("---")
        st.markdown("**Tech Stack:** Python | Pandas | NumPy | Scikit-learn | Yahoo Finance API (yfinance) | Streamlit | Plotly")
        st.markdown("⚠️ *Disclaimer: This tool is for educational purposes only. Not financial advice.*")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
        st.exception(e)
