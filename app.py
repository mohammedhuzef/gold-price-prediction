from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import yfinance as yf
import datetime
import plotly.graph_objects as go

# Page settings
st.set_page_config(
    page_title="Gold Price Prediction (INR)",
    page_icon="🥇",
    layout="wide"
)

st.title("🥇 Gold Price Prediction & Next-Day Forecasting (₹ INR)")
st.write("Real-time 24-Karat gold rate monitoring and ML-powered next-day price forecasting in Indian Rupees (₹).")

# Sidebar for currency & conversion settings
st.sidebar.header("⚙️ Settings & Market Calibration")

indian_reference_24k_10g = st.sidebar.number_input(
    "Today's Live Indian 24K Rate (₹ per 10g)",
    min_value=1.0,
    value=155680.0,
    step=10.0,
    help="Current 24K gold rate from your mobile app / MCX spot rate. The chart and ML forecasting are calibrated directly to this live price."
)

usd_to_inr = st.sidebar.number_input(
    "1 USD to INR Exchange Rate (₹)",
    min_value=1.0,
    max_value=200.0,
    value=85.0,
    step=0.5,
    help="Current USD to INR exchange rate."
)

india_premium = st.sidebar.number_input(
    "India Market Taxes & Premium (%)",
    min_value=0.0,
    max_value=50.0,
    value=28.0,
    step=1.0,
    help="Adjustment for India's Import Duty (~15%), GST (3%), and local dealer premiums."
)

unit_option = st.sidebar.radio(
    "Price Display Unit",
    options=["₹ per 10 Grams", "₹ per Troy Ounce"],
    index=0
)

# Model loading
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "gold_price_model.pkl"

if not MODEL_PATH.exists():
    st.error("Model file not found in models folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.divider()

# Timeframe selector
col_tf1, col_tf2 = st.columns([1, 3])
with col_tf1:
    timeframe_option = st.selectbox(
        "Chart History Timeframe",
        options=["1 Day", "4 Hours", "1 Hour", "15 Minutes"],
        index=0
    )

interval_map = {
    "15 Minutes": "15m",
    "1 Hour": "1h",
    "4 Hours": "4h",
    "1 Day": "1d"
}
selected_interval = interval_map[timeframe_option]

period_map = {
    "15m": "1mo",
    "1h": "1mo",
    "4h": "1mo",
    "1d": "2mo"
}
selected_period = period_map[selected_interval]

with st.spinner("Fetching gold market data and calculating tomorrow's forecast..."):
    try:
        data = yf.download('GC=F', period=selected_period, interval=selected_interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            df_usd = data.xs('GC=F', level=1, axis=1).dropna(subset=['Close'])
        else:
            df_usd = data.dropna(subset=['Close'])

        if df_usd.empty:
            st.warning("⚠️ No recent gold market data available from Yahoo Finance for this timeframe. Please choose '1 Day'.")
            st.stop()
            
        df_live = pd.DataFrame({
            'Datetime': pd.to_datetime(df_usd.index).tz_localize(None),
            'OpenUSD': df_usd['Open'].values,
            'HighUSD': df_usd['High'].values,
            'LowUSD': df_usd['Low'].values,
            'PriceUSD': df_usd['Close'].values
        }).sort_values("Datetime").reset_index(drop=True)
        
        # Unit conversion factors
        TROY_OUNCE_GRAMS = 31.1034768
        premium_multiplier = 1.0 + (india_premium / 100.0)
        
        if unit_option == "₹ per 10 Grams":
            conv_factor = (usd_to_inr / TROY_OUNCE_GRAMS) * 10.0 * premium_multiplier
            reference_price = indian_reference_24k_10g
        else:
            conv_factor = usd_to_inr * premium_multiplier
            reference_price = (indian_reference_24k_10g / 10.0) * TROY_OUNCE_GRAMS
            
        df_live['Open'] = df_live['OpenUSD'] * conv_factor
        df_live['High'] = df_live['HighUSD'] * conv_factor
        df_live['Low'] = df_live['LowUSD'] * conv_factor
        df_live['Price'] = df_live['PriceUSD'] * conv_factor

        # Calibrate historical prices to today's Indian market rate
        calibration_factor = reference_price / df_live['Price'].iloc[-1]
        df_live[['Open', 'High', 'Low', 'Price']] *= calibration_factor
        
        # Ensure Today (27 Sep 2026) is explicitly on the chart with today's live rate
        today_date = datetime.date(2026, 9, 27)
        tomorrow_date = today_date + datetime.timedelta(days=1)  # 28 Sep 2026
        
        last_hist_date = df_live['Datetime'].iloc[-1].date()
        if last_hist_date < today_date:
            today_row = pd.DataFrame({
                'Datetime': [pd.Timestamp(today_date)],
                'Open': [reference_price],
                'High': [reference_price],
                'Low': [reference_price],
                'Price': [reference_price]
            })
            df_live = pd.concat([df_live, today_row], ignore_index=True)
            
        # Calculate feature ratios for historical predictions
        df_live['price_1_day_ago'] = df_live['Price'].shift(1)
        df_live['price_2_days_ago'] = df_live['Price'].shift(2)
        df_live['price_3_days_ago'] = df_live['Price'].shift(3)
        df_live['price_7_days_ago'] = df_live['Price'].shift(7)
        
        df_live['ratio_1_2'] = df_live['price_1_day_ago'] / df_live['price_2_days_ago']
        df_live['ratio_1_3'] = df_live['price_1_day_ago'] / df_live['price_3_days_ago']
        df_live['ratio_1_7'] = df_live['price_1_day_ago'] / df_live['price_7_days_ago']
        
        features_live = ["ratio_1_2", "ratio_1_3", "ratio_1_7"]
        df_live['Predicted Price'] = None
        
        valid_idx = df_live.dropna(subset=features_live).index
        if not valid_idx.empty:
            input_data = df_live.loc[valid_idx, features_live]
            pred_ratios = model.predict(input_data)
            df_live.loc[valid_idx, 'Predicted Price'] = df_live.loc[valid_idx, "price_1_day_ago"] * pred_ratios

        # =====================================================================
        # PREDICT TOMORROW'S PRICE (28 Sep 2026)
        # =====================================================================
        p_today = df_live['Price'].iloc[-1]
        p_1 = df_live['Price'].iloc[-2]
        p_2 = df_live['Price'].iloc[-3]
        p_6 = df_live['Price'].iloc[-7] if len(df_live) >= 7 else df_live['Price'].iloc[0]

        r12 = p_today / p_1
        r13 = p_today / p_2
        r17 = p_today / p_6

        tomorrow_features = pd.DataFrame([{"ratio_1_2": r12, "ratio_1_3": r13, "ratio_1_7": r17}])
        pred_ratio_tomorrow = model.predict(tomorrow_features)[0]
        predicted_price_tomorrow = p_today * pred_ratio_tomorrow
        predicted_change = predicted_price_tomorrow - p_today
        predicted_pct_change = (predicted_change / p_today) * 100.0

        # =====================================================================
        # FORECAST HIGHLIGHT METRICS CARDS
        # =====================================================================
        st.subheader("🎯 Tomorrow's Gold Price Forecast")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        f_col1.metric(
            label="📅 Today's Live Rate (27 Sep)",
            value=f"₹{p_today:,.2f}"
        )
        
        f_col2.metric(
            label="🔮 Tomorrow's Forecast (28 Sep)",
            value=f"₹{predicted_price_tomorrow:,.2f}",
            delta=f"{predicted_change:+,.2f} ({predicted_pct_change:+.2f}%)"
        )
        
        f_col3.metric(
            label="📊 Expected Direction",
            value="📈 Uptrend (Gain)" if predicted_change >= 0 else "📉 Downtrend (Drop)"
        )

        f_col4.metric(
            label="⚖️ Unit",
            value=unit_option
        )
        
        # =====================================================================
        # INTERACTIVE CHART WITH TOMORROW'S PREDICTION PROJECTION
        # =====================================================================
        today_timestamp = pd.Timestamp(today_date)
        tomorrow_timestamp = pd.Timestamp(tomorrow_date)
        
        fig = go.Figure()
        
        # 1. Historical Actual Live Price (Teal Solid Line)
        fig.add_trace(go.Scatter(
            x=df_live['Datetime'],
            y=df_live['Price'],
            mode='lines+markers',
            name='Actual Market Price',
            line=dict(color='#26a69a', width=2.5),
            marker=dict(size=4),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual Price: ₹%{y:,.2f}<extra></extra>"
        ))
        
        # 2. Historical Model Predictions (Orange Solid Line)
        valid_preds = df_live.dropna(subset=['Predicted Price'])
        if not valid_preds.empty:
            fig.add_trace(go.Scatter(
                x=valid_preds['Datetime'],
                y=valid_preds['Predicted Price'],
                mode='lines',
                name='Model In-Sample Prediction',
                line=dict(color='#f39c12', width=1.5, dash='dash'),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Model Tracking: ₹%{y:,.2f}<extra></extra>"
            ))
            
        # 3. Tomorrow's Future Forecast Line (Glowing Gold Dotted Line from Today to Tomorrow)
        fig.add_trace(go.Scatter(
            x=[today_timestamp, tomorrow_timestamp],
            y=[p_today, predicted_price_tomorrow],
            mode='lines+markers',
            name="Tomorrow's Forecast (28 Sep)",
            line=dict(color='#e74c3c' if predicted_change < 0 else '#f1c40f', width=3, dash='dot'),
            marker=dict(size=[6, 12], symbol=['circle', 'star'], color='#f1c40f'),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: ₹%{y:,.2f}<extra></extra>"
        ))
        
        # 4. Highlight Callout for Tomorrow's Predicted Price
        fig.add_annotation(
            x=tomorrow_timestamp,
            y=predicted_price_tomorrow,
            text=f"<b>Tomorrow (28 Sep Forecast)</b><br>₹{predicted_price_tomorrow:,.2f} ({predicted_pct_change:+.2f}%)",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowcolor='#f1c40f',
            ax=0,
            ay=-45,
            bgcolor='rgba(30, 34, 45, 0.95)',
            bordercolor='#f1c40f',
            borderwidth=1.5,
            font=dict(color='#f1c40f', size=13)
        )
        
        # 5. Highlight Marker on Today (27 Sep)
        fig.add_annotation(
            x=today_timestamp,
            y=p_today,
            text=f"<b>Today (27 Sep)</b><br>₹{p_today:,.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.0,
            arrowcolor='#26a69a',
            ax=0,
            ay=45,
            bgcolor='rgba(30, 34, 45, 0.95)',
            bordercolor='#26a69a',
            borderwidth=1.5,
            font=dict(color='#26a69a', size=12)
        )
        
        fig.update_layout(
            title=dict(
                text=f"Gold Price Trend & Tomorrow's Future Prediction ({unit_option})",
                font=dict(size=18, color='#ffffff')
            ),
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor='rgba(255,255,255,0.08)'
            ),
            yaxis=dict(
                title=f"Price ({unit_option})",
                tickformat=",.0f",
                tickprefix="₹",
                showgrid=True,
                gridcolor='rgba(255,255,255,0.08)'
            ),
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            hovermode="x unified",
            height=520,
            margin=dict(l=20, r=20, t=60, b=30),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, width="stretch")
        
    except Exception as e:
        st.error(f"Could not load real-time data: {e}")

st.divider()

# =============================================================================
# MANUAL CUSTOM PREDICTION FORM
# =============================================================================
st.subheader("📝 Manual Price Calculator & Simulator")
st.write("You can test custom scenarios by modifying past prices below:")

# Default values based on current live Indian price
if unit_option == "₹ per 10 Grams":
    default_1d = indian_reference_24k_10g
    default_2d = indian_reference_24k_10g * 0.998
    default_3d = indian_reference_24k_10g * 0.996
    default_7d = indian_reference_24k_10g * 0.990
else:
    default_1d = (indian_reference_24k_10g / 10.0) * TROY_OUNCE_GRAMS
    default_2d = default_1d * 0.998
    default_3d = default_1d * 0.996
    default_7d = default_1d * 0.990

col1, col2 = st.columns(2)

with col1:
    val_1 = st.text_input(f"Today's Price — 1 day ago ({unit_option})", value=f"{default_1d:.2f}")
    price_1_day_ago = float(val_1) if val_1 else 0.0
    
    val_2 = st.text_input(f"2 days ago price ({unit_option})", value=f"{default_2d:.2f}")
    price_2_days_ago = float(val_2) if val_2 else 0.0

with col2:
    val_3 = st.text_input(f"3 days ago price ({unit_option})", value=f"{default_3d:.2f}")
    price_3_days_ago = float(val_3) if val_3 else 0.0
    
    val_7 = st.text_input(f"7 days ago price ({unit_option})", value=f"{default_7d:.2f}")
    price_7_days_ago = float(val_7) if val_7 else 0.0

if st.button("Predict Custom Scenario Price", type="primary"):
    if price_2_days_ago == 0 or price_3_days_ago == 0 or price_7_days_ago == 0:
        st.error("Previous prices must be greater than zero.")
        st.stop()

    r12 = price_1_day_ago / price_2_days_ago
    r13 = price_1_day_ago / price_3_days_ago
    r17 = price_1_day_ago / price_7_days_ago

    input_data = pd.DataFrame([{
        "ratio_1_2": r12,
        "ratio_1_3": r13,
        "ratio_1_7": r17
    }])

    pred_ratio = model.predict(input_data)[0]
    predicted_input_unit = price_1_day_ago * pred_ratio

    if unit_option == "₹ per 10 Grams":
        pred_inr_10g = predicted_input_unit
        pred_inr_1g = pred_inr_10g / 10.0
        pred_inr_oz = pred_inr_1g * TROY_OUNCE_GRAMS
    else:
        pred_inr_oz = predicted_input_unit
        pred_inr_1g = pred_inr_oz / TROY_OUNCE_GRAMS
        pred_inr_10g = pred_inr_1g * 10.0

    pred_usd_oz = pred_inr_oz / usd_to_inr

    st.success("### 🎯 Model Prediction Result:")

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Per 10 Grams (24K)", f"₹{pred_inr_10g:,.2f}")
    res_col2.metric("Per Troy Ounce", f"₹{pred_inr_oz:,.2f}")
    res_col3.metric("Per 1 Gram", f"₹{pred_inr_1g:,.2f}")

    st.info(f"💵 Equivalent USD Price: **${pred_usd_oz:,.2f} USD / oz** (Exchange rate: 1 USD = ₹{usd_to_inr:.2f})")
    st.caption("Note: This is an ML model prediction based on momentum ratios, not financial advice.")
