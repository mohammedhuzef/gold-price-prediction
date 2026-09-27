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
    layout="centered"
)

st.title("🥇 Gold Price Prediction (Indian Rupees ₹)")
st.write("Enter previous gold prices to predict the next-day price in Indian Rupees (₹).")

# Sidebar for currency & conversion settings
st.sidebar.header("⚙️ Settings")
usd_to_inr = st.sidebar.number_input(
    "1 USD to INR Exchange Rate (₹)",
    min_value=1.0,
    max_value=200.0,
    value=85.0,
    step=0.5,
    help="Current USD to INR exchange rate for conversion."
)

india_premium = st.sidebar.number_input(
    "India Market Taxes & Premium (%)",
    min_value=0.0,
    max_value=50.0,
    value=28.0,
    step=1.0,
    help="Global gold prices don't include India's Import Duty (~15%), GST (3%), and local dealer premiums. This adjusts the global price to match local retail prices."
)

indian_reference_24k_10g = st.sidebar.number_input(
    "Today's Indian 24K rate (₹ per 10g)",
    min_value=1.0,
    value=157050.0,
    step=100.0,
    help="Update this from your trusted local/mobile gold-rate source. The app calibrates international futures prices to this Indian retail rate."
)

unit_option = st.sidebar.radio(
    "Price Input Unit",
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
st.subheader("📈 Real-Time 24-Karat Gold Rate & Prediction")

timeframe_option = st.selectbox(
    "Chart Timeframe",
    options=["15 Minutes", "1 Hour", "4 Hours", "1 Day"],
    index=3
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

with st.spinner("Fetching live gold rates and generating predictions..."):
    try:
        data = yf.download('GC=F', period=selected_period, interval=selected_interval, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            df_usd = data.xs('GC=F', level=1, axis=1).dropna(subset=['Close'])
        else:
            df_usd = data.dropna(subset=['Close'])
            
        df_live = pd.DataFrame({
            'Datetime': df_usd.index,
            'OpenUSD': df_usd['Open'].values,
            'HighUSD': df_usd['High'].values,
            'LowUSD': df_usd['Low'].values,
            'PriceUSD': df_usd['Close'].values
        })
        # Keep exact datetime, just ensure it's sorted
        df_live = df_live.sort_values("Datetime").reset_index(drop=True)
        
        # Convert to selected unit and apply India premium
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

        # GC=F is an international futures contract. Calibrate it to the Indian
        # 24K retail reference rate so the displayed price is not far from local
        # market/mobile-app rates. Ratios remain unchanged, so the ML features work.
        calibration_factor = reference_price / df_live['Price'].iloc[-1]
        df_live[['Open', 'High', 'Low', 'Price']] *= calibration_factor
            
        df_live['price_1_day_ago'] = df_live['Price'].shift(1)
        df_live['price_2_days_ago'] = df_live['Price'].shift(2)
        df_live['price_3_days_ago'] = df_live['Price'].shift(3)
        df_live['price_7_days_ago'] = df_live['Price'].shift(7)
        
        df_live['ratio_1_2'] = df_live['price_1_day_ago'] / df_live['price_2_days_ago']
        df_live['ratio_1_3'] = df_live['price_1_day_ago'] / df_live['price_3_days_ago']
        df_live['ratio_1_7'] = df_live['price_1_day_ago'] / df_live['price_7_days_ago']
        
        features_live = ["ratio_1_2", "ratio_1_3", "ratio_1_7"]
        
        # Initialize Predicted Price column with NaN
        df_live['Predicted Price'] = None
        
        # Get valid rows where all features are present
        valid_idx = df_live.dropna(subset=features_live).index
        if not valid_idx.empty:
            input_data = df_live.loc[valid_idx, features_live]
            # Predict all valid rows at once (much faster than a loop)
            pred_ratios = model.predict(input_data)
            df_live.loc[valid_idx, 'Predicted Price'] = df_live.loc[valid_idx, "price_1_day_ago"] * pred_ratios
        
        # Dedicated one-day-ahead forecast. The model was trained on daily data,
        # so this must use daily closes even when the chart is set to 15m/1h/4h.
        daily_data = yf.download('GC=F', period='15d', interval='1d', progress=False)
        if isinstance(daily_data.columns, pd.MultiIndex):
            daily_usd = daily_data.xs('GC=F', level=1, axis=1).dropna(subset=['Close'])
        else:
            daily_usd = daily_data.dropna(subset=['Close'])

        daily_prices = (daily_usd['Close'].astype(float) * conv_factor).reset_index(drop=True)
        if len(daily_prices) < 7:
            raise ValueError("Not enough daily gold-price data to predict the next day.")

        day_1, day_2, day_3, day_7 = (
            daily_prices.iloc[-1], daily_prices.iloc[-2],
            daily_prices.iloc[-3], daily_prices.iloc[-7]
        )
        tomorrow_features = pd.DataFrame([{
            "ratio_1_2": day_1 / day_2,
            "ratio_1_3": day_1 / day_3,
            "ratio_1_7": day_1 / day_7,
        }])
        # Use the user-provided Indian rate as the price level. The model predicts
        # a ratio, therefore it still uses daily market movement from GC=F.
        next_day_price = reference_price * model.predict(tomorrow_features)[0]
        next_trading_day = pd.Timestamp(daily_usd.index[-1]) + pd.offsets.BDay(1)
        predicted_change = next_day_price - reference_price

        st.subheader("🎯 Next-Day Gold Price Forecast")
        forecast_col, current_col, change_col = st.columns(3)
        forecast_col.metric(
            f"Predicted price — {next_trading_day:%d %b %Y}",
            f"₹{next_day_price:,.2f}",
            help=f"Prediction in {unit_option}; based on the latest daily closing price."
        )
        current_col.metric("Today's Indian 24K rate", f"₹{reference_price:,.2f}")
        change_col.metric("Expected change", f"₹{predicted_change:,.2f}")
        st.caption(f"Forecast unit: {unit_option}. This is an estimate, not financial advice.")

        # Predict the next selected chart interval for the orange chart line.
        last_row = df_live.iloc[-1]
        tomorrow_1d = last_row["Price"]
        tomorrow_2d = last_row["price_1_day_ago"]
        tomorrow_3d = last_row["price_2_days_ago"]
        tomorrow_7d = df_live["Price"].shift(6).iloc[-1]
        
        r12 = tomorrow_1d / tomorrow_2d
        r13 = tomorrow_1d / tomorrow_3d
        r17 = tomorrow_1d / tomorrow_7d
        pred_ratio_tmrw = model.predict(pd.DataFrame([{"ratio_1_2": r12, "ratio_1_3": r13, "ratio_1_7": r17}]))[0]
        predicted_price_next_interval = tomorrow_1d * pred_ratio_tmrw
        
        # Append next time step
        time_delta_map = {
            "15m": datetime.timedelta(minutes=15),
            "1h": datetime.timedelta(hours=1),
            "4h": datetime.timedelta(hours=4),
            "1d": datetime.timedelta(days=1)
        }
        next_date = df_live['Datetime'].iloc[-1] + time_delta_map[selected_interval]
        
        new_row = pd.DataFrame({
            'Datetime': [next_date],
            'Open': [None],
            'High': [None],
            'Low': [None],
            'Price': [None],
            'Predicted Price': [predicted_price_next_interval]
        })
        df_live = pd.concat([df_live, new_row], ignore_index=True)
        
        # Filter dataframe so chart doesn't start with NaNs in predictions
        first_valid_pred = df_live['Predicted Price'].first_valid_index()
        if first_valid_pred is not None:
            df_live = df_live.iloc[first_valid_pred:].reset_index(drop=True)
            
        fig = go.Figure()
        
        # Line trace for actual prices
        fig.add_trace(go.Scatter(
            x=df_live['Datetime'],
            y=df_live['Price'],
            mode='lines',
            name='Actual Price',
            line=dict(color='#26a69a', width=2)  # Teal/Green color
        ))
        
        # Line trace for predicted prices
        fig.add_trace(go.Scatter(
            x=df_live['Datetime'],
            y=df_live['Predicted Price'],
            mode='lines',
            name='Predicted Price',
            line=dict(color='#f39c12', width=2)  # Binance Yellow/Orange
        ))
        
        fig.update_layout(
            title=f"Live Gold Rate ({timeframe_option})",
            xaxis_title="Time",
            yaxis_title=f"Price ({unit_option})",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not load real-time data: {e}")

st.divider()
st.subheader("Manual Prediction Form")

st.subheader("Previous Gold Prices Enter Karo")

# Default values based on realistic current gold prices in India
if unit_option == "₹ per 10 Grams":
    default_1d = indian_reference_24k_10g
    default_2d = indian_reference_24k_10g * 0.998
    default_3d = indian_reference_24k_10g * 0.996
    default_7d = indian_reference_24k_10g * 0.990
else:  # per Troy Ounce
    default_1d = (indian_reference_24k_10g / 10.0) * TROY_OUNCE_GRAMS
    default_2d = default_1d * 0.998
    default_3d = default_1d * 0.996
    default_7d = default_1d * 0.990

col1, col2 = st.columns(2)

with col1:
    val_1 = st.text_input(f"1 day ago price ({unit_option})", value=str(default_1d))
    price_1_day_ago = float(val_1) if val_1 else 0.0
    
    val_2 = st.text_input(f"2 days ago price ({unit_option})", value=str(default_2d))
    price_2_days_ago = float(val_2) if val_2 else 0.0

with col2:
    val_3 = st.text_input(f"3 days ago price ({unit_option})", value=str(default_3d))
    price_3_days_ago = float(val_3) if val_3 else 0.0
    
    val_7 = st.text_input(f"7 days ago price ({unit_option})", value=str(default_7d))
    price_7_days_ago = float(val_7) if val_7 else 0.0

if st.button("Predict Next-Day Gold Price", type="primary"):
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

    # Calculate prices in all standard units
    TROY_OUNCE_GRAMS = 31.1034768

    if unit_option == "₹ per 10 Grams":
        pred_inr_10g = predicted_input_unit
        pred_inr_1g = pred_inr_10g / 10.0
        pred_inr_oz = pred_inr_1g * TROY_OUNCE_GRAMS
    else:  # per Troy Ounce
        pred_inr_oz = predicted_input_unit
        pred_inr_1g = pred_inr_oz / TROY_OUNCE_GRAMS
        pred_inr_10g = pred_inr_1g * 10.0

    pred_usd_oz = pred_inr_oz / usd_to_inr

    st.success("### 🎯 Predicted Next-Day Gold Price:")

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Per 10 Grams (24K)", f"₹{pred_inr_10g:,.2f}")
    res_col2.metric("Per Troy Ounce", f"₹{pred_inr_oz:,.2f}")
    res_col3.metric("Per 1 Gram", f"₹{pred_inr_1g:,.2f}")

    st.info(f"💵 Equivalent USD Price: **${pred_usd_oz:,.2f} USD / oz** (Exchange rate: 1 USD = ₹{usd_to_inr:.2f})")
    st.caption("Note: This is an ML model prediction based on price momentum & trend, not financial advice.")
