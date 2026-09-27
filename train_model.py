"""Train and evaluate the next-day gold-price prediction model.

Run from the project directory:
    python train_model.py
"""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from model_utils import RatioLinearRegressor

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "gold_price_model.pkl"  # app.py loads this exact file

csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("data folder mein CSV file nahi mili.")

df = pd.read_csv(csv_files[0])
if not {"Date", "Price"}.issubset(df.columns):
    raise ValueError("CSV mein 'Date' aur 'Price' columns hone chahiye.")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Price"] = pd.to_numeric(
    df["Price"].astype(str).str.replace(",", "", regex=False), errors="coerce"
)
df = df.dropna(subset=["Date", "Price"]).sort_values("Date").reset_index(drop=True)

# Ratio features keep the model compatible with USD/oz and INR input units.
df["price_1_day_ago"] = df["Price"].shift(1)
df["price_2_days_ago"] = df["Price"].shift(2)
df["price_3_days_ago"] = df["Price"].shift(3)
df["price_7_days_ago"] = df["Price"].shift(7)
df["ratio_1_2"] = df["price_1_day_ago"] / df["price_2_days_ago"]
df["ratio_1_3"] = df["price_1_day_ago"] / df["price_3_days_ago"]
df["ratio_1_7"] = df["price_1_day_ago"] / df["price_7_days_ago"]
df["target_ratio"] = df["Price"] / df["price_1_day_ago"]

FEATURES = ["ratio_1_2", "ratio_1_3", "ratio_1_7"]
model_df = df.dropna(subset=FEATURES + ["target_ratio"]).copy()
X = model_df[FEATURES]
y = model_df["target_ratio"]

# Never shuffle financial time-series data: latest 20% stays untouched for testing.
split_index = int(len(model_df) * 0.80)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# A least-squares model is intentionally used instead of XGBoost so the app
# runs without a platform-specific DLL. The ratio features keep predictions
# independent of whether the user enters INR or USD prices.
model = RatioLinearRegressor().fit(X_train, y_train)

# Convert ratio predictions back to price before calculating user-facing metrics.
predicted_ratios = model.predict(X_test)
previous_prices = model_df["price_1_day_ago"].iloc[split_index:]
predicted_prices = previous_prices.to_numpy() * predicted_ratios
actual_prices = model_df["Price"].iloc[split_index:].to_numpy()

mae = np.mean(np.abs(actual_prices - predicted_prices))
rmse = np.sqrt(np.mean((actual_prices - predicted_prices) ** 2))
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
r2 = 1 - (np.sum((actual_prices - predicted_prices) ** 2) / np.sum((actual_prices - actual_prices.mean()) ** 2))

print("\nModel: ratio-based least-squares regression")
print("\nUntouched latest-20% test results:")
print(f"MAE:  ${mae:,.2f} per troy ounce")
print(f"RMSE: ${rmse:,.2f} per troy ounce")
print(f"MAPE: {mape:.2f}%  (lower is better)")
print(f"R²:   {r2:.3f}  (closer to 1 is better; can be negative)")

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved: {MODEL_PATH}")
