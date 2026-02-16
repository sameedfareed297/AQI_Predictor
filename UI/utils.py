import sys
from pathlib import Path
from datetime import timedelta
import pandas as pd

# Fix: Point Python to Src/features/ so it can find feature_engineering.py
# This file (utils.py) lives at:          AQI_PREDICTOR/utils.py
# feature_engineering.py lives at:        AQI_PREDICTOR/src/features/feature_engineering.py

_ROOT = Path(__file__).resolve().parent                      # AQI_PREDICTOR/
sys.path.insert(0, str(_ROOT / "Src" / "features"))          # adds src/features/ to path

from Src.features.feature_engineering import FEATURES, TARGET, LAGS      # now this works 


def generate_forecast(historical_df, model, days=3):

    if historical_df is None or historical_df.empty:
        return None

    hours = days * 24
    max_lag = max(LAGS)  # 48 — we need at least this many past rows

    # Keep enough history to compute all lags at every step
    history_aqi = list(historical_df["aqi"].tail(max_lag + 1))
    last_ts = pd.to_datetime(historical_df.iloc[-1]["timestamp"])

    forecasts = []

    for step in range(hours):
        # 1. Next timestamp 
        next_ts = last_ts + timedelta(hours=1)

        # 2. Build feature row
        row = {}

        for lag in LAGS:
            row[f"{TARGET}_lag_{lag}"] = history_aqi[-lag]

        # Time features
        row["hour"] = next_ts.hour
        row["day"] = next_ts.day
        row["month"] = next_ts.month
        row["weekday"] = next_ts.weekday()

        # 3. Predict 
        X = pd.DataFrame([row])[FEATURES] 
        aqi_pred = float(model.predict(X)[0])

        # 4. Record the prediction
        forecasts.append({
            "timestamp": next_ts,
            "aqi_predicted": aqi_pred
        })

        history_aqi.append(aqi_pred)

        # Move timestamp forward
        last_ts = next_ts

    return pd.DataFrame(forecasts)