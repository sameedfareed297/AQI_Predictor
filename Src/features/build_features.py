import pandas as pd


def compute_aqi_pm25(pm25):
    # EPA standard AQI breakpoints for PM2.5 (24-hr, but works for hourly too)
    breakpoints = [
        (0,    12.0,   0,   50),
        (12.1, 35.4,  51,  100),
        (35.5, 55.4, 101,  150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for lo, hi, aqi_lo, aqi_hi in breakpoints:
        if lo <= pm25 <= hi:
            return aqi_lo + (pm25 - lo) * (aqi_hi - aqi_lo) / (hi - lo)
    # If somehow above 500.4, scale linearly beyond 500
    return 500 + (pm25 - 500.4)

def build_features(df):
    """
    Comprehensive feature engineering for AQI prediction (Feature Group v6).
    
    Features:
    - Time features: hour, day, month, weekday, dayofyear, is_weekend
    - Lag features: [1, 3, 6, 24] for all pollutants + AQI
    - Rolling features: [6, 12, 24] for all pollutants + AQI
    - Difference features: [1, 24] for all pollutants + AQI
    - Target: AQI 24 hours ahead
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    #EVENT_ID: Primary key (as STRING for Hopsworks)
    df["event_id"] = (df["timestamp"].astype("int64") // 10**9).astype(str)

    # TARGET: Compute AQI from PM2.5
    df["aqi"] = df["pm2_5"].apply(compute_aqi_pm25)

    # TIME FEATURES
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["weekday"] = df["timestamp"].dt.weekday
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = (df["weekday"].isin([5, 6])).astype("int32")

    # POLLUTANTS 
    pollutants = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"]
    
    # LAG FEATURES [1, 3, 6, 24]
    lags = [1, 3, 6, 24]
    
    # Pollutant lags
    for pollutant in pollutants:
        for lag in lags:
            df[f"{pollutant}_lag{lag}"] = df[pollutant].shift(lag)
    
    # AQI lags
    for lag in lags:
        df[f"aqi_lag{lag}"] = df["aqi"].shift(lag)

    # ROLLING AVERAGE FEATURES [6, 12, 24]
    rolling_windows = [6, 12, 24]
    
    # Pollutant rolling averages
    for pollutant in pollutants:
        for window in rolling_windows:
            df[f"{pollutant}_roll{window}"] = df[pollutant].rolling(window).mean()
    
    # AQI rolling averages
    for window in rolling_windows:
        df[f"aqi_roll{window}"] = df["aqi"].rolling(window).mean()

    # DIFFERENCE FEATURES [1, 24]
    diffs = [1, 24]
    
    # Pollutant differences (momentum)
    for pollutant in pollutants:
        for diff in diffs:
            df[f"{pollutant}_diff{diff}"] = df[pollutant].diff(diff)
    
    # AQI differences
    for diff in diffs:
        df[f"aqi_diff{diff}"] = df["aqi"].diff(diff)

    # TARGET VARIABLE
    # Predict AQI 24 hours ahead (for supervised learning)
    df["target_aqi_t24"] = df["aqi"].shift(-24)

    # DROP NAs
    # Need ~49 rows before first valid row is complete
    df.dropna(inplace=True)

    return df

