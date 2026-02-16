import os

from datetime import datetime, timedelta
import hopsworks
import pandas as pd
import time
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

from Src.data_ingestion.fetch_openmeteo import fetch_openmeteo_data
from Src.features.build_features import build_features
from Src.feature_store.push_to_hopsworks import push_features

BOOTSTRAP = False  # <- Incremental mode: fetch new data daily


def safe_read(fg, retries=3, wait=10):
    """ Read from feature group with retry logic """
    for i in range(retries):
        try:
            return fg.read()
        except Exception as e:
            print(f"⚠️ Read failed ({i+1}/{retries}): {e}")
            if i < retries - 1:
                time.sleep(wait)
            else:
                raise RuntimeError(f"Feature store read failed after {retries} attempts: {e}")

def main():
    print(" => Starting Open-Meteo AQI pipeline...")

    # Login to Hopsworks
    try:
        project = hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project=os.getenv("HOPSWORKS_PROJECT_NAME"),
        )
        fs = project.get_feature_store()
        print("✅ Successfully logged into Hopsworks")
    except Exception as e:
        logger.error(f"❌ Failed to login to Hopsworks: {e}")
        raise

    # Get or create feature group (v6: comprehensive feature engineering)
    fg = fs.get_or_create_feature_group(
        name="karachi_air_quality",
        version=6,
        primary_key=["event_id"],
        event_time="timestamp",
        description="Karachi AQI hourly features - v6 with comprehensive lag/rolling/diff features",
        online_enabled=False
    )

    # BootStrap Mode
    if BOOTSTRAP:
        start = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")
        end = datetime.utcnow().strftime("%Y-%m-%d")

        print(f"🆕 Bootstrapping {start} → {end}")
        df_raw = fetch_openmeteo_data(start, end)

    # Incremental Mode
    else:
        df_hist = safe_read(fg)
    
        if df_hist.empty:
            print("🔴 Feature store empty — run BOOTSTRAP mode first")
            print("   Set BOOTSTRAP = True in main.py")
            return
    
        last_ts = df_hist["timestamp"].max()
        print(f"⏱️ Last timestamp in FS: {last_ts}")
    
        start = last_ts.strftime("%Y-%m-%d")
        end = datetime.utcnow().strftime("%Y-%m-%d")
        
        df_raw = fetch_openmeteo_data(
            start_date=start,
            end_date=end
        )

        if df_raw.empty:
            print("🫙 No new Open-Meteo data")
            return

    # Build Features
    if BOOTSTRAP:
        df_features = build_features(df_raw)
    else:
        # For incremental mode: need historical context to compute lags properly
        last_ts = safe_read(fg)["timestamp"].max()
        print(f"⏱️ Last timestamp in FS: {last_ts}")
        
        # Fetch raw data going back 30 hours to get L 24-hour lag context
        history_start = (last_ts - timedelta(hours=30)).strftime("%Y-%m-%d")
        
        df_raw_with_history = fetch_openmeteo_data(
            start_date=history_start,
            end_date=datetime.utcnow().strftime("%Y-%m-%d")
        )
        
        if df_raw_with_history.empty:
            print("🫙 No raw data available for feature building")
            return
        
        # Build features from history + new to compute lags correctly
        df_features_all = build_features(df_raw_with_history)
        
        # Only keep rows that are new (after last_ts in feature store)
        df_features = df_features_all[df_features_all["timestamp"] > last_ts].copy()

    
    if df_features.empty:
        print("🫙 No features generated")
        return
    
    print(f"📊 Generated {len(df_features)} feature rows")
    
    
    # Push To Hopsworks
    try:
        push_features(fg, df_features)
        
        # Save local copy on success
        df_features.to_parquet("latest_features.parquet", index=False)
        print("💾 Features saved locally to latest_features.parquet")
        
        print("✅ Pipeline finished successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to push features: {e}")
        logger.error("Pipeline failed during feature insertion")
        
        # Re-raise to fail the pipeline
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"🔴 Pipeline failed with error: {e}")
        exit(1)