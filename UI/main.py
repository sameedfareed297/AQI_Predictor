import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
import hopsworks
import os
from dotenv import load_dotenv
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

# Fix: Import the CORRECT recursive forecast from utils.py
from utils import generate_forecast

# Page Configuration
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/karachi-aqi-predictor',
        'Report a bug': 'https://github.com/your-repo/karachi-aqi-predictor/issues',
        'About': "# Karachi AQI Predictor\nML-powered air quality prediction using GradientBoosting."
    }
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Force light theme */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        color-scheme: light !important;
        background-color: #f0f2f6 !important;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        padding: 0 !important;
    }
    
    .block-container {
        padding: 1.5rem 2rem !important;
        max-width: 1600px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 24px;
        margin: 1rem auto;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
    }
    
    /* Simple Header */
    .hero-header {
        background: #667eea;
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 2rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        margin: 0 0 1rem 0;
        opacity: 0.95;
    }
    
    .status-info {
        font-size: 0.9rem;
        display: flex;
        gap: 1.5rem;
        flex-wrap: wrap;
    }
    
    /* Simple Metric Cards */
    .metric-box {
        background: white;
        padding: 1.8rem 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
    }
    
    .metric-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: #7c8db5;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
    }
    
    .metric-value {
        font-size: 3.2rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.6rem 0;
        color: #667eea;
    }
    
    .metric-desc {
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 0.6rem;
    }
    
    /* Simple Forecast Cards */
    .forecast-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid;
    }
    
    /* Simple Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Simple Alert Boxes */
    .alert-box {
        background: #dc3545;
        color: white !important;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        font-weight: 500;
    }
    
    .alert-box * {
        color: white !important;
    }
    
    .warning-box {
        background: #f39c12;
        color: white !important;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        font-weight: 500;
    }
    
    .warning-box * {
        color: white !important;
    }
    
    /* Sidebar - Modern Dark Design */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #e4e4e7 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stMarkdown strong {
        color: #a78bfa !important;
        font-weight: 600;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(167, 139, 250, 0.2) !important;
        margin: 1rem 0;
    }
    
    [data-testid="stSidebar"] .stButton button {
        background: #667eea !important;
        color: white !important;
        border: none !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background: #5568d3 !important;
        opacity: 0.9;
    }
    
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.3rem;
    }
    
    /* Simple Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        background: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: #667eea !important;
        color: white !important;
    }
    
    /* Info boxes */
    div[data-testid="stInfo"] {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }
    
    /* Success boxes */
    div[data-testid="stSuccess"] {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        border-radius: 8px;
    }
    
    /* Warning boxes */
    div[data-testid="stWarning"] {
        background: rgba(251, 191, 36, 0.1);
        border-left: 4px solid #fbbf24;
        border-radius: 8px;
    }
    
    /* Error boxes */
    div[data-testid="stError"] {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Main Content Area Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f0f2f6;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 5px;
        border: 2px solid #f0f2f6;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #5568d3;
    }
    
    /* Sidebar scrollbar */
    [data-testid="stSidebar"]::-webkit-scrollbar {
        width: 10px;
    }
    
    [data-testid="stSidebar"]::-webkit-scrollbar-track {
        background: rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 5px;
        border: 2px solid rgba(26, 26, 46, 0.5);
    }
    
    [data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
        background: #a78bfa;
    }
</style>
""", unsafe_allow_html=True)


# Helper Functions
def get_aqi_category_info(aqi_value: float):
    """Get category label and color for an AQI value."""
    if aqi_value <= 50:
        return "Good 😌", "#00e400"
    elif aqi_value <= 100:
        return "Moderate 🙂", "#ffff00"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive 😐", "#ff7e00"
    elif aqi_value <= 200:
        return "Unhealthy 😷", "#ff0000"
    elif aqi_value <= 300:
        return "Very Unhealthy 🤢", "#8f3f97"
    else:
        return "Hazardous ☠️", "#7e0023"

def get_health_recommendation(aqi_value: float) -> str:
    """Get health recommendation based on AQI value."""
    if aqi_value > 200:
        return "Avoid outdoor activities. Wear N95 mask if necessary. Keep windows closed."
    elif aqi_value > 150:
        return "Sensitive groups should limit outdoor exposure. Consider using air purifiers."
    elif aqi_value > 100:
        return "Unusually sensitive individuals should consider limiting prolonged outdoor exertion."
    else:
        return "Air quality is acceptable for most people. Enjoy outdoor activities!"

# FIXED: Load historical data WITHOUT cached Streamlit commands
def load_historical_data_internal():
    """Load historical AQI data from Hopsworks Feature Store (no st.xxx calls)."""
    
    # Get API credentials - try Streamlit secrets first, then .env
    try:
        api_key = st.secrets["HOPSWORKS_API_KEY"]
        project_name = st.secrets["HOPSWORKS_PROJECT_NAME"]
    except (FileNotFoundError, KeyError):
        api_key = os.getenv("HOPSWORKS_API_KEY")
        project_name = os.getenv("HOPSWORKS_PROJECT_NAME")
    
    if not api_key or not project_name:
        raise ValueError("Missing Hopsworks credentials (HOPSWORKS_API_KEY or HOPSWORKS_PROJECT_NAME)")
    
    # Login to Hopsworks
    project = hopsworks.login(
        api_key_value=api_key,
        project=project_name
    )
    fs = project.get_feature_store()
    
    # Load from Feature Group
    fg = fs.get_feature_group(
        name="karachi_air_quality",
        version=6
    )
    
    # Read all data from feature group
    df = fg.read()
    
    if df is None or df.empty:
        raise ValueError("Feature group returned empty data")
    
    # Ensure timestamp column exists
    if "timestamp" not in df.columns:
        raise ValueError(f"'timestamp' column not found. Available columns: {df.columns.tolist()}")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Add time-based features if missing
    if "hour" not in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    if "day" not in df.columns:
        df["day"] = df["timestamp"].dt.day
    if "month" not in df.columns:
        df["month"] = df["timestamp"].dt.month
    if "weekday" not in df.columns:
        df["weekday"] = df["timestamp"].dt.weekday
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df

@st.cache_data(ttl=1800)
def load_historical_data():
    """Cached wrapper for loading data."""
    return load_historical_data_internal()

        
@st.cache_resource(show_spinner=False)
def get_model_metadata():
    """Get model metadata from artifacts directory."""
    metadata = {
        "name": "GradientBoosting",
        "best_model": "GradientBoosting",
        "mae": 0.2776,
        "rmse": 2.0713,
        "r2": 0.9976,
        "status": "✔️ Model Loaded"
    }
    
    try:
        project_root = Path(__file__).parent.parent
        metrics_path = project_root / "artifacts" / "metrics.json"
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                loaded_metrics = json.load(f)
                
                if 'GradientBoosting' in loaded_metrics:
                    gb_metrics = loaded_metrics['GradientBoosting']
                    metadata.update({
                        "mae": gb_metrics.get('MAE', metadata['mae']),
                        "rmse": gb_metrics.get('RMSE', metadata['rmse']),
                        "r2": gb_metrics.get('R2', metadata['r2']),
                        "best_model": loaded_metrics.get('best_model', 'GradientBoosting')
                    })
                else:
                    metadata.update({
                        "mae": loaded_metrics.get('mae', metadata['mae']),
                        "rmse": loaded_metrics.get('rmse', metadata['rmse']),
                        "r2": loaded_metrics.get('r2', metadata['r2']),
                        "best_model": loaded_metrics.get('best_model', 'GradientBoosting')
                    })
                
                metadata['status'] = " ✔️ Metrics Loaded"
    except Exception as e:
        pass
    
    return metadata

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model from artifacts directory."""
    try:
        project_root = Path(__file__).parent.parent
        model_path = project_root / "artifacts" / "model.joblib"
        
        if model_path.exists():
            model = joblib.load(model_path)
            return model
        else:
            return None
    except Exception as e:
        st.warning(f"⚠️ Could not load model: {str(e)}")
        return None


# Show Initial Loading State
loading_placeholder = st.empty()
with loading_placeholder.container():
    st.markdown("""
    <div style='text-align: center; padding: 3rem;'>
        <h2 style='color: #667eea;'>🌫️ Karachi Air Quality Monitor</h2>
        <p style='color: #a0adc7;'>Loading air quality data...</p>
    </div>
    """, unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("🔑 Connecting to Hopsworks...")
    progress_bar.progress(25)

# Load Data - With Handling
try:
    with st.spinner("🔐 Connecting to Hopsworks..."):
        historical_df = load_historical_data()
    
    progress_bar.progress(60)
    status_text.text("📊 Loading model metadata...")
    
    model_metadata = get_model_metadata()
    progress_bar.progress(80)
    status_text.text("🤖 Loading prediction model...")
    
    model = load_model()
    progress_bar.progress(100)
    status_text.text("✔️ Ready!")
    
    # Clear loading screen
    import time
    time.sleep(0.5)
    loading_placeholder.empty()
    
except Exception as e:
    loading_placeholder.empty()
    st.error(f"❌ Failed to initialize app")
    st.error(f"Error type: {type(e).__name__}")
    st.error(f"Error message: {str(e)}")
    
    with st.expander("🔍 Full Error Traceback"):
        import traceback
        st.code(traceback.format_exc())
    
    with st.expander("🔧 Troubleshooting Steps"):
        st.markdown("""
        **1. Verify Hopsworks Credentials**
        - API Key: `HOPSWORKS_API_KEY` in `.env` or Streamlit secrets
        - Project: `HOPSWORKS_PROJECT_NAME` in `.env` or Streamlit secrets
        
        **2. Check Feature Group Exists**
        - Feature group name: `karachi_air_quality`
        - Version: `6`
        
        **3. Manual Connection Test**
        Run in terminal:
```bash
        python -c "import hopsworks; p=hopsworks.login(api_key_value='YOUR_KEY', project='PROJECT_NAME'); print(p.get_feature_store().get_feature_group('karachi_air_quality', 6).read())"
```
        """)
    
    st.stop()

# Get recent data (last 7 days) - MUST BE AFTER LOADING
recent_df = historical_df.tail(24 * 7)
latest_data = historical_df.iloc[-1]

# Calculate data freshness - MUST BE AFTER LOADING
latest_ts = pd.to_datetime(latest_data['timestamp'])
now_aware = datetime.now(timezone.utc) if latest_ts.tzinfo else datetime.now()
data_age_hours = (now_aware - latest_ts).total_seconds() / 3600

# SIDEBAR: CI/CD PIPELINE STATUS
with st.sidebar:
    st.markdown("### 🪈 Pipeline Status")
    st.markdown("---")
    
    st.markdown("**📊 Data Ingestion**")
    st.caption("Runs: Hourly via GitHub Actions")
    st.success("Active", icon="✔️")
    st.caption(f"Last data point: {pd.to_datetime(historical_df['timestamp'].iloc[-1]).strftime('%b %d, %I:%M %p')}")
    
    st.markdown("**🤖 Training Pipeline**")
    st.caption("Runs: Daily @ 8:00 AM")
    st.info("Scheduled", icon="📊")
    
    st.markdown("**🎯 Active Model**")
    if model is not None:
        st.success(f" {model_metadata['best_model']}", icon="✔️")
        st.caption(f"Status: {model_metadata['status']}")
        st.markdown("**Performance Metrics:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MAE", f"{model_metadata['mae']:.2f}")
            st.metric("R²", f"{model_metadata['r2']:.4f}")
        with col2:
            st.metric("RMSE", f"{model_metadata['rmse']:.2f}")
    else:
        st.error("❌ Model not loaded")
    
    st.markdown("---")
    
    st.markdown("**📅 Data Information**")
    st.caption(f"Updated: {latest_data['timestamp'].strftime('%b %d, %I:%M %p')}")
    st.caption(f"Total Records: {len(historical_df)}")
    st.caption(f"7-Day Average: {recent_df['aqi'].mean():.0f}")
    
    st.markdown("---")
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# Handle future timestamps (timezone bug)
if data_age_hours < -1:
    data_freshness_icon = "🚨"
    data_freshness_text = f"Sync issue ({abs(data_age_hours):.1f}h)"
elif data_age_hours < 0:
    data_freshness_icon = "✔️"
    data_freshness_text = "Just updated"
else:
    data_freshness_icon = "✔️"
    data_freshness_text = f"Updated {data_age_hours:.1f}h ago"

# Header
st.markdown(f"""
<div class="hero-header">
    <h1 class="hero-title">🌫️ Karachi Air Quality Monitor</h1>
    <p class="hero-subtitle">Real-Time Air Quality Monitoring & Forecasting</p>
    <div class="status-info">
        <span>� Last Measurement: {latest_data['timestamp'].strftime('%B %d, %Y at %I:%M %p')}</span>
        <span>{data_freshness_icon} {data_freshness_text}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Key Metrics
current_aqi = latest_data['aqi']
current_category, current_color = get_aqi_category_info(current_aqi)
avg_24h = recent_df.tail(24)['aqi'].mean()
avg_category, avg_color = get_aqi_category_info(avg_24h)
max_7d = recent_df['aqi'].max()
max_category, max_color = get_aqi_category_info(max_7d)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Current AQI</div>
        <div class="metric-value" style='color: {current_color};'>{current_aqi:.0f}</div>
        <div class="metric-desc" style='color: {current_color};'>{current_category}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">24-Hour Avg</div>
        <div class="metric-value" style='color: {avg_color};'>{avg_24h:.0f}</div>
        <div class="metric-desc" style='color: {avg_color};'>{avg_category}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">7-Day Peak</div>
        <div class="metric-value" style='color: {max_color};'>{max_7d:.0f}</div>
        <div class="metric-desc" style='color: {max_color};'>{max_category}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    unhealthy_hours = len(recent_df[recent_df['aqi'] > 150])
    if unhealthy_hours == 0:
        alert_icon, alert_text, alert_color = "✔️", "All Clear", "#28a745"
    elif unhealthy_hours <= 24:
        alert_icon, alert_text, alert_color = "⚠️", f"{unhealthy_hours}h Alert", "#f39c12"
    else:
        alert_icon, alert_text, alert_color = "🚨", f"{unhealthy_hours}h Critical", "#e74c3c"
    
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">Health Alert</div>
        <div style='font-size: 2.5rem; margin: 0.5rem 0;'>{alert_icon}</div>
        <div class="metric-desc" style='color: {alert_color};'>{alert_text}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Health Alert Banner
if current_aqi > 150:
    st.markdown(f"""
    <div class='alert-box'>
        <div style='font-size: 1.8rem; margin-bottom: 0.5rem; font-weight: 700;'>🚨 Air Quality Health Alert</div>
        <div style='font-size: 1.1rem; margin-bottom: 0.5rem;'>Current AQI: {current_aqi:.0f} - {current_category}</div>
        <div style='margin-top: 0.5rem; font-size: 0.95rem; line-height: 1.5;'>{get_health_recommendation(current_aqi)}</div>
    </div>
    """, unsafe_allow_html=True)
elif current_aqi > 100:
    st.markdown(f"""
    <div class='warning-box'>
        <div style='font-size: 1.5rem; margin-bottom: 0.3rem; font-weight: 700;'>⚠️ Air Quality Advisory</div>
        <div style='font-size: 1rem; margin-bottom: 0.5rem;'>Current AQI: {current_aqi:.0f} - {current_category}</div>
        <div style='margin-top: 0.5rem; font-size: 0.9rem; line-height: 1.5;'>{get_health_recommendation(current_aqi)}</div>
    </div>
    """, unsafe_allow_html=True)

# 3-DAY Forecast
st.markdown("<div class='section-header'>🔮 3-Day Forecast</div>", unsafe_allow_html=True)

if model is not None:
    st.info(f"📊 Predictions from {model_metadata['best_model']} (MAE: {model_metadata['mae']:.2f}) — Recursive multi-step forecasting")

st.markdown("<br>", unsafe_allow_html=True)

if model is not None:
    forecast_df = generate_forecast(historical_df, model, days=3)
    
    if forecast_df is not None and not forecast_df.empty:
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Combine historical (last 7 days) and forecast
            recent_history = historical_df.tail(24 * 7)[['timestamp', 'aqi']].copy()
            recent_history['type'] = 'Historical'
            recent_history.rename(columns={'aqi': 'value'}, inplace=True)
            
            forecast_plot = forecast_df.copy()
            forecast_plot['type'] = 'Forecast'
            forecast_plot.rename(columns={'aqi_predicted': 'value'}, inplace=True)
            
            combined = pd.concat([recent_history, forecast_plot], ignore_index=True)
            
            fig = px.line(combined, x='timestamp', y='value', color='type',
                         title='AQI Trend: Historical + 3-Day Forecast',
                         labels={'value': 'AQI', 'timestamp': 'Date & Time'},
                         color_discrete_map={'Historical': '#667eea', 'Forecast': '#f39c12'})
            
            fig.add_hline(y=50, line_dash="dash", line_color="green", opacity=0.3,
                         annotation_text="Good")
            fig.add_hline(y=100, line_dash="dash", line_color="yellow", opacity=0.3,
                         annotation_text="Moderate")
            fig.add_hline(y=150, line_dash="dash", line_color="orange", opacity=0.3,
                         annotation_text="Unhealthy")
            
            fig.update_layout(
                height=400, 
                hovermode='x unified',
                plot_bgcolor='rgba(37, 45, 61, 0.5)',
                paper_bgcolor='#1a1f2e',
                font=dict(color='#e0e0e0'),
                xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(102, 126, 234, 0.2)'),
                yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(102, 126, 234, 0.2)')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.markdown("#### 📅 Next 3 Days")
            
            for day in range(1, 4):
                day_start = day - 1
                day_data = forecast_df.iloc[day_start * 24:(day_start + 1) * 24]
                avg_aqi = day_data['aqi_predicted'].mean()
                max_aqi = day_data['aqi_predicted'].max()
                min_aqi = day_data['aqi_predicted'].min()
                
                category, color = get_aqi_category_info(avg_aqi)
                today = datetime.now()
                forecast_date = (today + timedelta(days=day)).strftime('%b %d')
                
                st.markdown(f"""
                <div class='forecast-card' style='border-left-color: {color};'>
                    <div style='font-weight: 600; font-size: 0.85rem; color: #666; margin-bottom: 0.5rem;'>
                        {forecast_date} (Day {day})
                    </div>
                    <div style='font-size: 2rem; font-weight: 700; color: {color}; margin: 0.3rem 0;'>{avg_aqi:.0f}</div>
                    <div style='font-size: 0.9rem; color: {color}; font-weight: 600; margin-bottom: 0.5rem;'>{category}</div>
                    <div style='font-size: 0.8rem; color: #888;'>Range: {min_aqi:.0f} - {max_aqi:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.caption(f"Model: {model_metadata['best_model']}")
    else:
        st.warning("⚠️ Forecast returned empty. Check that historical_df has an 'aqi' column and at least 49 rows.")
else:
    st.warning("⚠️ Prediction model not available. Please train the model first.")
    st.info("💡 Run: `python -m Src.Pipeline.train_daily` to train the model.")

st.markdown("<br>", unsafe_allow_html=True)

# Health Recommendations
st.markdown("<div class='section-header'>💡 Health Recommendations</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

recommendation = get_health_recommendation(current_aqi)
if current_aqi <= 50:
    st.success(f"✔️ {recommendation}")
elif current_aqi <= 100:
    st.info(f"ℹ️ {recommendation}")
elif current_aqi <= 150:
    st.warning(f"⚠️ {recommendation}")
else:
    st.error(f"🚨 {recommendation}")

st.divider()

# Detailed Analytics (TABS)
st.markdown("<div class='section-header'>📊 Detailed Analytics & Insights</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Historical Trends", "🧪 Pollutant Analysis", "📖 AQI Guide & Model Info"])

with tab1:
    st.markdown("#### Past 7 Days Air Quality Trend")
    
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Scatter(
        x=recent_df['timestamp'],
        y=recent_df['aqi'],
        mode='lines',
        name='AQI',
        line=dict(color='#667eea', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)',
        hovertemplate='<b>%{x|%b %d, %I:%M %p}</b><br>AQI: %{y:.0f}<extra></extra>'
    ))
    
    fig_hist.add_hline(y=50, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Good")
    fig_hist.add_hline(y=100, line_dash="dash", line_color="yellow", opacity=0.5, annotation_text="Moderate")
    fig_hist.add_hline(y=150, line_dash="dash", line_color="orange", opacity=0.5, annotation_text="Unhealthy")
    
    fig_hist.update_layout(
        title="Past 7 Days Air Quality Trend",
        xaxis_title="Date",
        yaxis_title="Air Quality Index (AQI)",
        height=450,
        hovermode='x unified',
        plot_bgcolor='rgba(37, 45, 61, 0.5)',
        paper_bgcolor='#1a1f2e',
        font=dict(color='#e0e0e0'),
        showlegend=False
    )
    
    fig_hist.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(102, 126, 234, 0.2)')
    fig_hist.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(102, 126, 234, 0.2)')
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Weekly Average", f"{recent_df['aqi'].mean():.0f}")
    with col2:
        st.metric("📈 This Week (PEAK)", f"{recent_df['aqi'].max():.0f}")
    with col3:
        st.metric("📉 This Week (BEST)", f"{recent_df['aqi'].min():.0f}")

with tab2:
    st.markdown("#### Current Pollutant Breakdown")
    
    pollutants = {
        "PM2.5": "pm2_5",
        "PM10": "pm10",
        "CO": "carbon_monoxide",
        "NO₂": "nitrogen_dioxide",
        "SO₂": "sulphur_dioxide",
        "O₃": "ozone"
    }
    
    pollutant_data = []
    for name, col in pollutants.items():
        if col in latest_data.index and pd.notna(latest_data[col]):
            try:
                pollutant_data.append({"Pollutant": name, "Concentration": float(latest_data[col])})
            except (ValueError, TypeError):
                pass
    
    if pollutant_data:
        pollutant_df = pd.DataFrame(pollutant_data)
        
        fig_pollutants = px.bar(
            pollutant_df,
            x="Pollutant",
            y="Concentration",
            title="Current Pollutant Concentrations (μg/m³)",
            color="Concentration",
            color_continuous_scale=["#28a745", "#ffc107", "#dc3545"],
            text="Concentration"
        )
        
        fig_pollutants.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_pollutants.update_layout(
            height=400, 
            showlegend=False,
            plot_bgcolor='rgba(37, 45, 61, 0.5)',
            paper_bgcolor='#1a1f2e',
            font=dict(color='#e0e0e0'),
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(102, 126, 234, 0.2)'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(102, 126, 234, 0.2)')
        )
        
        st.plotly_chart(fig_pollutants, use_container_width=True)
        
        st.info("""
        **📌 Pollutant Guide:**
        - **PM2.5 & PM10**: Particulate matter - main AQI contributor
        - **CO**: Carbon monoxide from vehicles
        - **NO₂**: Nitrogen dioxide from traffic and industry
        - **SO₂**: Sulfur dioxide from fossil fuels
        - **O₃**: Ozone from sunlight reacting with pollutants
        """)
    else:
        st.info("📊 Pollutant data not available in current dataset")

with tab3:
    st.markdown("#### Understanding AQI Levels")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🟢 Good (0–50)
        Air quality is excellent — perfect for all outdoor activities.

        #### 🟡 Moderate (51–100)
        Air quality is acceptable — safe for most people.

        #### 🟠 Unhealthy for Sensitive Groups (101–150)
        Sensitive groups should limit prolonged outdoor exposure.
        """)

    with col2:
        st.markdown("""
        #### 🔴 Unhealthy (151–200)
        Everyone should reduce prolonged outdoor activities.

        #### 🟣 Very Unhealthy (201–300)
        Health alert — avoid outdoor activities.

        #### 🟤 Hazardous (301+)
        Emergency conditions — stay indoors.
        """)
            

    st.divider()

    st.markdown("🤖 Model Info")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        <div class="model-card">
            <h4>📊 Active Model: {model_metadata['best_model']}</h4>
            <p><strong>Status:</strong> {model_metadata['status']}</p>
            <p><strong>Performance Metrics:</strong></p>
            <ul>
                <li><strong>MAE:</strong> {model_metadata['mae']:.4f} AQI points (mean absolute error)</li>
                <li><strong>RMSE:</strong> {model_metadata['rmse']:.4f} (root mean squared error)</li>
                <li><strong>R² Score:</strong> {model_metadata['r2']:.4f} (prediction accuracy)</li>
            </ul>
            <p style='margin-top: 1rem; color: #28a745;'><strong>✔️ Best High-performing model trained on Karachi air quality data</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.info("""
        **🔬 Model Insights**

        📓 **SHAP Analysis**
        `notebooks/shap_analysis.ipynb`

        Understand:
        - Feature importance
        - Model decisions
        - Prediction breakdown
        """)

    st.markdown("#### 🏆 Model Comparison")

    metrics_data = {
        "Model": ["LinearRegression", "RandomForest", "GradientBoosting"],
        "MAE": [6.9263, 6.9418, 6.5196],
        "RMSE": [10.7103, 10.2606, 9.7755],
        "R²": [0.8809, 0.8907, 0.9008]
    }
    metrics_df = pd.DataFrame(metrics_data)

    def highlight_best(row):
        if row["Model"] == "GradientBoosting":
            return ['background-color: lightgreen'] * len(row)
        return [''] * len(row)

    st.dataframe(
        metrics_df.style.apply(highlight_best, axis=1)
                        .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}),
        use_container_width=True,
        hide_index=True
    )

    st.success("✔️ **Best Model: GradientBoosting** (Lowest MAE & RMSE, Highest R²)")
    st.caption("**Training Data: ** 8736 Samples  ")
st.divider()
