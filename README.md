# 🌫️ Karachi Air Quality Index (AQI) Prediction System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-red.svg)

An **end-to-end MLOps system** for monitoring, predicting, and visualizing air quality in Karachi, Pakistan. The platform automates data ingestion, feature engineering, model training, and deployment to provide real-time AQI monitoring and 3-day forecasts.

This project was developed during a Data Science internship at **10Pearls** and follows modern production-level ML and MLOps practices.

---

## 🎯 Key Features

### ⚡ Automated Data Pipeline

* Hourly air quality data collection from Open-Meteo API
* Advanced feature engineering with lag, rolling, and temporal features
* Versioned storage using **Hopsworks Feature Store**

### 🤖 Machine Learning & Model Registry

* Models: Ridge Regression, Random Forest, XGBoost
* Automatic model selection based on RMSE
* Time-series optimized training
* Model versioning using Hopsworks Model Registry

### 📊 Interactive Dashboard

Built with **Streamlit**:

* Real-time AQI monitoring
* 3-day hourly forecast
* Historical AQI analysis
* Health recommendations
* Live model metrics

### 🔄 CI/CD Automation

* Hourly feature pipeline
* Daily model training
* Fully automated GitHub Actions workflows

---

## 📊 System Architecture

```
Open-Meteo API
      ↓
Feature Pipeline (Hourly)
      ↓
Hopsworks Feature Store
      ↓
Training Pipeline (Daily)
      ↓
Hopsworks Model Registry
      ↓
Streamlit Dashboard
```

---

## 🚀 Quick Start

### ✅ Prerequisites

* Python 3.10+
* Git
* Free Hopsworks account

---

### ⚙️ Local Setup

#### 1️⃣ Clone repository

```bash
git clone https://github.com/sameedfareed297/AQI_Predictor.git
cd AQI_Predictor
```

#### 2️⃣ Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

#### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Setup environment variables

Create `.env` file:

```env
HOPSWORKS_API_KEY=your_api_key
HOPSWORKS_PROJECT_NAME=your_project_name
```

#### 5️⃣ Run Streamlit dashboard

```bash
streamlit run UI/main.py
```

---

## ☁️ Deployment

### 🚀 Deploy on Streamlit Cloud

1. Fork this repository
2. Go to Streamlit Cloud
3. Create a new app
4. Main file:

```
UI/main.py
```

5. Add secrets:

```toml
HOPSWORKS_API_KEY="your_key"
HOPSWORKS_PROJECT_NAME="your_project"
```

---

## 🧠 Manual Pipeline Execution

### Feature pipeline

```bash
python -m Src.main
```

### Training pipeline

```bash
python -m Src.Pipeline.train_daily
```

---

## 📁 Project Structure

```
AQI_Predictor/
│
├─ .github/workflows/         # CI/CD automation
│  ├─ feature_pipeline.yml    # Hourly pipeline
│  └─ training_pipeline.yml   # Daily training
│
├─ Artifacts/                 # Persisted models & metrics
│
├─ Src/
│  ├─ data_ingestion/         # API clients and ingestion
│  ├─ features/               # Feature engineering
│  ├─ feature_store/          # Hopsworks adapters
│  ├─ models/                 # Training, evaluation
│  ├─ Pipeline/               # Orchestration
│  └─ utils/                  # Shared utilities
│
├─ UI/                        # Streamlit dashboard
│
├─ Scripts/                   # Helper scripts
├─ notebooks/                 # EDA and SHAP analysis
├─ requirements.txt
└─ README.md
```

---

## 📈 Model Performance

Metrics are automatically updated after each training run.

| Metric | Description             |
| ------ | ----------------------- |
| RMSE   | Root Mean Squared Error |
| MAE    | Mean Absolute Error     |
| R²     | Model accuracy          |

---

## 🔬 Model Explainability

SHAP analysis includes:

* Feature importance
* Temporal patterns
* Pollutant impact
* Interaction effects

---

## 🔄 CI/CD Pipelines

### 🕐 Feature Pipeline (Hourly)

* Fetch data
* Build features
* Validate
* Store in feature store

### 🌙 Training Pipeline (Daily)

* Load feature store
* Train models
* Evaluate
* Select best
* Register model

---

## 🔐 Security

* Secrets stored securely
* No sensitive data
* Public AQI datasets

---

## 🐛 Troubleshooting

### Model not loading

* Check Hopsworks credentials
* Ensure model registry contains latest version
* Reboot Streamlit app

### Pipeline failure

* Verify GitHub secrets
* Check workflow logs

---

## 📝 Future Improvements

* Deep learning models (LSTM, Transformers)
* Ensemble forecasting
* Mobile app
* Alert notifications
* Azure deployment

---

## 🤝 Contributing

Pull requests are welcome.

---

## 📧 Contact

Sameed Fareed
GitHub: [https://github.com/sameedfareed297](https://github.com/sameedfareed297)

---

**Last Updated: February 2026**

