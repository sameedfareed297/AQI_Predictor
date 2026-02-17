import joblib
import os
import json
import hopsworks
from dotenv import load_dotenv

load_dotenv()

def save_models(models, metrics, folder="Artifacts"):
    os.makedirs(folder, exist_ok=True)

    # Pick best model based on RMSE
    best_model_name = min(metrics, key=lambda x: metrics[x]["RMSE"])
    best_model = models[best_model_name]

    # Save locally first
    model_path = os.path.join(folder, "model.joblib")
    joblib.dump(best_model, model_path)

    # Save metrics
    metrics["best_model"] = best_model_name
    metrics_path = os.path.join(folder, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f" - Model saved locally: {model_path}")
    print(f" - Metrics saved locally: {metrics_path}")
    print(f"🏆 Best model: {best_model_name}")

    # ==============================
    # Upload to Hopsworks Model Registry
    # ==============================

    try:
        api_key = os.getenv("HOPSWORKS_API_KEY")
        project_name = os.getenv("HOPSWORKS_PROJECT_NAME")

        project = hopsworks.login(
            api_key_value=api_key,
            project=project_name
        )

        mr = project.get_model_registry()

        # Create new version each day
        model = mr.python.create_model(
            name="aqi_model",
            metrics=metrics,
            description="Daily AQI model training pipeline"
        )

        model.save(model_path)

        print("✔ Model uploaded to Hopsworks Model Registry")

    except Exception as e:
        print("⚠️ Failed to upload model to registry:", e)
