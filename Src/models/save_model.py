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

        # Only include numeric metrics (exclude 'best_model' which is a string)
        best_model_name = metrics.get("best_model", "unknown")
        best_scores = metrics.get(best_model_name, {})
        flat_metrics = {}
        for k, v in best_scores.items():
            try:
                flat_metrics[k] = float(v)
            except Exception:
                pass

        project = hopsworks.login(
            api_key_value=api_key,
            project=project_name
        )
        mr = project.get_model_registry()
        model = mr.python.create_model(
            name="aqi_model",
            metrics=flat_metrics,
            description="Daily AQI model training pipeline"
        )
        model.save(model_path)
        print("✔ Model uploaded to Hopsworks Model Registry")
    except Exception as e:
        import traceback
        print("⚠️ Failed to upload model to registry:", e)
        traceback.print_exc()
