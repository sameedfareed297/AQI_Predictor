import os
import hopsworks
from dotenv import load_dotenv
import joblib
import json

load_dotenv()

model_path = "Artifacts/model.joblib"

with open("Artifacts/metrics.json", "r") as f:
    metrics = json.load(f)
best_model_name = metrics.get("best_model", "unknown")
best_scores = metrics.get(best_model_name, {})
flat_metrics = {}
for k, v in best_scores.items():
    try:
        flat_metrics[k] = float(v)
    except Exception:
        pass
print(f"[DEBUG] Numeric-only metrics: {flat_metrics}")

try:
    api_key = os.getenv("HOPSWORKS_API_KEY")
    project_name = os.getenv("HOPSWORKS_PROJECT_NAME")
    print(f"[DEBUG] Using Hopsworks project: {project_name}")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    mr = project.get_model_registry()
    print("[DEBUG] Accessed model registry.")
    model_obj = mr.python.create_model(
        name="aqi_model",
        metrics=flat_metrics,
        description="Manual upload of AQI model with numeric-only metrics."
    )
    print("[DEBUG] Created model object in registry.")
    print("[DEBUG] About to save main model to registry...")
    model_obj.save(model_path)
    print("[DEBUG] Main model save() call completed.")
    print("✔ Main model uploaded to Hopsworks Model Registry (manual, numeric-only metrics)")
except Exception as e:
    import traceback
    print("⚠️ Failed to upload main model to registry:", e)
    traceback.print_exc()
