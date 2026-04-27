import joblib
import os

def load_model(model_name: str):
    # model_name: 'general' or 'expert'
    filename = "marathon_pipeline.joblib" if model_name == "general" else "marathon_expert_pipeline.joblib"
    model_path = os.path.join("models", filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file couldn't found: {model_path}")

    return joblib.load(model_path)
