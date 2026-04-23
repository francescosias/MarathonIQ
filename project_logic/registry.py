import joblib
import os

def load_model_trained():
    model_path = os.path.join("models", "marathon_pipeline.joblib")

    model = joblib.load(model_path)
    return model
