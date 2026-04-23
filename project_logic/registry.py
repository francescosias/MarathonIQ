import joblib
import os

def load_model_trained():
    model_path = os.path.join("models", "xgboost_model.joblib")

    model = joblib.load(model_path)
    return model
