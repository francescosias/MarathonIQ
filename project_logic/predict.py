import pandas as pd
import shap
from project_logic.preprocess import preprocess_features

def predict(model, data: dict, model_type: str = "general") -> float:
    """
    Revised to pass the model_type to the preprocessor.
    This ensures 'personal_best_minutes' is handled for expert models.
    """
    # Important: Pass the model_type down to determine column structure
    X_processed = preprocess_features(data, model_type=model_type)

    prediction_array = model.predict(X_processed)

    final_prediction = float(prediction_array[0])


    scaler = model.named_steps['scaler']
    X_scaled = scaler.transform(X_processed)

    xgb_model = model.named_steps['xgb_model']

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_scaled)
    shap_dict = dict(zip(X_processed.columns, shap_values[0].tolist()))

    return {
        'prediction': final_prediction,
        'shap_values': shap_dict,
        'base_value': float(explainer.expected_value)
        }
