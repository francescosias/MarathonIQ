import pandas as pd
from project_logic.preprocess import preprocess_features

def predict(model, data: dict, model_type: str = "general") -> float:
    """
    Revised to pass the model_type to the preprocessor.
    This ensures 'personal_best_minutes' is handled for expert models.
    """
    # Important: Pass the model_type down to determine column structure
    #X_processed = preprocess_features(data, model_type=model_type)

    prediction_array = model.predict(data)

    final_prediction = float(prediction_array[0])

    return final_prediction
