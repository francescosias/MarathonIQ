import pandas as pd
from project_logic.preprocess import preprocess_features

def predict(model, data: dict) -> float:

    X_processed = preprocess_features(data)

    prediction_array = model.predict(X_processed)

    final_prediction = float(prediction_array[0])

    return final_prediction
