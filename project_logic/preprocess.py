import pandas as pd
import numpy as np

def preprocess_features(user_dict: dict, model_type: str = "general") -> pd.DataFrame:
    """
    Converts raw JSON data into the exact column format expected by the
    specific model (general or expert).
    """
    # 1. Convert the single-row dictionary into a DataFrame
    df = pd.DataFrame([user_dict])

    # 2. Define the base skeleton (15 columns for general model)
    if model_type == "expert":
        expected_columns = [
            'age',
            'running_experience_months',
            'personal_best_minutes',  # Correct position: 3rd column (index 2)
            'weekly_mileage_km',
            'resting_heart_rate_bpm',
            'vo2_max',
            'recovery_score',
            'injury_count',
            'injury_severity',
            'nutrition_score',
            'run_club_attendance_rate',
            'course_difficulty',
            'marathon_weather_Cold',
            'marathon_weather_Hot',
            'marathon_weather_Rainy',
            'marathon_weather_Windy'
        ]
    else:
        # General model doesn't have personal_best_minutes
        expected_columns = [
            'age', 'running_experience_months', 'weekly_mileage_km', 'resting_heart_rate_bpm',
            'vo2_max', 'recovery_score', 'injury_count', 'injury_severity', 'nutrition_score',
            'run_club_attendance_rate', 'course_difficulty',
            'marathon_weather_Cold', 'marathon_weather_Hot', 'marathon_weather_Rainy', 'marathon_weather_Windy'
        ]

    # 4. Create a new DataFrame filled with zeros to ensure strict column count and order
    df_processed = pd.DataFrame(0, index=np.arange(1), columns=expected_columns)

<<<<<<< HEAD
    # 4. Populate with frontend values
    for col in expected_columns:
        if col in df.columns:
            df_processed[col] = df[col].values
=======
    # 5. Populate the skeleton with the user's actual data
    for col in expected_columns:
        if col in df.columns:
            df_processed[col] = df[col]
>>>>>>> master



    # 6. Fill Missing Values (Medians)
    # Added personal_best_minutes median for safety
    default_medians = {
        'vo2_max': 45.0,
        'recovery_score': 70.0,
        'personal_best_minutes': 240.0
    }

    for col, val in default_medians.items():
        if col in df_processed.columns and df_processed[col].iloc[0] == 0:
            # We only fill if it's strictly zero/missing and exists in the model
            df_processed[col] = df_processed[col].replace(0, val)

    return df_processed
