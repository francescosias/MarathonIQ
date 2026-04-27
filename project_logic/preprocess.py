import pandas as pd
import numpy as np

def preprocess_features(user_dict: dict) -> pd.DataFrame:
    """
    Converts the raw JSON data (dictionary) received from the user
    into the exact 15-column format expected by the trained XGBoost model.
    """
    # 1. Convert the single-row dictionary into a DataFrame
    df = pd.DataFrame([user_dict])

    # 2. Ordinal Encoding (Mapping logic kept exactly from Chris's notebook)
    course_map   = {"Flat": 1, "Mixed": 2, "Hilly": 3}
    injury_map   = {"Minor": 1, "Moderate": 2, "Severe": 3}

    # Apply mapping if these keys exist in the user's input
    if 'course_difficulty' in df.columns:
        df['course_difficulty'] = df['course_difficulty'].map(course_map).fillna(1)
    if 'injury_severity' in df.columns:
        df['injury_severity']   = df['injury_severity'].map(injury_map).fillna(0)

    # 3. Build the skeleton of the 15 columns (Order must be identical to training!)
    expected_columns = [
        'age', 'running_experience_months', 'weekly_mileage_km', 'resting_heart_rate_bpm',
        'vo2_max', 'recovery_score', 'injury_count', 'injury_severity', 'nutrition_score',
        'run_club_attendance_rate', 'course_difficulty',
        'marathon_weather_Cold', 'marathon_weather_Hot', 'marathon_weather_Rainy', 'marathon_weather_Windy'
    ]

    # Create a new DataFrame filled with zeros to ensure strict column count
    df_processed = pd.DataFrame(0, index=np.arange(1), columns=expected_columns)

    # 4. Populate with frontend values
    for col in expected_columns:
        if col in df.columns:
            df_processed[col] = df[col].values

    # 5. Manual One-Hot Encoding (Adapted for single API request)
    # e.g., If user selects marathon_weather="Cold", set 'marathon_weather_Cold' to 1
    if 'marathon_weather' in df.columns:
        weather_val = df.iloc[0]['marathon_weather']
        weather_col = f"marathon_weather_{weather_val}"
        if weather_col in expected_columns:
            df_processed[weather_col] = 1

    # 6. Fill Missing Values (Using baseline medians from Chris's notebook)
    default_medians = {
        'vo2_max': 45.0,
        'nutrition_score': 5.0
    }

    for col, default_val in default_medians.items():
        if pd.isna(df_processed.iloc[0][col]):
            df_processed.at[0, col] = default_val

    return df_processed
