import pandas as pd
import numpy as np

# ============================================================
# MEDIANS — from training data (update if retrained)
# ============================================================
MEDIANS_GENERAL = {
    'vo2_max':                45.2,
    'resting_heart_rate_bpm': 68.0,
    'recovery_score':          6.0,
}

MEDIANS_EXPERT = {
    'vo2_max':                45.2,
    'resting_heart_rate_bpm': 68.0,
    'recovery_score':          6.0,
    'personal_best_minutes':  240.0,
}

# ============================================================
# EXPECTED COLUMNS — order must match training exactly
# ============================================================
COLUMNS_GENERAL = [
    'age', 'running_experience_months', 'weekly_mileage_km',
    'resting_heart_rate_bpm', 'vo2_max', 'recovery_score',
    'injury_count', 'injury_severity',
    'run_club_attendance_rate', 'course_difficulty',
    'marathon_weather_Cold', 'marathon_weather_Hot',
    'marathon_weather_Rainy', 'marathon_weather_Windy'
]

COLUMNS_EXPERT = [
    'age', 'running_experience_months', 'personal_best_minutes',
    'weekly_mileage_km', 'resting_heart_rate_bpm', 'vo2_max',
    'recovery_score', 'injury_count', 'injury_severity',
    'run_club_attendance_rate', 'course_difficulty',
    'marathon_weather_Cold', 'marathon_weather_Hot',
    'marathon_weather_Rainy', 'marathon_weather_Windy'
]

def preprocess_features(user_dict: dict,
                        model_type: str = "general") -> pd.DataFrame:
    """
    Frontend sends fully encoded numeric values.
    Backend only handles median imputation and column alignment.

    Fields already handled by frontend:
    - injury_severity: int (0/1/2/3)
    - course_difficulty: int (1/2/3)
    - marathon_weather_*: int (0/1)

    Fields imputed here if 0:
    - vo2_max, resting_heart_rate_bpm, recovery_score, nutrition_score
    """

    # 1. Convert dict to DataFrame
    df = pd.DataFrame([user_dict])

    # 2. Select config
    expected_columns = COLUMNS_EXPERT if model_type == "expert" else COLUMNS_GENERAL
    medians          = MEDIANS_EXPERT  if model_type == "expert" else MEDIANS_GENERAL

    # 3. Build zero skeleton — correct column order
    df_processed = pd.DataFrame(0, index=np.arange(1), columns=expected_columns)

    # 4. Populate with frontend values
    for col in expected_columns:
        if col in df.columns:
            df_processed[col] = df[col].values

    # 5. Median imputation — optional fields left at 0
    for col, median in medians.items():
        if col in df_processed.columns:
            if df_processed[col].iloc[0] == 0:
                df_processed[col] = median

    return df_processed
