from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import joblib
import os

# Custom project imports
from project_logic.predict import predict

app = FastAPI()

# --- MODEL LOADING ---
# Ensure models are in the 'models' folder
GEN_MODEL_PATH = os.path.join("models", "marathon_pipeline.joblib")
EXP_MODEL_PATH = os.path.join("models", "marathon_expert_pipeline.joblib")

app.state.model_general = joblib.load(GEN_MODEL_PATH)
app.state.model_expert = joblib.load(EXP_MODEL_PATH)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SCHEMA ---
class RunnerData(BaseModel):
    age: int
    weekly_mileage_km: float
    running_experience_months: int
    resting_heart_rate_bpm: int
    vo2_max: float
    recovery_score: float
    injury_count: int
    nutrition_score: float
    run_club_attendance_rate: int
    course_difficulty: int
    injury_severity: float
    marathon_weather_Cold: float
    marathon_weather_Hot: float
    marathon_weather_Rainy: float
    marathon_weather_Windy: float
    # Expert feature is optional to keep general endpoint stable
    personal_best_minutes: Optional[float] = None

@app.get("/")
def index():
    return {"status": "Marathon API with Multi-Model support is running"}

# --- PREDICTION ENDPOINTS ---

@app.post('/predict/general')
def get_general_prediction(runner: RunnerData):
    """Uses general model logic"""
    data = runner.dict()
    # Explicitly set model_type to 'general'
    res = predict(app.state.model_general, data, model_type="general")
    return {"predicted_finish_time": res}

@app.post('/predict/expert')
def get_expert_prediction(runner: RunnerData):
    """Uses expert model logic including personal best"""
    data = runner.dict()

    if data.get('personal_best_minutes') is None:
        return {"error": "Expert model requires 'personal_best_minutes' input."}

    # Explicitly set model_type to 'expert'
    res = predict(app.state.model_expert, data, model_type="expert")
    return {"predicted_finish_time": res}
