from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from project_logic.registry import load_model_trained
from project_logic.predict import predict


app = FastAPI()

app.state.model = load_model_trained()

# # Allow all requests (optional, good for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#WHICH DATA WE EXPECT FROM USER TO SUBMIT, WILL BE ADJUSTED LATER
class RunnerData(BaseModel):
    age: int
    weekly_training_km: float

# Index / Status Route
@app.get("/")
def index():
    return {"status": "Marathon API is running smoothly!"}

# Predict Route (JSON)
@app.post('/predict')
def get_prediction(runner: RunnerData):
    # We convert the JSON data sent by the user into a dictionary
    input_data = runner.dict()

    model = app.state.model
    assert model is not None, "Model couldn't uploaded"

    prediction = predict(model=model, data=input_data)

    return {"predicted_finish_time": prediction}
