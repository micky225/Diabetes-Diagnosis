from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the model
try:
    model = joblib.load("model/RandomForest_best_model.joblib")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

# Initialize FastAPI app
app = FastAPI()

# Define the input data model
class InputData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    SegmentBMI: str  # Assuming this is categorical

# Preprocess SegmentBMI feature
def preprocess_segment_bmi(segment: str):
    if segment == "Low":
        return [1, 0, 0]
    elif segment == "Medium":
        return [0, 1, 0]
    elif segment == "High":
        return [0, 0, 1]
    else:
        raise ValueError("Unknown SegmentBMI category")

# Define the prediction endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    try:
        # Extract the input features from request data
        data = [
            input_data.Pregnancies,
            input_data.Glucose,
            input_data.BloodPressure,
            input_data.SkinThickness,
            input_data.Insulin,
            input_data.BMI,
            input_data.DiabetesPedigreeFunction,
            input_data.Age
        ]

        # Preprocess SegmentBMI
        segment_bmi_encoded = preprocess_segment_bmi(input_data.SegmentBMI)
        data.extend(segment_bmi_encoded)

        # Convert data to a NumPy array
        data_array = np.array(data).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(data_array)
        return {"prediction": int(prediction[0])}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

# Root endpoint for status check
@app.get("/")
async def root():
    return {"message": "Model API is up and running!"}
