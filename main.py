from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging
import os
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model_path = "model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
    logger.info("Model loaded successfully from %s", model_path)
else:
    logger.error("Model file not found!")
    model = None


app = FastAPI()


class CustomerData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float

@app.get("/")
def home():
    return {"message": "Welcome to the Churn Prediction API. Use /docs for Swagger UI."}


@app.get("/health")
def health():
    return {"status": "Healthy"}


@app.post("/predict")
def predict(input_data: CustomerData):
    try:
        logger.info("Received prediction request.")
        
  
        input_array = np.array([
            input_data.CreditScore,
            input_data.Geography == "France",
            input_data.Geography == "Spain",
            input_data.Gender == "Male",
            input_data.Age,
            input_data.Tenure,
            input_data.Balance,
            input_data.NumOfProducts,
            input_data.HasCrCard,
            input_data.IsActiveMember,
            input_data.EstimatedSalary,
        ]).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0, 1]

        result = {
            "prediction": int(prediction),
            "probability_of_churn": round(probability, 4),
        }

        logger.info(f"Prediction result: {result}")
        return result

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}
