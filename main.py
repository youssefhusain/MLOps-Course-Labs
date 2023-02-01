from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import logging
import os


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
    logger.info("Home endpoint hit")
    return {"message": "Welcome to the ML prediction API"}


@app.get("/health")
def health_check():
    status = model is not None
    logger.info("Health check: %s", status)
    return {"status": "ok" if status else "model not loaded"}


@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        logger.error("Prediction attempted but model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")

    geography_map = {"France": 0, "Spain": 1, "Germany": 2}
    gender_map = {"Male": 0, "Female": 1}

    try:
        geo = geography_map[data.Geography]
        gender = gender_map[data.Gender]
    except KeyError as e:
        logger.error("Invalid input: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid category: {e}")

    features = [[
        data.CreditScore, geo, gender, data.Age, data.Tenure, data.Balance,
        data.NumOfProducts, data.HasCrCard, data.IsActiveMember, data.EstimatedSalary
    ]]

    logger.info("Received prediction request with data: %s", features)
    prediction = model.predict(features)
    logger.info("Prediction result: %s", prediction)

    return {"prediction": int(prediction[0])}