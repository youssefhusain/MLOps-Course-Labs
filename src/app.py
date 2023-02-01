import os
import joblib
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from typing import List, Dict
import logging
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST



# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load model and columns
script_dir = os.path.dirname(os.path.abspath(__file__))  # src/
model_path = os.path.join(script_dir, "..", "model.pkl")
columns_path = os.path.join(script_dir, "..", "columns.pkl")

try:
    model = joblib.load(model_path)
    columns = joblib.load(columns_path)
    logger.info("Model and columns loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or columns: {e}")
    model = None
    columns = None

# Pydantic model for input validation
class PredictionRequest(BaseModel):
    data: List[Dict[str, float]]

@app.get("/")
async def home():
    return {"message": "Welcome to the Bank Churn Prediction API"}

@app.get("/health")
async def health():
    if model is not None:
        return {"status": "healthy"}
    else:
        return {"status": "unhealthy"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    if model is None or columns is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_df = []
    for record in request.data:
        # Create a row with all columns set to 0 initially
        row = dict.fromkeys(columns, 0)
        # Update with actual features from input
        for k, v in record.items():
            if k in row:
                row[k] = v
            else:
                logger.warning(f"Unknown feature '{k}' in input")
        input_df.append(row)

    import pandas as pd
    input_df = pd.DataFrame(input_df, columns=columns)
    logger.info(f"Input data for prediction: {input_df.to_dict(orient='records')}")

    preds = model.predict(input_df)
    logger.info(f"Predictions: {preds.tolist()}")

    return {"predictions": preds.tolist()}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)