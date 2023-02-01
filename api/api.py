from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


model = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: IrisFeatures):
    input_df = pd.DataFrame([data.dict()])

    prediction = model.pridict(input_df)
    
    return {"prediction": prediction[0]}
