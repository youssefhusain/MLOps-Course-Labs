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


#model = joblib.load("model.pkl")
import random
@app.post("/predict")
def predict(data: IrisFeatures):
    input_df = pd.DataFrame([data.dict()])

    prediction = random.choice([1, 2, 3])
    
    return {"prediction": prediction[0]}
