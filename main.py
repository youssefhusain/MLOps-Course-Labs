from fastapi import FastAPI
from pydantic import BaseModel
from model import predict

app = FastAPI()

class inputdata(BaseModel):
    f1: float
    f2: float
    f3: float
    f4: float

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/predict")
def predict_species(input_data: inputdata):
    return predict(input_data.dict())
