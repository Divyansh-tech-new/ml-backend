from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

class InputData(BaseModel):
    q1: int
    q2: int
    q3: int
    q4: int
    q5: int

@app.post("/predict")
def predict(data: InputData):
    input_vector = [[data.q1, data.q2, data.q3, data.q4, data.q5]]
    prediction = model.predict(input_vector)
    return {"result": prediction[0]}
