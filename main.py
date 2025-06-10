from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# CORS configuration
origins = [
    "https://v0-react-frontend-orcin.vercel.app",  # your Vercel frontend
    "http://localhost:3000"  # optional for local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # only allow listed origins
    allow_credentials=True,
    allow_methods=["*"],    # allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # allow all headers
)

# Load model
model = joblib.load("model.pkl")

# Define input schema
class InputData(BaseModel):
    q1: int
    q2: int
    q3: int
    q4: int
    q5: int

# Define prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    input_vector = [[data.q1, data.q2, data.q3, data.q4, data.q5]]
    prediction = model.predict(input_vector)
    return {"result": prediction[0]}
