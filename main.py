from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

origins = [
    "https://v0-react-frontend-orcin.vercel.app",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")

class InputData(BaseModel):
    Time_spent_Alone: int
    Stage_fear: int
    Social_events: int
    Going_outside: int
    Drained_after_socializing: int
    Friends_circle: int
    Post_frequency: int

@app.post("/predict")
def predict(data: InputData):
    input_vector = [[
        data.Time_spent_Alone,
        data.Stage_fear,
        data.Social_events,
        data.Going_outside,
        data.Drained_after_socializing,
        data.Friends_circle,
        data.Post_frequency
    ]]
    prediction = model.predict(input_vector)
    return {"result": int(prediction[0])}
