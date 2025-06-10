from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for frontend
origins = [
    "https://v0-react-frontend-orcin.vercel.app",  # your deployed frontend
    "http://localhost:3000"  # for local dev
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("model.pkl")

# Define input data schema
class InputData(BaseModel):
    Going_outside: int
    Time_spent_Alone: int
    Stage_fear: int
    Drained_after_socializing: int
    Reading_books: int
    Talkativeness: int
    Energy_level: int

# Define prediction route
@app.post("/predict")
def predict(data: InputData):
    input_vector = [[
        data.Going_outside,
        data.Time_spent_Alone,
        data.Stage_fear,
        data.Drained_after_socializing,
        data.Reading_books,
        data.Talkativeness,
        data.Energy_level
    ]]
    prediction = model.predict(input_vector)
    # Convert numpy.int64 to plain Python int
    return {"result": int(prediction[0])}
