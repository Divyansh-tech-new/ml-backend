from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy

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
# Ensure this model was trained with the new feature set in the correct order
model = joblib.load("model.pkl")

# Define input data schema based on the fields from the image
class InputData(BaseModel):
    Time_spent_Alone: int
    Stage_fear: int
    Social_event_frequency: int
    Going_out: int
    Drained_after_socializing: int
    Friends_circle_size: int
    Post_frequency: int

# Define prediction route
@app.post("/predict")
def predict(data: InputData):
    # The order of features here MUST match the order used to train 'model.pkl'
    input_vector = [[
        data.Time_spent_Alone,
        data.Stage_fear,
        data.Social_event_frequency,
        data.Going_out,
        data.Drained_after_socializing,
        data.Friends_circle_size,
        data.Post_frequency
    ]]
    
    prediction = model.predict(input_vector)
    
    # Convert numpy types to plain Python types for JSON serialization
    result = prediction[0]
    if isinstance(result, numpy.integer):
        result = int(result)
        
    return {"result": result}
