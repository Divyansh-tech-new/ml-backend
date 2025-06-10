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

# Load the trained model AND the scaler
model = joblib.load("model.pkl")
# --- ADD THIS LINE ---
scaler = joblib.load("scaler.pkl")

# Define input data schema - This part is correct.
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
    # This input vector correctly matches your training data order.
    input_vector = [[
        data.Time_spent_Alone,
        data.Stage_fear,
        data.Social_event_frequency,
        data.Going_out,
        data.Drained_after_socializing,
        data.Friends_circle_size,
        data.Post_frequency
    ]]
    
    # --- ADD THIS LINE: Apply the scaling transformation ---
    input_vector_scaled = scaler.transform(input_vector)
    
    # --- CHANGE THIS LINE: Predict using the SCALED vector ---
    prediction = model.predict(input_vector_scaled)
    
    # Convert numpy types to plain Python types for JSON serialization
    result = prediction[0]
    if isinstance(result, numpy.integer):
        result = int(result)
        
    return {"result": result}
