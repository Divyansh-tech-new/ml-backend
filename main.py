from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy
import os
from pymongo import MongoClient

# --- NEW: MongoDB Setup ---
# Get the connection string from the environment variable set in Render
MONGO_URI = os.getenv("MONGO_URI")

# Create a MongoDB client instance. Handle the case where the URI is not set.
client = None
if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI)
        # Select your database (it will be created if it doesn't exist)
        db = client.personality_db
        # Select your collection (like a table in SQL)
        collection = db.assessments
        print("✅ Successfully connected to MongoDB.")
    except Exception as e:
        print(f"❌ Could not connect to MongoDB. Error: {e}")
        client = None
else:
    print("⚠️ MONGO_URI environment variable not set. Database functionality will be disabled.")


# Initialize FastAPI app
app = FastAPI()

# CORS Middleware (no changes needed)
origins = ["https://v0-react-frontend-orcin.vercel.app", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Load the model and scaler (no changes needed)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Define input schema for prediction (no changes needed)
class InputData(BaseModel):
    Time_spent_Alone: int
    Stage_fear: int
    Social_event_frequency: int
    Going_out: int
    Drained_after_socializing: int
    Friends_circle_size: int
    Post_frequency: int

# --- NEW: Define schema for data to be saved to MongoDB ---
# It includes all the input data AND the prediction result
class AssessmentData(InputData):
    prediction: int

# Prediction route (no changes needed)
@app.post("/predict")
def predict(data: InputData):
    input_vector = [[
        data.Time_spent_Alone, data.Stage_fear, data.Social_event_frequency,
        data.Going_out, data.Drained_after_socializing, data.Friends_circle_size,
        data.Post_frequency
    ]]
    input_vector_scaled = scaler.transform(input_vector)
    prediction = model.predict(input_vector_scaled)
    result = prediction[0]
    return {"result": int(result)}

# --- NEW: Endpoint to save assessment data ---
@app.post("/save-assessment")
def save_assessment(data: AssessmentData):
    # Check if the database connection was successful
    if not client:
        return {"status": "error", "message": "Database not configured or connection failed."}
    
    try:
        # Convert Pydantic model to a dictionary to insert into MongoDB
        data_dict = data.dict()
        # Insert the document into the collection
        collection.insert_one(data_dict)
        return {"status": "success", "message": "Assessment saved successfully."}
    except Exception as e:
        # Basic error handling
        return {"status": "error", "message": str(e)}
