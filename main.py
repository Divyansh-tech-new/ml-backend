from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy
import os
from pymongo import MongoClient

# --- MongoDB Setup ---
# (This part already has good logging, no changes needed here)
MONGO_URI = os.getenv("MONGO_URI")

client = None
if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI)
        db = client.personality_db
        collection = db.assessments
        print("✅ Successfully connected to MongoDB.")
    except Exception as e:
        print(f"❌ Could not connect to MongoDB. Error: {e}")
        client = None
else:
    print("⚠️ MONGO_URI environment variable not set. Database functionality will be disabled.")


# Initialize FastAPI app
app = FastAPI()

# --- CORRECTED: The CORS origins list has been updated ---
origins = [
    "https://v0-react-frontend-orcin.vercel.app",  # Your existing Vercel frontend
    "http://localhost:3000",                       # For local development
    "https://dynamic-bubbles.preview.emergentagent.com"  # New URL added
]

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

# Define schema for data to be saved to MongoDB (no changes needed)
class AssessmentData(InputData):
    prediction: int

# --- PREDICTION ROUTE WITH LOGGING ---
@app.post("/predict")
def predict(data: InputData):
    print("\n--- 🔮 /predict endpoint hit ---")
    print(f"1. Received raw data from frontend: {data.dict()}")

    input_vector = [[
        data.Time_spent_Alone, data.Stage_fear, data.Social_event_frequency,
        data.Going_out, data.Drained_after_socializing, data.Friends_circle_size,
        data.Post_frequency
    ]]
    print(f"2. Created input vector: {input_vector}")

    input_vector_scaled = scaler.transform(input_vector)
    print(f"3. Scaled vector for model: {input_vector_scaled}")

    prediction = model.predict(input_vector_scaled)
    result = int(prediction[0])
    print(f"4. Model prediction: {result}")
    
    print("5. Sending response back to frontend.")
    print("--------------------------------\n")
    return {"result": result}

# --- SAVE ASSESSMENT ROUTE WITH LOGGING ---
@app.post("/save-assessment")
def save_assessment(data: AssessmentData):
    print("\n--- 💾 /save-assessment endpoint hit ---")
    
    if not client:
        print("❌ Database not configured. Aborting save.")
        print("-------------------------------------\n")
        return {"status": "error", "message": "Database not configured or connection failed."}
    
    try:
        data_dict = data.dict()
        print(f"1. Received data to save: {data_dict}")

        collection.insert_one(data_dict)
        print("2. ✅ Successfully inserted data into MongoDB.")
        
        print("3. Sending success response back to frontend.")
        print("------------------------------------------\n")
        return {"status": "success", "message": "Assessment saved successfully."}
    except Exception as e:
        print(f"❌ An error occurred while saving to MongoDB: {e}")
        print("------------------------------------------\n")
        return {"status": "error", "message": str(e)}
