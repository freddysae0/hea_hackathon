from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.agents import HealthRiskAgent

app = FastAPI(title="Hea Hackathon API", description="API for predicting health risks based on longitudinal data.")

# --- Pydantic Models ---

class PredictionFactor(BaseModel):
    name: str
    impact: float

class PredictionResult(BaseModel):
    risk_score: float
    risk_level: str
    top_factors: List[PredictionFactor]
    trend_signal: str

class HealthFormValues(BaseModel):
    age: str
    sex: str
    education_level: str
    employment_status: str
    marital_status: str
    bmi: str
    systolic_bp: str
    diastolic_bp: str
    resting_heart_rate: str
    chronic_conditions_count: str
    recent_weight_change: str
    physical_activity_days_per_week: str
    sleep_hours_avg: str
    smoking_status: str
    alcohol_frequency: str
    stress_level: int 
    depressive_symptoms_score: str

# --- Agent Initialization ---
# In a real app, you might want to load the model once at startup
agent = HealthRiskAgent()

# --- Endpoints ---

@app.post("/predict", response_model=PredictionResult)
async def predict_risk(data: HealthFormValues):
    try:
        result = agent.predict(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}
