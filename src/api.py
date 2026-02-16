from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents import HealthRiskAgent

app = FastAPI(
    title="Hea Hackathon API",
    description="API for early health-risk prediction from self-reported inputs.",
    version="0.1.0",
)

# Frontend integration: explicit origins for production safety
origins = [
    "http://localhost:3000",
    "http://localhost:8001",
    "http://localhost:5173",
    "https://altair-hackathon.web.app",
    "https://altair-hackathon.firebaseapp.com",
    "https://back.redblock.online",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionFactor(BaseModel):
    name: str
    impact: float
    direction: Optional[str] = None
    value: Optional[float] = None


class DiseaseRisk(BaseModel):
    disease: str
    risk_probability: float
    threshold: float
    is_high_risk: bool
    top_factors: List[PredictionFactor]


class PredictionResult(BaseModel):
    model_config = {"protected_namespaces": ()}

    risk_score: float
    risk_level: str
    top_factors: List[PredictionFactor]
    trend_signal: str
    data_quality_score: float
    disease_risks: List[DiseaseRisk]
    recommendations: List[str]
    missing_fields: List[str]
    model_version: str


class HealthFormValues(BaseModel):
    age: str
    sex: str
    age: str
    sex: str
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
    stress_level: Union[int, str] = Field(..., description="0-10 user-reported stress")
    depressive_symptoms_score: str


agent = HealthRiskAgent()


@app.post("/predict", response_model=PredictionResult)
async def predict_risk(data: HealthFormValues) -> PredictionResult:
    try:
        result = agent.predict(data)
        return PredictionResult(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


@app.get("/metadata")
async def metadata() -> dict:
    return agent.metadata()
