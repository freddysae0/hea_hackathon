# AI Agents Manifest

## Health Risk Prediction Agent

### Description
The **Health Risk Prediction Agent** is a specialized AI agent designed to analyze longitudinal self-reported health data and identify early signals of health risks. It serves as the core intelligence engine for the "Common Table" platform, providing personalized risk assessments and actionable insights.

### Capabilities
1.  **Risk Scoring**: Calculates a probabilistic risk score (0-1) for developing specific health conditions based on a variety of health determinants.
2.  **Explainability**: Identifies the top contributing factors (e.g., "Sleep disruption", "Elevated blood pressure") to the calculated risk, offering transparency into the model's reasoning.
3.  **Trend Analysis**: (Planned) Detects changes in risk trajectory over time (e.g., "Increasing risk over last 2 periods").

### input Schema
The agent accepts a JSON payload representing a user's current health snapshot:

```json
{
  "age": "65",
  "sex": "Male",
  "education_level": "College",
  "employment_status": "Retired",
  "marital_status": "Married",
  "bmi": "28.5",
  "systolic_bp": "145",
  "diastolic_bp": "90",
  "resting_heart_rate": "75",
  "chronic_conditions_count": "2",
  "recent_weight_change": "-1.5",
  "physical_activity_days_per_week": "1",
  "sleep_hours_avg": "5.5",
  "smoking_status": "Former",
  "alcohol_frequency": "Weekly",
  "stress_level": 8,
  "depressive_symptoms_score": "4.2"
}
```

### Output Schema
The agent returns a structured prediction result:

```json
{
  "risk_score": 0.72,
  "risk_level": "High",
  "top_factors": [
    { "name": "Low physical activity", "impact": 0.31 },
    { "name": "Sleep disruption", "impact": 0.22 },
    { "name": "Elevated blood pressure", "impact": 0.19 }
  ],
  "trend_signal": "Increasing risk over last 2 periods"
}
```

### Underlying Technology
- **Model Architecture**: LightGBM (Gradient Boosting Machine)
- **Training Data**: Synthetic longitudinal data modeled after the RAND HRS (Health and Retirement Study) dataset.
- **Explainability**: Custom feature importance extraction (simulating SHAP-like values for the hackathon prototype).
- **Implementation**: Python (`src/agents.py`), exposed via FastAPI (`src/api.py`).

### Integration
The agent is containerized and accessible via a RESTful API endpoint: `POST /predict`.

### Future Integration (Model Connection)
Currently, the agent supports both a specific LightGBM model (`models/model.pkl`) and a heuristic fallback mode if no model is found.

To connect a new or improved model:
1.  **Train your model** ensuring it accepts the features defined in `src/agents.py::_preprocess`.
    *   Expected features: `age`, `sex_encoded`, `bmi`, `systolic_bp`, etc.
2.  **Save the model** as a pickle file: `joblib.dump(model, 'models/model.pkl')`.
3.  **Restart the API**. The `HealthRiskAgent` will automatically detect and load `models/model.pkl`.
4.  **Verification**: Check logs for "Loading model from models/model.pkl" to confirm successful loading.

If `models/model.pkl` is missing, the agent gracefully downgrades to "Mock/Heuristic Mode" using rule-based logic to ensure the API remains functional for frontend development.
