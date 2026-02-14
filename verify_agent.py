from src.agents import HealthRiskAgent
from src.api import HealthFormValues

# Sample Data
data = HealthFormValues(
    age="65",
    sex="Male",
    bmi="28.5",
    systolic_bp="145",
    diastolic_bp="90",
    resting_heart_rate="75",
    chronic_conditions_count="2",
    recent_weight_change="-1.5",
    physical_activity_days_per_week="1",
    sleep_hours_avg="5.5",
    smoking_status="Former",
    alcohol_frequency="Weekly",
    stress_level=8,
    depressive_symptoms_score="4.2",
    education_level="College",
    employment_status="Retired",
    marital_status="Married"
)

agent = HealthRiskAgent()
result = agent.predict(data)

print("\n--- Prediction Result ---")
print(result)

# Assertions
assert "risk_score" in result
assert "risk_level" in result
assert "top_factors" in result
assert len(result["top_factors"]) > 0

print("\nVerification Successful!")
