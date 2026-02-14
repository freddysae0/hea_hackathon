import pandas as pd
import numpy as np
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthRiskAgent:
    def __init__(self, model_path: str = "models/model.pkl"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                logger.info(f"Loading model from {self.model_path}")
                return joblib.load(self.model_path)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return None
        else:
            logger.warning(f"Model not found at {self.model_path}. Using fallback logic.")
            return None

    def _preprocess(self, data) -> pd.DataFrame:
        """
        Converts HealthFormValues object to a DataFrame compatible with the model.
        """
        # Convert Pydantic model to dict
        data_dict = data.dict()
        
        # Feature Engineering / Transformation
        # We need to ensure these keys match what the model was trained on
        # For now, we manually map them to expected numeric types
        
        # Helper for safer conversion
        def to_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        def to_int(val):
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0

        features = {
            'age': to_int(data_dict.get('age')),
            'sex_encoded': 1 if data_dict.get('sex', '').lower() == 'male' else 0,
            'bmi': to_float(data_dict.get('bmi')),
            'systolic_bp': to_float(data_dict.get('systolic_bp')),
            'diastolic_bp': to_float(data_dict.get('diastolic_bp')),
            'resting_heart_rate': to_float(data_dict.get('resting_heart_rate')),
            'chronic_conditions_count': to_int(data_dict.get('chronic_conditions_count')),
            'physical_activity': to_int(data_dict.get('physical_activity_days_per_week')),
            'sleep_hours': to_float(data_dict.get('sleep_hours_avg')),
            'stress_level': to_int(data_dict.get('stress_level')),
            'depression_score': to_float(data_dict.get('depressive_symptoms_score')),
            # Add one-hot encoding or other transformations as needed
            # For brevity, mapped categorical fields are skipped or need encoding
        }
        
        return pd.DataFrame([features])

    def _explain_risk(self, features: pd.DataFrame, risk_score: float) -> list:
        """
        Generates explanation factors. In a real scenario, this would use SHAP.
        For now, it uses simple heuristic rules.
        """
        factors = []
        row = features.iloc[0]
        
        # Simple heuristics for "top_factors"
        if row['physical_activity'] < 3:
            factors.append({"name": "Low physical activity", "impact": 0.15})
        if row['sleep_hours'] < 6:
            factors.append({"name": "Sleep disruption", "impact": 0.10})
        if row['systolic_bp'] > 130:
            factors.append({"name": "Elevated blood pressure", "impact": 0.20})
        if row['stress_level'] > 7:
            factors.append({"name": "High stress", "impact": 0.12})
        if row['bmi'] > 30:
             factors.append({"name": "High BMI", "impact": 0.18})
             
        # Sort by impact
        factors.sort(key=lambda x: x['impact'], reverse=True)
        return factors[:3]

    def predict(self, data) -> dict:
        """
        Receives data (HealthFormValues) and returns a PredictionResult dict.
        """
        features_df = self._preprocess(data)
        
        # --- MODEL INFERENCE SECTION ---
        # TODO: FUTURE INTEGRATION
        # When you have a trained model, ensure it is saved to 'self.model_path' (e.g., models/model.pkl).
        # The code below will automatically use it if found.
        # Ensure the model accepts the DataFrame structure created in '_preprocess'.
        
        if self.model:
            try:
                # Assuming model has predict_proba
                risk_score = float(self.model.predict_proba(features_df)[:, 1][0])
                logger.info("Used loaded model for prediction.")
            except Exception as e:
                # Fallback if model doesn't support proba or fails
                logger.warning(f"Model prediction failed ({e}). Using simplistic fallback.")
                risk_score = float(self.model.predict(features_df)[0])
        else:
            # --- MOCK / HEURISTIC FALLBACK ---
            # This section runs when no model is found.
            # TODO: Replace this with your actual model logic or keep as a failsafe.
            
            logger.info("No model found. Using heuristic fallback logic.")
            
            # Base risk
            risk_score = 0.1
            row = features_df.iloc[0]
            
            # Simple heuristic rules to simulate a model
            if row['age'] > 50:
                risk_score += (row['age'] - 50) * 0.015
            if row['chronic_conditions_count'] > 0:
                risk_score += row['chronic_conditions_count'] * 0.1
            if row['systolic_bp'] > 130:
                risk_score += 0.15
            if row['bmi'] > 30:
                risk_score += 0.1
            
            # Cap at 0.99
            risk_score = min(0.99, max(0.01, risk_score))

        # --- EXPLAINABILITY SECTION ---
        top_factors = self._explain_risk(features_df, risk_score)
        
        return {
            "risk_score": round(risk_score, 2),
            "risk_level": "High" if risk_score > 0.5 else "Low",  # Threshold could be dynamic
            "top_factors": top_factors,
            "trend_signal": "Stable" # Placeholder, implies need for historical data
        }
