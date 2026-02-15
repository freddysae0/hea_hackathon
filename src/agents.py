import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HealthRiskAgent:
    """Prediction agent with model-first inference and heuristic fallback."""

    DISEASES = ["hibpe", "diabe", "hearte", "stroke", "arthre"]

    DISEASE_THRESHOLDS = {
        "hibpe": 0.0055,
        "diabe": 0.0220,
        "hearte": 0.0690,
        "stroke": 0.0355,
        "arthre": 0.0375,
    }

    def __init__(
        self,
        model_path: str = "models/model.pkl",
        model_meta_path: str = "models/model_meta.pkl",
    ) -> None:
        self.model_path = model_path
        self.model_meta_path = model_meta_path
        self.model = self._load_model(model_path)
        self.model_meta = self._load_model_meta(model_meta_path)
        self.model_threshold = self._extract_model_threshold(self.model_meta)

    def _load_model(self, model_path: str) -> Optional[Any]:
        if not os.path.exists(model_path):
            logger.warning("Model not found at %s. Using heuristic fallback.", model_path)
            return None

        try:
            logger.info("Loading model from %s", model_path)
            return joblib.load(model_path)
        except Exception as exc:
            logger.error("Failed to load model from %s: %s", model_path, exc)
            return None

    def _load_model_meta(self, model_meta_path: str) -> Dict[str, Any]:
        if not os.path.exists(model_meta_path):
            logger.warning("Model metadata not found at %s. Using default threshold.", model_meta_path)
            return {}

        try:
            meta = joblib.load(model_meta_path)
            if isinstance(meta, dict):
                return meta
            logger.warning("Model metadata at %s is not a dict. Ignoring.", model_meta_path)
            return {}
        except Exception as exc:
            logger.error("Failed to load model metadata from %s: %s", model_meta_path, exc)
            return {}

    @staticmethod
    def _extract_model_threshold(meta: Dict[str, Any]) -> float:
        if not isinstance(meta, dict):
            return 0.33

        value = meta.get("best_threshold_f2", 0.33)
        try:
            return float(np.clip(float(value), 0.01, 0.99))
        except (TypeError, ValueError):
            return 0.33

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_get(data_dict: Dict[str, Any], key: str) -> Any:
        return data_dict.get(key, None)

    def _normalize_smoking(self, raw: Any) -> Optional[float]:
        if raw is None:
            return None
        value = str(raw).strip().lower()
        mapping = {
            "never": 0.0,
            "no": 0.0,
            "former": 0.5,
            "past": 0.5,
            "current": 1.0,
            "yes": 1.0,
            "daily": 1.0,
        }
        return mapping.get(value, None)

    def _normalize_alcohol(self, raw: Any) -> Optional[float]:
        if raw is None:
            return None
        value = str(raw).strip().lower()
        mapping = {
            "never": 0.0,
            "none": 0.0,
            "monthly": 0.3,
            "occasionally": 0.3,
            "weekly": 0.6,
            "daily": 1.0,
            "often": 0.8,
        }
        return mapping.get(value, None)

    def _normalize_sex(self, raw: Any) -> Optional[float]:
        if raw is None:
            return None
        value = str(raw).strip().lower()
        if value in {"male", "m", "man"}:
            return 1.0
        if value in {"female", "f", "woman"}:
            return 0.0
        return None

    def _preprocess(self, data: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if hasattr(data, "model_dump"):
            data_dict = data.model_dump()
        elif hasattr(data, "dict"):
            data_dict = data.dict()
        elif isinstance(data, dict):
            data_dict = data
        else:
            raise TypeError("Unsupported input type for preprocessing")

        raw_fields = [
            "age",
            "sex",
            "bmi",
            "systolic_bp",
            "diastolic_bp",
            "resting_heart_rate",
            "chronic_conditions_count",
            "recent_weight_change",
            "physical_activity_days_per_week",
            "sleep_hours_avg",
            "smoking_status",
            "alcohol_frequency",
            "stress_level",
            "depressive_symptoms_score",
            "education_level",
            "employment_status",
            "marital_status",
        ]

        missing_fields: List[str] = []
        for field in raw_fields:
            raw_val = data_dict.get(field, None)
            if raw_val is None or (isinstance(raw_val, str) and raw_val.strip() == ""):
                missing_fields.append(field)

        age = self._to_int(self._safe_get(data_dict, "age"))
        sex_encoded = self._normalize_sex(self._safe_get(data_dict, "sex"))
        bmi = self._to_float(self._safe_get(data_dict, "bmi"))
        systolic_bp = self._to_float(self._safe_get(data_dict, "systolic_bp"))
        diastolic_bp = self._to_float(self._safe_get(data_dict, "diastolic_bp"))
        resting_hr = self._to_float(self._safe_get(data_dict, "resting_heart_rate"))
        chronic_count = self._to_int(self._safe_get(data_dict, "chronic_conditions_count"))
        weight_change = self._to_float(self._safe_get(data_dict, "recent_weight_change"))
        activity_days = self._to_int(self._safe_get(data_dict, "physical_activity_days_per_week"))
        sleep_hours = self._to_float(self._safe_get(data_dict, "sleep_hours_avg"))
        stress_level = self._to_int(self._safe_get(data_dict, "stress_level"))
        depression_score = self._to_float(self._safe_get(data_dict, "depressive_symptoms_score"))
        smoking_encoded = self._normalize_smoking(self._safe_get(data_dict, "smoking_status"))
        alcohol_encoded = self._normalize_alcohol(self._safe_get(data_dict, "alcohol_frequency"))

        defaults = {
            "age": 60,
            "sex_encoded": 0.0,
            "bmi": 27.0,
            "systolic_bp": 125.0,
            "diastolic_bp": 80.0,
            "resting_heart_rate": 72.0,
            "chronic_conditions_count": 0,
            "recent_weight_change": 0.0,
            "physical_activity": 3,
            "sleep_hours": 7.0,
            "stress_level": 5,
            "depression_score": 2.0,
            "smoking_encoded": 0.0,
            "alcohol_encoded": 0.3,
        }

        features = {
            "age": defaults["age"] if age is None else age,
            "sex_encoded": defaults["sex_encoded"] if sex_encoded is None else sex_encoded,
            "bmi": defaults["bmi"] if bmi is None else bmi,
            "systolic_bp": defaults["systolic_bp"] if systolic_bp is None else systolic_bp,
            "diastolic_bp": defaults["diastolic_bp"] if diastolic_bp is None else diastolic_bp,
            "resting_heart_rate": defaults["resting_heart_rate"] if resting_hr is None else resting_hr,
            "chronic_conditions_count": defaults["chronic_conditions_count"] if chronic_count is None else chronic_count,
            "recent_weight_change": defaults["recent_weight_change"] if weight_change is None else weight_change,
            "physical_activity": defaults["physical_activity"] if activity_days is None else activity_days,
            "sleep_hours": defaults["sleep_hours"] if sleep_hours is None else sleep_hours,
            "stress_level": defaults["stress_level"] if stress_level is None else stress_level,
            "depression_score": defaults["depression_score"] if depression_score is None else depression_score,
            "smoking_encoded": defaults["smoking_encoded"] if smoking_encoded is None else smoking_encoded,
            "alcohol_encoded": defaults["alcohol_encoded"] if alcohol_encoded is None else alcohol_encoded,
        }

        quality_score = max(0.0, 1.0 - (len(missing_fields) / max(len(raw_fields), 1)))

        metadata = {
            "missing_fields": missing_fields,
            "data_quality_score": round(float(quality_score), 3),
        }
        return pd.DataFrame([features]), metadata

    @staticmethod
    def _clip_probability(value: float) -> float:
        return float(np.clip(value, 0.01, 0.99))

    def _heuristic_disease_probs(self, row: pd.Series) -> Dict[str, float]:
        age_risk = np.clip((row["age"] - 50.0) / 35.0, 0.0, 1.5)
        bmi_risk = np.clip((row["bmi"] - 25.0) / 15.0, 0.0, 1.5)
        sbp_risk = np.clip((row["systolic_bp"] - 120.0) / 40.0, 0.0, 1.5)
        dbp_risk = np.clip((row["diastolic_bp"] - 80.0) / 20.0, 0.0, 1.5)
        bp_risk = np.clip(0.7 * sbp_risk + 0.3 * dbp_risk, 0.0, 1.5)
        activity_risk = np.clip((3.0 - row["physical_activity"]) / 3.0, 0.0, 1.5)
        sleep_risk = np.clip((7.0 - row["sleep_hours"]) / 3.0, 0.0, 1.5)
        stress_risk = np.clip(row["stress_level"] / 10.0, 0.0, 1.5)
        dep_risk = np.clip(row["depression_score"] / 8.0, 0.0, 1.5)
        chronic_risk = np.clip(row["chronic_conditions_count"] / 4.0, 0.0, 1.5)
        smoke_risk = np.clip(row["smoking_encoded"], 0.0, 1.0)
        alcohol_risk = np.clip(row["alcohol_encoded"], 0.0, 1.0)

        probs = {
            "hibpe": 0.05 + 0.18 * age_risk + 0.22 * bp_risk + 0.10 * bmi_risk + 0.08 * chronic_risk + 0.05 * stress_risk,
            "diabe": 0.04 + 0.14 * age_risk + 0.20 * bmi_risk + 0.10 * activity_risk + 0.07 * sleep_risk + 0.10 * chronic_risk,
            "hearte": 0.03 + 0.17 * age_risk + 0.18 * bp_risk + 0.08 * bmi_risk + 0.10 * smoke_risk + 0.09 * chronic_risk,
            "stroke": 0.02 + 0.16 * age_risk + 0.18 * bp_risk + 0.07 * smoke_risk + 0.08 * chronic_risk + 0.06 * stress_risk,
            "arthre": 0.05 + 0.16 * age_risk + 0.10 * bmi_risk + 0.06 * chronic_risk + 0.05 * activity_risk + 0.03 * dep_risk,
        }

        # Light penalty to reflect unhealthy alcohol profile.
        for disease in probs:
            probs[disease] = self._clip_probability(probs[disease] + 0.03 * alcohol_risk)

        return probs

    def _overall_from_disease_probs(self, disease_probs: Dict[str, float]) -> float:
        values = np.array(list(disease_probs.values()), dtype=float)
        blended = 0.60 * float(values.max()) + 0.40 * float(values.mean())
        return self._clip_probability(blended)

    def _risk_level(self, risk_score: float) -> str:
        if self.model is not None:
            high_cut = float(self.model_threshold)
            medium_cut = max(0.10, high_cut * 0.60)
            if risk_score >= high_cut:
                return "High"
            if risk_score >= medium_cut:
                return "Medium"
            return "Low"

        if risk_score >= 0.66:
            return "High"
        if risk_score >= 0.33:
            return "Medium"
        return "Low"

    def _factor_signals(self, row: pd.Series) -> Dict[str, Tuple[str, float, str, float]]:
        signals = {
            "low_activity": (
                "Low physical activity",
                np.clip((3.0 - row["physical_activity"]) / 3.0, 0.0, 1.5),
                "up",
                float(row["physical_activity"]),
            ),
            "sleep_disruption": (
                "Sleep disruption",
                np.clip((7.0 - row["sleep_hours"]) / 3.0, 0.0, 1.5),
                "up",
                float(row["sleep_hours"]),
            ),
            "elevated_bp": (
                "Elevated blood pressure",
                np.clip((row["systolic_bp"] - 120.0) / 40.0, 0.0, 1.5),
                "up",
                float(row["systolic_bp"]),
            ),
            "high_bmi": (
                "High BMI",
                np.clip((row["bmi"] - 25.0) / 15.0, 0.0, 1.5),
                "up",
                float(row["bmi"]),
            ),
            "high_stress": (
                "High stress",
                np.clip(row["stress_level"] / 10.0, 0.0, 1.5),
                "up",
                float(row["stress_level"]),
            ),
            "depressive_symptoms": (
                "Depressive symptoms",
                np.clip(row["depression_score"] / 8.0, 0.0, 1.5),
                "up",
                float(row["depression_score"]),
            ),
            "chronic_burden": (
                "Chronic condition burden",
                np.clip(row["chronic_conditions_count"] / 4.0, 0.0, 1.5),
                "up",
                float(row["chronic_conditions_count"]),
            ),
            "smoking": (
                "Smoking profile",
                np.clip(row["smoking_encoded"], 0.0, 1.0),
                "up",
                float(row["smoking_encoded"]),
            ),
        }
        return signals

    def _top_factors_for_weights(
        self,
        row: pd.Series,
        weights: Dict[str, float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        signals = self._factor_signals(row)
        scored = []
        for key, weight in weights.items():
            if key not in signals:
                continue
            name, signal, direction, value = signals[key]
            impact = float(weight * signal)
            if impact <= 0:
                continue
            scored.append(
                {
                    "name": name,
                    "impact": round(impact, 3),
                    "direction": direction,
                    "value": value,
                }
            )
        scored.sort(key=lambda item: item["impact"], reverse=True)
        return scored[:top_k]

    def _disease_factor_weights(self) -> Dict[str, Dict[str, float]]:
        return {
            "hibpe": {
                "elevated_bp": 0.35,
                "high_bmi": 0.22,
                "chronic_burden": 0.16,
                "high_stress": 0.12,
                "low_activity": 0.10,
                "smoking": 0.08,
            },
            "diabe": {
                "high_bmi": 0.34,
                "low_activity": 0.24,
                "sleep_disruption": 0.16,
                "chronic_burden": 0.16,
                "high_stress": 0.10,
                "depressive_symptoms": 0.10,
            },
            "hearte": {
                "elevated_bp": 0.30,
                "high_bmi": 0.18,
                "chronic_burden": 0.18,
                "smoking": 0.16,
                "high_stress": 0.12,
                "low_activity": 0.08,
            },
            "stroke": {
                "elevated_bp": 0.34,
                "chronic_burden": 0.18,
                "smoking": 0.16,
                "high_stress": 0.14,
                "sleep_disruption": 0.10,
                "low_activity": 0.08,
            },
            "arthre": {
                "high_bmi": 0.24,
                "low_activity": 0.22,
                "chronic_burden": 0.18,
                "sleep_disruption": 0.12,
                "depressive_symptoms": 0.12,
                "high_stress": 0.08,
            },
        }

    def _recommendations_from_factors(self, top_factors: List[Dict[str, Any]]) -> List[str]:
        recs = []
        names = {item["name"] for item in top_factors}

        if "Low physical activity" in names:
            recs.append("Increase activity gradually (e.g., 20-30 min walk, 5 days/week).")
        if "Sleep disruption" in names:
            recs.append("Stabilize sleep routine and aim for 7-8 hours of sleep.")
        if "Elevated blood pressure" in names:
            recs.append("Track blood pressure weekly and consult a clinician if persistently elevated.")
        if "High BMI" in names:
            recs.append("Start a weight management plan with nutrition and movement goals.")
        if "High stress" in names:
            recs.append("Use daily stress-reduction practices and monitor stress trends.")
        if "Depressive symptoms" in names:
            recs.append("Consider mental health follow-up and routine symptom check-ins.")

        if not recs:
            recs.append("Keep current healthy habits and continue periodic check-ins.")

        return recs[:4]

    def _predict_with_loaded_model(self, features: pd.DataFrame) -> Optional[float]:
        if self.model is None:
            return None

        try:
            if hasattr(self.model, "predict_proba"):
                proba = float(self.model.predict_proba(features)[:, 1][0])
                return self._clip_probability(proba)
            if hasattr(self.model, "predict"):
                pred = float(self.model.predict(features)[0])
                if pred > 1.0:
                    pred = pred / 100.0
                return self._clip_probability(pred)
        except Exception as exc:
            logger.warning("Loaded model inference failed: %s", exc)
            return None

        return None

    def predict(self, data: Any) -> Dict[str, Any]:
        features_df, metadata = self._preprocess(data)
        row = features_df.iloc[0]

        disease_probs = self._heuristic_disease_probs(row)
        heuristic_overall = self._overall_from_disease_probs(disease_probs)

        model_overall = self._predict_with_loaded_model(features_df)
        risk_score = heuristic_overall if model_overall is None else model_overall

        factor_weights = self._disease_factor_weights()
        top_factors = self._top_factors_for_weights(
            row,
            {
                "elevated_bp": 0.30,
                "high_bmi": 0.20,
                "low_activity": 0.18,
                "sleep_disruption": 0.14,
                "high_stress": 0.10,
                "chronic_burden": 0.12,
                "depressive_symptoms": 0.08,
                "smoking": 0.08,
            },
            top_k=4,
        )

        disease_risks: List[Dict[str, Any]] = []
        for disease in self.DISEASES:
            prob = float(disease_probs[disease])
            threshold = float(self.DISEASE_THRESHOLDS[disease])
            disease_risks.append(
                {
                    "disease": disease,
                    "risk_probability": round(prob, 4),
                    "threshold": round(threshold, 4),
                    "is_high_risk": bool(prob >= threshold),
                    "top_factors": self._top_factors_for_weights(
                        row,
                        factor_weights[disease],
                        top_k=3,
                    ),
                }
            )

        disease_risks.sort(key=lambda item: item["risk_probability"], reverse=True)
        recommendations = self._recommendations_from_factors(top_factors)

        result = {
            "risk_score": round(float(risk_score), 4),
            "risk_level": self._risk_level(risk_score),
            "top_factors": top_factors,
            "trend_signal": "Stable (single-snapshot estimate)",
            "data_quality_score": metadata["data_quality_score"],
            "disease_risks": disease_risks,
            "recommendations": recommendations,
            "missing_fields": metadata["missing_fields"],
            "model_version": (
                "model.pkl+meta"
                if self.model is not None and bool(self.model_meta)
                else ("model.pkl" if self.model is not None else "heuristic-v1")
            ),
        }

        return result

    def metadata(self) -> Dict[str, Any]:
        return {
            "diseases": self.DISEASES,
            "thresholds": self.DISEASE_THRESHOLDS,
            "model_path": self.model_path,
            "model_loaded": self.model is not None,
            "model_meta_path": self.model_meta_path,
            "model_meta_loaded": bool(self.model_meta),
            "model_threshold": float(self.model_threshold),
            "training_metrics": self.model_meta.get("metrics", {}) if isinstance(self.model_meta, dict) else {},
        }
