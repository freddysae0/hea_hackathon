import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    data = pd.DataFrame({
        'age': np.random.randint(50, 90, n_samples),
        'sex_encoded': np.random.choice([0.0, 1.0], n_samples),
        'bmi': np.random.normal(27, 5, n_samples),
        'systolic_bp': np.random.normal(130, 15, n_samples),
        'diastolic_bp': np.random.normal(80, 10, n_samples),
        'resting_heart_rate': np.random.normal(70, 10, n_samples),
        'chronic_conditions_count': np.random.poisson(1.5, n_samples),
        'recent_weight_change': np.random.normal(0, 2, n_samples),
        'physical_activity': np.random.randint(0, 8, n_samples), # Days per week, wait, agent pre-process output is days per week?
        # Agent uses: activity_days = _to_int(...) and defaults["physical_activity"] = 3.
        # So yes, 0-7.
        'sleep_hours': np.random.normal(7, 1.5, n_samples),
        'stress_level': np.random.randint(0, 11, n_samples),
        'depression_score': np.random.uniform(0, 8, n_samples),
        'smoking_encoded': np.random.choice([0.0, 0.5, 1.0], n_samples),
        'alcohol_encoded': np.random.choice([0.0, 0.3, 0.6, 1.0], n_samples),
    })
    
    # Clip values to reasonable ranges
    data['bmi'] = data['bmi'].clip(15, 50)
    data['systolic_bp'] = data['systolic_bp'].clip(90, 200)
    data['physical_activity'] = data['physical_activity'].clip(0, 7)
    
    # Synthetic Target Generation (Heuristic formula + noise)
    # Similar to agent heuristic but with some noise
    risk = (
        (data['age'] - 50) / 40 * 0.3 + 
        (data['bmi'] - 25) / 15 * 0.2 +
        (data['systolic_bp'] - 120) / 40 * 0.2 +
        (data['chronic_conditions_count'] / 4) * 0.15 +
        (data['smoking_encoded']) * 0.1 +
        np.random.normal(0, 0.1, n_samples)
    )
    data['target'] = np.clip(risk, 0, 1)
    
    return data

def train():
    print("Generating synthetic data...")
    df = generate_synthetic_data(5000)
    X = df.drop(columns=['target'])
    y = df['target']
    
    print("Training LightGBM model...")
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    model_path = 'models/model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Quick verification
    sample = X.iloc[0:1]
    pred = model.predict(sample)[0]
    print(f"Sample prediction: {pred}")

if __name__ == "__main__":
    train()
