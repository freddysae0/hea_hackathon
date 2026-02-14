import pandas as pd
import numpy as np
import argparse
import os

def generate_synthetic_data(output_path, n_people=1000, n_waves=5):
    print(f"Generating synthetic data for {n_people} people over {n_waves} waves...")
    np.random.seed(42)
    
    records = []
    for person_id in range(n_people):
        # Static traits
        age_base = np.random.randint(50, 80)
        sex = np.random.choice(['Male', 'Female'])
        base_health = np.random.rand() # 0 = healthy, 1 = sick prone
        
        for wave in range(n_waves):
            year = 2010 + (wave * 2)
            age = age_base + (wave * 2)
            
            # Time-varying features
            # Correlate these with base_health (higher base_health -> worse metrics)
            bmi = np.random.normal(25 + base_health * 5, 4)
            systolic_bp = np.random.normal(120 + base_health * 20, 15)
            diastolic_bp = np.random.normal(80 + base_health * 10, 10)
            resting_hr = np.random.normal(70 + base_health * 10, 8)
            chronic_conds = np.random.poisson(base_health * 2)
            phys_act = max(0, min(7, int(np.random.normal(3 - base_health * 2, 2))))
            sleep_hours = max(4, min(10, np.random.normal(7 - base_health, 1.5)))
            stress = max(1, min(10, int(np.random.normal(5 + base_health * 3, 2))))
            depression = max(0, np.random.normal(base_health * 5, 3))

            # Target: High Risk of "Event" (e.g. hospitalization or diagnosis)
            # Logit model
            logit = -5 + (age-60)*0.05 + (bmi-25)*0.1 + (systolic_bp-120)*0.05 - phys_act*0.3 + stress*0.1
            prob_risk = 1 / (1 + np.exp(-logit))
            is_high_risk = 1 if np.random.rand() < prob_risk else 0
            
            records.append({
                'person_id': person_id,
                'year': year,
                'age': age,
                'sex': sex,
                'bmi': round(bmi, 1),
                'systolic_bp': int(systolic_bp),
                'diastolic_bp': int(diastolic_bp),
                'resting_heart_rate': int(resting_hr),
                'chronic_conditions_count': chronic_conds,
                'physical_activity_days_per_week': phys_act,
                'sleep_hours_avg': round(sleep_hours, 1),
                'stress_level': stress,
                'depressive_symptoms_score': round(depression, 1),
                'target_high_risk': is_high_risk
            })
            
    df = pd.DataFrame(records)
    
    # Save
    # Save to processed as parquet for train.py compatibility
    output_parquet = output_path.replace('raw/synthetic.csv', 'processed/train.parquet')
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Synthetic data saved to {output_parquet}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/raw/synthetic.csv')
    args = parser.parse_args()
    
    generate_synthetic_data(args.output)
