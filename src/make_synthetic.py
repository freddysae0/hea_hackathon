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
            
            # Time-varying features with some signal
            # Sick people have higher feature_1 trends
            feature_1 = np.random.normal(loc=base_health * (wave+1), scale=1.0)
            feature_2 = np.random.normal(loc=50, scale=10)
            
            # Target: Sick onset (binary) - slightly correlated with feature_1 and age
            prob_sick = 1 / (1 + np.exp(-(feature_1 * 0.5 + (age - 60) * 0.1 - 2)))
            sick_onset = 1 if np.random.rand() < prob_sick else 0
            
            records.append({
                'person_id': person_id,
                'year': year,
                'age': age,
                'sex': sex,
                'feature_1': feature_1,
                'feature_2': feature_2,
                'sick_onset': sick_onset
            })
            
    df = pd.DataFrame(records)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/raw/synthetic.csv')
    args = parser.parse_args()
    
    generate_synthetic_data(args.output)
