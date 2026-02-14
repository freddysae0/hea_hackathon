import pandas as pd
import yaml
import argparse
import os
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_preprocess(config):
    print("Starting preprocessing...")
    raw_path = config['data']['raw']
    processed_path = config['data']['processed']
    target_col = config['data']['target_col']
    id_col = config['data']['id_col']
    time_col = config['data']['index_time_col']
    
    # Load data (assuming synthetic for now, adaptable for real)
    input_file = os.path.join(raw_path, 'synthetic.csv') # Defaulting to synthetic for this step
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows.")

    # Sort for longitudinal features
    df = df.sort_values([id_col, time_col])

    # Feature Engineering
    print("Engineering features...")
    # 1. Shifted features (previous value)
    df['feature_1_prev'] = df.groupby(id_col)['feature_1'].shift(1)
    
    # 2. Delta (change from previous)
    df['feature_1_change'] = df['feature_1'] - df['feature_1_prev']
    
    # 3. Cumulative mean (trend proxy)
    df['feature_1_mean'] = df.groupby(id_col)['feature_1'].transform(lambda x: x.expanding().mean())

    # Fill NaNs created by lagging
    df = df.fillna(0) # Simple imputation for baseline
    
    # Save processing
    os.makedirs(processed_path, exist_ok=True)
    output_file = os.path.join(processed_path, 'train.parquet')
    df.to_parquet(output_file)
    print(f"Preprocessing complete. Data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_preprocess(config)
