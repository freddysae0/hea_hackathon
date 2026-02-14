import pandas as pd
import joblib
import yaml
import argparse
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def predict(config, input_path, output_path):
    print(f"Loading model from models/model.pkl...")
    # model = joblib.load('models/model.pkl')
    
    print(f"Loading input data from {input_path}...")
    # df = pd.read_csv(input_path)
    
    print("Generating predictions...")
    # preds = model.predict(df)
    
    # Save predictions
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # pd.DataFrame({'pred': preds}).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    predict(config, args.input, args.output)
