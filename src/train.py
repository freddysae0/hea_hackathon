import pandas as pd
import lightgbm as lgb
import yaml
import argparse
import joblib
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(config):
    print("Starting training...")
    params = config['model']['params']
    processed_path = config['data']['processed']
    target_col = config['data']['target_col']
    id_col = config['data']['id_col']
    
    # Load data
    input_file = os.path.join(processed_path, 'train.parquet')
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found. Run preprocess first.")
        return

    df = pd.read_parquet(input_file)
    
    # Prepare X and y
    # Drop non-feature columns
    drop_cols = [target_col, id_col, config['data']['index_time_col'], 'sex'] # 'sex' is categorical string, need encoding if used
    # Simple encoding for 'sex' if it exists and is string
    if 'sex' in df.columns and df['sex'].dtype == 'O':
        df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
        drop_cols.remove('sex')

    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]
    groups = df[id_col]
    
    print(f"Training on {X.shape[1]} features...")

    # Cross Validation
    gkf = GroupKFold(n_splits=config['training']['cv_folds'])
    
    aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
        
        model = lgb.train(
            params, 
            train_set, 
            valid_sets=[val_set], 
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        
        # Predict
        preds = model.predict(X_val)
        score = roc_auc_score(y_val, preds)
        aucs.append(score)
        print(f"Fold {fold+1} AUC: {score:.4f}")
    
    print(f"Average AUC: {sum(aucs)/len(aucs):.4f}")

    # Train final model on all data
    print("Training final model on full_data...")
    full_set = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, full_set, num_boost_round=100)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_model, 'models/model.pkl')
    print("Training complete. Model saved to models/model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_model(config)
