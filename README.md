# HEA Hackathon 2025 - Hidden Health Signals

## Quick Start (3 commands to win)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline**:
   ```bash
   # Preprocess data
   python src/preprocess.py --config config.yaml
   
   # Train model (saves artifacts to models/)
   python src/train.py --config config.yaml
   
   # Evaluate (generates reports)
   python src/evaluate.py --config config.yaml
   ```

3. **Predict**:
   ```bash
   python src/predict.py --input data/raw/test.csv --output data/processed/predictions.csv
   ```

## Structure
- `data/`: managed by `config.yaml`. Do not commit raw data.
- `src/`: reproducible source code.
- `reports/`: `leakage_report.md`, `fairness_report.md`, `model_card.md`.
- `notebooks/`: `01_eda.ipynb` for initial exploration.

## Reports
- [Model Card](reports/model_card.md)
- [Leakage Audit](reports/leakage_report.md)
