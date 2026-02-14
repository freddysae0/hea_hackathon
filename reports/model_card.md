# Model Card

## Model Details
- **Architecture**: LightGBM (Gradient Boosting)
- **Version**: 1.0
- **License**: MIT
- **Developed by**: [Team Name]

## Intended Use
- **Primary Use**: Early prediction of [Disease Name] onset based on longitudinal self-reported data.
- **Intended Users**: Researchers, Public Health Officials.
- **Out of Scope**: Clinical diagnosis replacement.

## Training Data
- **Source**: [Dataset Name] (e.g., RAND HRS, NLSY97)
- **Time Range**: 1992-2020
- **Preprocessing**: Group-based splitting, leakage removal.

## Performance
- **Primary Metric**: F2 Score (Recall-weighted)
- **Secondary Metrics**: PR-AUC, ROC-AUC

## Ethical Considerations
- **Fairness**: Evaluate performance across Age, Sex, and Race.
- **Privacy**: No PII used.

## Caveats and Recommendations
- Based on self-reported data, which may subjective.
