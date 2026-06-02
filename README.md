# Credit Risk Default Prediction

Binary classification model to predict loan default using LightGBM,
with full explainability via SHAP values.

## Results

| Model               | AUC-ROC | PR-AUC | KS     | F1     | FNR    |
|---------------------|---------|--------|--------|--------|--------|
| Logistic Regression | 0.8631  | 0.3578 | 0.5707 | 0.2150 | 0.8663 |
| XGBoost             | 0.8696  | 0.3946 | 0.5795 | 0.2729 | 0.8249 |
| **LightGBM**        |**0.8707**|**0.3995**|**0.5824**|**0.3301**|**0.7706**|

> AUC-ROC 0.87 and KS 0.58 are above industry standard for credit risk models.

## Dataset

[Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit)
— 150,000 users, binary default prediction, 6.7% default rate.

## Project Structure

| Notebook | Description |
|----------|-------------|
| [01 - EDA](notebooks/credit-risk-01-eda.ipynb) | Data quality, class imbalance, missing values, distributions |
| [02 - Features](notebooks/credit-risk-02-features.ipynb) | Feature engineering — 11 → 22 features |
| [03 - Baseline](notebooks/credit-risk-03-baseline.ipynb) | XGBoost with calibration, threshold analysis |
| [04 - Comparison](notebooks/credit-risk-04-comparison.ipynb) | 3 models compared on identical holdout set |
| [05 - SHAP](notebooks/credit-risk-05-shap.ipynb) | LightGBM explainability — global + individual |

## Key Techniques

- **Class imbalance** — `scale_pos_weight=13.96` to weight defaulters
- **Probability calibration** — Isotonic regression for reliable probabilities
- **Threshold analysis** — FNR/FPR tradeoff mapped to financial impact
- **SHAP explainability** — Feature importance + individual prediction explanations
- **Evaluation metrics** — AUC-ROC, PR-AUC, KS Statistic, F1, FNR

## Key Findings

- `TotalLatePayments` is the #1 driver of default risk
- `RevolvingUtilization` is the #2 driver
- At threshold 0.2, FNR drops to ~0.46 (vs 0.77 at threshold 0.5)
- LightGBM outperforms XGBoost on all 5 metrics

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
