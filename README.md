# Financial Fraud Detection: Unsupervised Anomaly Detection vs. Supervised Ensembles

> End-to-end fraud detection system on 590K real-world financial transactions,
> comparing a deep AutoEncoder anomaly detector against XGBoost and LightGBM
> supervised classifiers — with SHAP explainability and MLflow experiment tracking.

---

## Results Summary

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| **XGBoost** | **0.9483** | **0.6755** | 0.6424 | 0.7100 | 0.5865 |
| LightGBM | 0.9393 | 0.6438 | 0.6201 | 0.6723 | 0.5754 |
| AutoEncoder | 0.6727 | 0.1153 | 0.1911 | 0.1745 | 0.2112 |

> **Primary metric: PR-AUC** — standard accuracy is misleading at 3.5% fraud rate.
> A random classifier scores 0.035 on PR-AUC; XGBoost achieves 0.676.

---

## Business Context

| Metric | Value |
|---|---|
| Total transactions | 590,540 |
| Fraudulent transactions | 20,663 |
| Fraud rate | 3.50% |
| Avg transaction amount | $135.03 |
| Total fraud exposure | $3,083,844 |
| Legitimate per fraud | ~27:1 |

Standard accuracy is misleading on this dataset — a model predicting
"always legitimate" achieves 96.5% accuracy while catching zero fraud.
This project optimizes for **PR-AUC** and **business cost-aware thresholds**.

---

## Dataset

**IEEE-CIS Fraud Detection** — Kaggle competition dataset  
590K e-commerce transactions with two joinable tables:

- `train_transaction.csv` — 590K rows, financial features (TransactionAmt,
  ProductCD, card1-6, C1-C14 counters, D1-D15 time deltas, M1-M9 match
  fields, Vxxx engineered features)
- `train_identity.csv` — 144K rows, device/network identity features
  (IP, browser, OS, proxy info) joined on `TransactionID`

**Access:** https://www.kaggle.com/c/ieee-fraud-detection/data

---

## Pipeline Overview

```
train_transaction.csv + train_identity.csv
        │
        ▼
Left join on TransactionID
(590K transactions, 144K with identity info)
        │
        ▼
Missing Value Analysis
Drop 213 columns with >50% missing
        │
        ▼
Feature Engineering
  Temporal  │  hour, day_of_week, is_night, is_weekend
  Amount    │  log_amount, is_round_amount, amount_cents
  Card      │  card1_freq, card2_freq (transaction velocity)
  Email     │  same_email flag (buyer/seller domain match)
  Identity  │  has_identity flag
        │
        ▼
sklearn Pipeline
MedianImputer + StandardScaler + OrdinalEncoder
        │
        ▼
Stratified Train/Test Split (80/20)
229 features │ 472K train │ 118K test
        │
   ┌────┴────┐
   ▼         ▼
AutoEncoder        XGBoost + LightGBM
(unsupervised)     (supervised)
Train on legit     scale_pos_weight /
only — no labels   class_weight='balanced'
   │               │
   ▼               ▼
Reconstruction     Fraud probability score
error as score     + F1-optimal threshold
   └────┬────┘
        │
        ▼
Model Comparison
ROC + PR curves │ Confusion matrices │ Business impact
        │
        ▼
SHAP Explainability (XGBoost — TreeExplainer)
Beeswarm + Waterfall + Dependence plots
        │
        ▼
MLflow Experiment Tracking
params + metrics + model artifacts
```
    
---

## Module Breakdown

### Module 1 — Data Loading, EDA & Preprocessing
- Joins transaction + identity tables
- Missing value heatmap and drop strategy (213 columns >50% missing)
- EDA: fraud by product code, fraud rate by hour of day, amount distributions
- Feature engineering: 11 new domain features
- `sklearn` ColumnTransformer pipeline (reproducible, production-ready)

### Module 2 — AutoEncoder Anomaly Detection
- PyTorch AutoEncoder: `229 → 128 → 64 → 32 → 64 → 128 → 229`
- Trained exclusively on legitimate transactions (no fraud labels)
- Reconstruction error as anomaly score
- **4.13x separation ratio** between fraud and legitimate mean error
- F1-optimal threshold via precision-recall curve

### Module 3 — Supervised Ensemble Baseline
- XGBoost with `scale_pos_weight` for class imbalance
- LightGBM with `class_weight='balanced'`
- Head-to-head comparison vs AutoEncoder on ROC + PR curves
- XGBoost: **ROC-AUC 0.9483**, **PR-AUC 0.6755**

### Module 4 — SHAP Explainability
- `TreeExplainer` on XGBoost (stratified 5K sample)
- Beeswarm summary: global feature importance
- Waterfall plots: individual fraud vs. legitimate transaction explanation
- Dependence plot: C13 interaction with C1
- **Top finding:** C13 (card address count ≈ 0) is the strongest fraud signal

### Module 5 — MLflow Experiment Tracking
- All three model runs logged with full hyperparameters and metrics
- XGBoost and LightGBM models serialized as MLflow artifacts
- Comparable experiment table across unsupervised and supervised approaches

---

## Key Findings

**1. Supervised vs. Unsupervised tradeoff**

XGBoost outperforms the AutoEncoder by 0.28 ROC-AUC points. The gap
reflects a fundamental difference: XGBoost directly optimizes the
fraud/legitimate boundary using labels, while the AutoEncoder infers
anomaly from reconstruction error alone — a weaker signal on tabular
financial data where fraud doesn't deviate dramatically from normal
feature distributions.

**2. AutoEncoder is not useless**

The 4.13x reconstruction error separation demonstrates the AutoEncoder
learned meaningful structure. Its advantage is label-free detection —
valuable in production cold-start scenarios where labeled fraud data is
scarce, delayed, or unavailable. In practice, both models running in
parallel is the strongest setup.

**3. C13 is the top fraud predictor**

SHAP identifies C13 (count of addresses associated with a payment card)
as the most predictive feature. Cards with C13 ≈ 0 strongly indicate
fraud — fraudsters tend to use cards with minimal transaction history.
C1 and C14 (also counter features) rank #2 and #3.

**4. Engineered features contribute meaningfully**
`card1_freq` (card transaction velocity — engineered, not in raw data)
ranks in the SHAP top 10, validating the feature engineering step.


---

## Why PR-AUC over ROC-AUC?

At 3.5% fraud rate, ROC-AUC can be inflated by the model's ability to correctly classify the abundant legitimate class. PR-AUC focuses specifically on the minority class — precision and recall trade-offs are what actually matter when a fraud team is setting production thresholds.


---

## How to Run

### On Kaggle
1. Go to https://www.kaggle.com/c/ieee-fraud-detection
2. Create a new notebook and attach the competition dataset
3. Upload `fraud-detection-system.ipynb`
4. Enable GPU accelerator (T4 — free tier)
5. Run All

### Locally
```bash
git clone https://github.com/neshatsh/fraud-detection-system
cd fraud-detection-system
pip install -r requirements.txt

# Download data from Kaggle and place in ./data/
# Then open and run the notebook
jupyter notebook notebook/fraud_detection.ipynb
```

---

## Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
xgboost
lightgbm
shap
mlflow
```

