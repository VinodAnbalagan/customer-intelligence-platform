# Customer Intelligence Platform

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-blue.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An end-to-end machine learning system for customer churn prediction with MLOps, explainability, and interactive business analytics.**

---

## Project Evolution

> **Important:** This project represents a complete rebuild from the original "Customer Purchasing Behaviors" analysis.

### Why the Change?

The original project used a **238-record synthetic dataset** that presented several limitations:

| Issue | Impact |
|-------|--------|
| **Tiny sample size (238 records)** | High overfitting risk, unreliable cross-validation |
| **Synthetic/generated data** | Near-perfect correlations (>0.97) that don't exist in real business data |
| **Limited features** | Only 7 features with artificial relationships |
| **Inflated metrics** | R² scores of 0.999 that would never occur in production |

### The New Approach

This rebuild uses the **Telco Customer Churn dataset** — real-world data with real-world messiness:

| Aspect | Old Dataset | New Dataset |
|--------|-------------|-------------|
| **Records** | 238 | 7,043 |
| **Source** | Synthetic/Generated | Real Telco Provider |
| **Features** | 7 | 21 + 13 engineered |
| **Target** | Spending prediction | Churn prediction (business-critical) |
| **Class Balance** | N/A | 26.5% churn (realistic imbalance) |
| **Data Quality** | Perfect | Missing values, mixed types (realistic) |

This transition demonstrates a key data science principle: **methodology matters more than metrics**. A robust pipeline on real data with honest 63% F1 is far more valuable than a 99.9% R² on synthetic data.

---

## Business Problem

Customer churn is one of the most critical challenges facing subscription-based businesses. Acquiring new customers costs 5-25x more than retaining existing ones. This platform enables:

- **Early identification** of at-risk customers before they churn
- **Data-driven retention** campaigns with measurable ROI
- **What-if analysis** to optimize intervention strategies
- **Cost-benefit optimization** for retention budgets

---

## Key Results

| Metric | Value | Business Impact |
|--------|-------|-----------------|
| **Model AUC-ROC** | 0.847 | Strong discrimination between churners and non-churners |
| **Recall** | 78.3% | Catches ~4 out of 5 potential churners |
| **Precision** | 52.5% | Half of flagged customers will actually churn |
| **F1 Score** | 0.629 | Balanced precision-recall performance |

### Model Comparison (Validation Set)

| Model | F1 Score | AUC-ROC | AUC-PR | Recall |
|-------|----------|---------|--------|--------|
| Logistic Regression | 0.632 | 0.850 | 0.655 | 83.6% |
| Random Forest | 0.631 | 0.840 | 0.629 | 73.2% |
| XGBoost | 0.616 | 0.835 | 0.619 | 75.7% |
| LightGBM | 0.616 | 0.840 | 0.623 | 77.1% |

---

## Features

### Data Pipeline
- Automated data loading and validation
- Comprehensive preprocessing (missing values, encoding, scaling)
- 13 engineered features with documented business rationale
- Stratified train/val/test splits (70/15/15) preserving class distribution

### Machine Learning
- Model comparison: Logistic Regression, Random Forest, XGBoost, LightGBM
- Hyperparameter optimization with Optuna
- Class imbalance handling (SMOTE, class weights, undersampling)
- Probability calibration for reliable predictions

### Explainability
- Global SHAP analysis for feature importance
- Local SHAP explanations for individual predictions
- Feature importance comparison across methods
- Business-friendly interpretation of model decisions

### MLOps
- MLflow experiment tracking and model registry
- Automated metric logging and artifact storage
- Model versioning and deployment artifacts
- Comprehensive model card with documentation

### Business Analytics
- Cost-benefit analysis with customizable assumptions
- Optimal threshold calculation for business objectives
- Customer risk segmentation (Safe, Monitor, At Risk, Critical)
- ROI projections for intervention strategies

### Interactive Dashboard (5 Pages)
1. **Executive Summary** — Key metrics, churn drivers, risk distribution
2. **Customer Risk Explorer** — Browse and filter at-risk customers
3. **What-If Simulator** — Test intervention impact in real-time
4. **Model Performance** — ROC curves, confusion matrices, calibration
5. **Business Impact** — Cost analysis and ROI calculations

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit Dashboard                       │
├─────────────────────────────────────────────────────────────────┤
│  Executive  │  Risk      │  What-If   │  Model    │  Business   │
│  Summary    │  Explorer  │  Simulator │  Perf     │  Impact     │
└──────┬──────┴─────┬──────┴─────┬──────┴─────┬─────┴──────┬──────┘
       │            │            │            │            │
       └────────────┴────────────┼────────────┴────────────┘
                                 │
       ┌─────────────────────────┴─────────────────────────┐
       │                   ML Pipeline                      │
       ├───────────────────────────────────────────────────┤
       │  Data Processing → Feature Eng → Model Training   │
       │         ↓              ↓              ↓           │
       │  Preprocessing   → 13 Features  → 4 Models       │
       │  (missing vals,     (business     (LR, RF,       │
       │   encoding)          rationale)    XGB, LGBM)    │
       └───────────────────────────────────────────────────┘
                                 │
       ┌─────────────────────────┴─────────────────────────┐
       │                 MLOps Layer                        │
       ├───────────────────────────────────────────────────┤
       │  MLflow Tracking → Model Registry → Artifacts     │
       └───────────────────────────────────────────────────┘
```

---

## Dataset

**Telco Customer Churn** from Kaggle:
- **Source**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Size**: 7,043 customers, 21 features
- **Class Distribution**: 26.5% churn (realistic imbalance)
- **Features**: Demographics, account info, services, charges

### Engineered Features

| Feature | Business Rationale |
|---------|-------------------|
| `tenure_group` | Customer lifecycle stages have different churn risks |
| `is_new_customer` | First 6 months are critical retention period |
| `monthly_to_total_ratio` | High ratio indicates new customers |
| `service_count` | More services = higher switching costs |
| `has_premium_support` | Support services reduce churn |
| `contract_value` | Commitment level (0=month-to-month, 2=two-year) |
| `auto_payment` | Automatic payments indicate trust and reduce friction |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/VinodAnbalagan/customer-intelligence-platform.git
cd customer-intelligence-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place in data/raw/
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn
```

## Usage

### Train Models
```bash
python -m src.models.train
```

### Launch Dashboard
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

### View MLflow Experiments
```bash
mlflow ui --backend-store-uri mlruns
```

---

## Project Structure

```
customer-intelligence-platform/
├── app.py                          # Streamlit main app
├── pages/                          # Dashboard pages
│   ├── 1_Executive_Summary.py
│   ├── 2_Customer_Risk_Explorer.py
│   ├── 3_What_If_Simulator.py
│   ├── 4_Model_Performance.py
│   └── 5_Business_Impact.py
├── src/
│   ├── data/                       # Data pipeline (4 modules)
│   ├── models/                     # ML pipeline (4 modules)
│   ├── explainability/             # SHAP analysis (2 modules)
│   ├── mlops/                      # MLflow tracking (2 modules)
│   └── business/                   # Cost analysis (2 modules)
├── data/
│   ├── raw/                        # Raw dataset
│   └── processed/                  # Train/val/test splits
├── models/                         # Saved model artifacts
├── tests/                          # Unit tests
├── requirements.txt
├── model_card.md
└── README.md
```

---

## Key Insights from Real Data

1. **Contract type is the #1 predictor**: Month-to-month customers churn at 42.7% vs 2.8% for two-year contracts

2. **Tenure matters**: New customers (< 12 months) are highest risk; loyalty builds over time

3. **Support services reduce churn**: Customers without Tech Support churn at 41% vs 15% with support

4. **Fiber optic shows higher churn**: Likely due to price sensitivity (higher monthly charges)

5. **Payment method signals intent**: Electronic check users churn more; auto-pay customers stay

---

## Technical Decisions

### Why Multiple Models?
We compare 4 algorithms to find the best fit. Interestingly, Logistic Regression performed competitively due to the relatively linear decision boundary in this problem.

### Why 78% Recall over Higher Precision?
In churn prediction, **missing a churner (false negative) costs more than a wasted retention offer (false positive)**. We optimize for catching churners, accepting some false alarms.

### Why SHAP?
SHAP provides consistent, theoretically grounded explanations that show both feature importance AND direction of impact, enabling actionable business insights.

---

## Lessons Learned

This project reinforced several key principles:

1. **Real data > Synthetic data**: Honest metrics on messy data are more valuable than inflated metrics on clean synthetic data

2. **Business context matters**: A 63% F1 score that saves $125K annually is more impactful than a 99% score on meaningless data

3. **Explainability is essential**: Stakeholders need to understand WHY the model makes predictions, not just WHAT it predicts

4. **End-to-end thinking**: A model is only valuable if it's deployed, monitored, and actionable

---

## Future Improvements

- [ ] Add time-series features (usage trends, payment history)
- [ ] Implement real-time scoring API with FastAPI
- [ ] Add A/B testing framework for retention campaigns
- [ ] Build automated retraining pipeline
- [ ] Deploy to Hugging Face Spaces

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

**Vinod Anbalagan**
- GitHub: [@VinodAnbalagan](https://github.com/VinodAnbalagan)
- LinkedIn: [vinodanbalagan](https://linkedin.com/in/vinodanbalagan)

---

*Built with Python, Streamlit, XGBoost, SHAP, and MLflow*
