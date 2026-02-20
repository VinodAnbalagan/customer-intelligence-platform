# Customer Intelligence Platform

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.8+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**An end-to-end machine learning system for customer churn prediction with MLOps, explainability, and interactive business analytics.**


---

## Recent Update: Complete Project Rebuild

This repository was completely rebuilt in February 2025 to address fundamental limitations in the original implementation.

### The Problem with the Original Project

The original "Customer Purchasing Behaviors" project used a 238-record synthetic dataset that presented critical issues that made it unsuitable for demonstrating production-grade data science:

| Issue | Technical Impact | Business Impact |
|-------|------------------|-----------------|
| Extremely small sample size (238 records) | High variance in model estimates, unreliable cross-validation, severe overfitting risk | Results cannot be trusted for real business decisions |
| Synthetic/artificially generated data | Near-perfect correlations (>0.97) between features that would never exist in real customer data | Feature importance analysis is meaningless |
| Limited feature set (7 features) | Insufficient complexity to demonstrate real feature engineering | Does not reflect actual business data challenges |
| Artificially clean data | No missing values, no data quality issues | Does not demonstrate real-world data handling skills |
| Inflated performance metrics | R-squared scores of 0.999 that are impossible in real applications | Misrepresents model capabilities and sets unrealistic expectations |

The original project produced metrics that looked impressive on paper but would fail immediately when applied to real business data. A model trained on 238 synthetic records with 99.9% accuracy demonstrates nothing about a data scientist's ability to handle real-world challenges.

### The Solution: Real Data, Honest Metrics

This rebuild uses the Telco Customer Churn dataset, a real-world dataset from an actual telecommunications provider with genuine business complexity:

| Aspect | Original Dataset | New Dataset |
|--------|------------------|-------------|
| Sample Size | 238 records | 7,043 records |
| Data Source | Synthetic/Generated | Real Telecom Provider |
| Feature Count | 7 basic features | 21 original + 13 engineered features |
| Prediction Target | Spending amount (regression) | Customer churn (classification) |
| Class Distribution | Not applicable | 26.5% churn rate (realistic imbalance) |
| Data Quality | Artificially perfect | Real missing values, mixed types, inconsistencies |
| Feature Correlations | Artificially high (>0.97) | Realistic correlations reflecting actual business relationships |

This transition demonstrates a fundamental data science principle: a robust methodology applied to real data with honest 63% F1 score is infinitely more valuable than a 99.9% R-squared on synthetic data that would never generalize to production.

---

## Business Problem Statement

Customer churn represents one of the most significant challenges facing subscription-based businesses across all industries. Research consistently shows that acquiring a new customer costs between 5 to 25 times more than retaining an existing one. For telecommunications companies specifically, churn rates typically range from 20-40% annually, representing billions of dollars in lost revenue industry-wide.

This platform addresses churn prediction as a business problem by providing:

- Early identification of at-risk customers before they make the decision to leave
- Data-driven retention campaign targeting with measurable return on investment
- Interactive what-if analysis to understand which interventions are most effective
- Cost-benefit optimization to maximize retention budget efficiency
- Explainable predictions that business stakeholders can understand and act upon

---

## Model Performance Results

### Test Set Evaluation (Hold-Out Data)

The following metrics were computed on a 15% hold-out test set (1,057 customers) that the model never saw during training or validation:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| AUC-ROC | 0.847 | Strong ability to distinguish between churners and non-churners |
| Recall | 78.3% | The model correctly identifies approximately 4 out of 5 customers who will churn |
| Precision | 52.5% | Among customers flagged as high-risk, about half will actually churn |
| F1 Score | 0.629 | Balanced trade-off between precision and recall |
| Accuracy | 75.4% | Overall correct classification rate |

### Model Comparison (Validation Set)

Four algorithms were trained and compared on the validation set:

| Model | F1 Score | AUC-ROC | AUC-PR | Recall | Precision |
|-------|----------|---------|--------|--------|-----------|
| Logistic Regression | 0.632 | 0.850 | 0.655 | 83.6% | 50.9% |
| Random Forest | 0.631 | 0.840 | 0.629 | 73.2% | 55.4% |
| XGBoost | 0.616 | 0.835 | 0.619 | 75.7% | 52.0% |
| LightGBM | 0.616 | 0.840 | 0.623 | 77.1% | 51.3% |

Logistic Regression performed competitively with tree-based methods, suggesting relatively linear decision boundaries in this problem. This finding itself is valuable business insight, as it indicates the churn decision is driven by clear, interpretable factors rather than complex non-linear interactions.

### Business Impact Estimation

Based on standard industry cost assumptions:
- Cost of losing a customer: $500 (estimated customer lifetime value impact)
- Cost of retention campaign per customer: $100
- Expected retention success rate: 50%

At optimal threshold (0.35), the model is projected to save approximately $125,000 annually compared to no intervention, based on the test set scaled to annual customer volume.

---

## Technical Implementation

### Data Pipeline Architecture

The data pipeline follows a modular design with clear separation of concerns:

```
Raw Data (CSV)
    |
    v
load_data.py: Validation, schema checking, error handling
    |
    v
preprocess.py: Missing value imputation, type conversion, encoding
    |
    v
feature_engineering.py: 13 business-driven features with documentation
    |
    v
split.py: Stratified 70/15/15 train/validation/test split
    |
    v
Processed Data (Train, Validation, Test CSVs)
```

### Feature Engineering

Thirteen features were engineered based on domain knowledge and business rationale:

| Feature | Calculation | Business Rationale |
|---------|-------------|-------------------|
| tenure_group | Bucketed tenure (0-12, 13-24, 25-48, 49-72, 72+ months) | Customer lifecycle stages have different churn behaviors |
| is_new_customer | 1 if tenure <= 6 months | First 6 months are the critical retention window |
| monthly_to_total_ratio | MonthlyCharges / (TotalCharges + 1) | High ratio indicates new customers or billing anomalies |
| avg_monthly_charge | TotalCharges / (tenure + 1) | Historical spending pattern differs from current rate if plan changed |
| charge_consistency | MonthlyCharges / avg_monthly_charge | Values far from 1.0 indicate recent pricing changes |
| high_monthly_charge | 1 if MonthlyCharges > median | Price-sensitive customers with high bills may churn more |
| service_count | Count of subscribed services | More services equals higher switching costs |
| has_premium_support | 1 if OnlineSecurity=Yes OR TechSupport=Yes | Support services indicate customer investment |
| has_streaming | 1 if StreamingTV=Yes OR StreamingMovies=Yes | Entertainment users have specific value-add needs |
| contract_value | 0=month-to-month, 1=one-year, 2=two-year | Numerical encoding of commitment level |
| auto_payment | 1 if PaymentMethod contains automatic | Automatic payments indicate trust and reduce friction |
| has_family | 1 if Partner=Yes OR Dependents=Yes | Family accounts often have different stability |
| is_senior_alone | 1 if SeniorCitizen=1 AND no partner/dependents | Potentially vulnerable segment needing different approach |

### Model Training Pipeline

```
Feature Matrix (X) + Target (y)
    |
    v
StandardScaler: Normalize numerical features
LabelEncoder: Encode categorical features
    |
    v
Train 4 Models: Logistic Regression, Random Forest, XGBoost, LightGBM
    |
    v
Evaluate on Validation Set: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR
    |
    v
Select Best Model based on F1 Score (churn class)
    |
    v
Final Evaluation on Test Set (held out until this point)
    |
    v
SHAP Analysis for Explainability
    |
    v
Save Model Artifacts (model.pkl, preprocessor.pkl)
```

### MLOps Integration

- MLflow experiment tracking logs all hyperparameters, metrics, and artifacts
- Model versioning through MLflow model registry
- Reproducible experiments with logged random seeds
- Artifact storage for confusion matrices, ROC curves, and feature importance plots

---

## Key Business Insights

Analysis of the model's feature importance and SHAP values reveals actionable business insights:

### 1. Contract Type is the Strongest Predictor

Month-to-month customers churn at 42.7% compared to just 2.8% for two-year contract customers. This 15x difference represents the single most actionable insight: converting month-to-month customers to annual contracts should be the primary retention strategy.

### 2. Customer Tenure Follows a Risk Curve

New customers (0-12 months tenure) represent the highest risk segment. Churn risk decreases substantially after the first year as customers develop loyalty and switching costs accumulate. Early engagement programs targeting the first 6 months are critical.

### 3. Support Services Reduce Churn

Customers without Tech Support churn at 41% versus 15% for those with support. This suggests that support services either attract more committed customers or that the support experience itself builds loyalty. Either interpretation supports bundling support services with retention offers.

### 4. Fiber Optic Customers Show Higher Churn

Despite being a premium service, fiber optic customers churn more than DSL customers. This counterintuitive finding likely reflects price sensitivity: fiber optic service costs more, and customers may be more likely to shop for alternatives.

### 5. Payment Method Signals Intent

Customers using electronic check (manual monthly payment) churn at significantly higher rates than those with automatic payments. Encouraging automatic payment enrollment reduces friction and serves as a soft commitment mechanism.

---

## Interactive Dashboard

The Streamlit dashboard provides five pages for different user needs:

### Page 1: Executive Summary
- Key performance metrics displayed as metric cards
- Churn distribution visualization
- Top 5 churn drivers from SHAP analysis
- Risk segment breakdown

### Page 2: Customer Risk Explorer
- Filterable data table of all customers
- Color-coded churn probability (green/yellow/orange/red)
- Risk segment assignment
- Top 3 risk factors per customer from SHAP values
- Click-to-expand individual customer explanations

### Page 3: What-If Simulator
This is the key differentiating feature of the dashboard. Users can:
- Select any customer or create a hypothetical customer profile
- Adjust feature values using sliders and dropdowns (contract type, tenure, services, payment method)
- See real-time probability updates as features change
- View SHAP waterfall plot updating live
- Quantify intervention impact (e.g., "Switching to 1-year contract reduces churn risk from 78% to 23%")

### Page 4: Model Performance
- ROC curve with AUC annotation
- Precision-Recall curve
- Interactive confusion matrix with adjustable threshold slider
- Calibration curve showing probability reliability
- Model comparison table

### Page 5: Business Impact
- Cost analysis chart showing total cost at different thresholds
- Optimal threshold calculation based on business costs
- Segment-level ROI analysis
- Expected savings calculator

---

## Deployment

### Streamlit Cloud Deployment

This application is deployed on Streamlit Cloud for public access. The deployment process:

1. Repository connected to Streamlit Cloud
2. Requirements installed from requirements.txt
3. Model artifacts (best_model.pkl, preprocessor.pkl) loaded at startup
4. No retraining required on deployment - uses pre-trained models

### Local Development

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

# Train models (generates model artifacts)
python -m src.models.train

# Launch dashboard
streamlit run app.py
```

---

## Project Structure

```
customer-intelligence-platform/
|
|-- app.py                          # Streamlit main application entry point
|-- requirements.txt                # Python dependencies
|-- model_card.md                   # Model documentation following best practices
|-- README.md                       # This file
|
|-- pages/                          # Streamlit multi-page app pages
|   |-- 1_Executive_Summary.py      # Key metrics and overview
|   |-- 2_Customer_Risk_Explorer.py # Customer browsing and filtering
|   |-- 3_What_If_Simulator.py      # Interactive intervention testing
|   |-- 4_Model_Performance.py      # Model evaluation visualizations
|   |-- 5_Business_Impact.py        # Cost analysis and ROI
|
|-- src/                            # Source code modules
|   |-- __init__.py
|   |-- data/                       # Data processing pipeline
|   |   |-- __init__.py
|   |   |-- load_data.py            # Data loading and validation
|   |   |-- preprocess.py           # Cleaning and transformation
|   |   |-- feature_engineering.py  # Feature creation with documentation
|   |   |-- split.py                # Stratified train/val/test splitting
|   |
|   |-- models/                     # Machine learning pipeline
|   |   |-- __init__.py
|   |   |-- train.py                # Model training and comparison
|   |   |-- hyperparameter_tuning.py # Optuna optimization
|   |   |-- handle_imbalance.py     # SMOTE, class weights comparison
|   |   |-- calibration.py          # Probability calibration
|   |
|   |-- explainability/             # Model interpretation
|   |   |-- __init__.py
|   |   |-- shap_analysis.py        # SHAP value computation and plots
|   |   |-- feature_importance.py   # Multi-method importance comparison
|   |
|   |-- mlops/                      # MLflow integration
|   |   |-- __init__.py
|   |   |-- experiment_tracking.py  # Experiment logging
|   |   |-- model_registry.py       # Model versioning
|   |
|   |-- business/                   # Business logic
|       |-- __init__.py
|       |-- cost_analysis.py        # Threshold optimization
|       |-- customer_segments.py    # Risk segmentation
|
|-- data/
|   |-- raw/                        # Original dataset (gitignored)
|   |-- processed/                  # Train/val/test splits
|
|-- models/                         # Saved model artifacts
|   |-- best_model.pkl              # Trained XGBoost model
|   |-- preprocessor.pkl            # Fitted preprocessors
|
|-- tests/                          # Unit tests
|   |-- __init__.py
|   |-- test_preprocessing.py       # Data pipeline tests
|   |-- test_model.py               # Model prediction tests
|
|-- reports/
    |-- figures/                    # Generated visualizations
```

---

## Technical Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | >= 1.30.0 | Interactive dashboard |
| pandas | >= 2.0.0 | Data manipulation |
| numpy | >= 1.24.0 | Numerical operations |
| scikit-learn | >= 1.3.0 | Machine learning utilities |
| xgboost | >= 1.7.0 | Gradient boosting model |
| lightgbm | >= 4.0.0 | Gradient boosting model |
| optuna | >= 3.4.0 | Hyperparameter optimization |
| shap | >= 0.43.0 | Model explainability |
| mlflow | >= 2.8.0 | Experiment tracking |
| plotly | >= 5.18.0 | Interactive visualizations |
| matplotlib | >= 3.8.0 | Static visualizations |
| seaborn | >= 0.13.0 | Statistical visualizations |
| imbalanced-learn | >= 0.11.0 | SMOTE and resampling |
| joblib | >= 1.3.0 | Model serialization |

---

## Lessons Learned

This project rebuild reinforced several important principles:

### Real Data Beats Synthetic Data Every Time

Working with the Telco Churn dataset exposed challenges that synthetic data hides: missing values in unexpected places (TotalCharges as empty strings), categorical variables with "No internet service" as a distinct value, and realistic class imbalance. These challenges are precisely what production data science requires.

### Honest Metrics Build Trust

A 63% F1 score on real data is infinitely more valuable than a 99.9% score on synthetic data. Stakeholders who understand the difficulty of churn prediction will trust honest metrics. Those who expect perfection will be disappointed regardless.

### Explainability is Not Optional

The SHAP analysis revealed that contract type is by far the most important predictor. This insight alone justifies the entire project because it tells the business exactly where to focus retention efforts. A black-box model with slightly better accuracy but no explanation would be less useful.

### End-to-End Implementation Matters

A model in a Jupyter notebook is not a data science project. This platform includes data pipelines, model training, experiment tracking, explainability, business analysis, and an interactive dashboard. The complete system demonstrates production readiness.

---

## Future Enhancements

The following improvements are planned for future iterations:

- Time-series feature engineering incorporating usage trends and payment history patterns
- Real-time scoring API using FastAPI for integration with CRM systems
- A/B testing framework to measure actual retention campaign effectiveness
- Automated model retraining pipeline triggered by performance degradation
- Additional deployment to Hugging Face Spaces for redundancy

---

## Dataset Citation

The Telco Customer Churn dataset is available on Kaggle:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## Contact

Vinod Anbalagan
- GitHub: https://github.com/VinodAnbalagan
- LinkedIn: https://linkedin.com/in/vinodanbalagan

---

Built with Python, Streamlit, XGBoost, SHAP, and MLflow
