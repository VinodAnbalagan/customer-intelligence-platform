# Model Card: Customer Churn Prediction

## Model Description

**Model Type:** XGBoost Classifier
**Task:** Binary Classification (Customer Churn Prediction)
**Framework:** XGBoost 1.7+, scikit-learn 1.3+

### Intended Use

This model predicts the probability that a telecom customer will churn (cancel their subscription) based on their account characteristics, usage patterns, and demographic information.

**Primary Use Cases:**
- Identify at-risk customers for proactive retention campaigns
- Calculate optimal intervention thresholds for cost-effective retention
- Provide customer service representatives with churn risk scores
- Power interactive dashboards for business analytics

**Users:**
- Marketing and retention teams
- Customer service departments
- Business intelligence analysts
- Data science teams

**Out of Scope:**
- Real-time fraud detection
- Customer acquisition scoring
- Credit risk assessment

## Training Data

**Dataset:** Telco Customer Churn (Kaggle)
**Source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
**Size:** 7,043 customers
**Features:** 21 original features + 13 engineered features
**Class Distribution:** 73.5% retained, 26.5% churned (imbalanced)

### Data Splits
- Training: 70% (4,930 samples)
- Validation: 15% (1,056 samples)
- Test: 15% (1,057 samples)
- Stratified splitting to preserve class ratios

### Data Processing
1. Missing values in TotalCharges (11 rows) filled with MonthlyCharges
2. Target encoded: "Yes" → 1, "No" → 0
3. Categorical features label-encoded
4. Numerical features standardized (StandardScaler)
5. Class imbalance handled via `scale_pos_weight` parameter

### Feature Engineering
- `tenure_group`: Customer lifecycle stage
- `monthly_to_total_ratio`: Billing pattern indicator
- `service_count`: Number of subscribed services
- `has_premium_support`: TechSupport or OnlineSecurity
- `contract_value`: Numerical contract commitment (0-2)
- `auto_payment`: Automatic payment method flag
- See `src/data/feature_engineering.py` for full documentation

## Model Architecture

**Algorithm:** XGBoost (Extreme Gradient Boosting)
**Hyperparameters:**
- `n_estimators`: 100
- `max_depth`: 6
- `learning_rate`: 0.1
- `scale_pos_weight`: 2.74 (ratio of negative to positive class)
- `eval_metric`: logloss

**Tuning:** Optuna optimization (100 trials) maximizing F1 score

## Performance Metrics

### Test Set Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.803 |
| Precision | 0.658 |
| Recall | 0.524 |
| F1 Score | 0.583 |
| AUC-ROC | 0.847 |
| AUC-PR | 0.672 |
| Log Loss | 0.421 |

### Confusion Matrix (at 0.5 threshold)
```
              Predicted
              No    Yes
Actual No    700     77
       Yes   133    147
```

### Calibration
- Expected Calibration Error (ECE): 0.045
- Brier Score: 0.142
- Probabilities are well-calibrated and suitable for business decisions

## Feature Importance

Top 5 features by SHAP importance:
1. Contract (Month-to-month): +0.25 impact
2. tenure: -0.18 impact (longer tenure → lower churn)
3. TechSupport (No): +0.10 impact
4. OnlineSecurity (No): +0.09 impact
5. MonthlyCharges: +0.08 impact

## Limitations

### Data Limitations
- Single telecom provider dataset; may not generalize to other industries
- Historical data from one time period; customer behavior may change
- Limited demographic features (no income, education, etc.)
- US-centric data; international markets may differ

### Performance Limitations
- Recall of 52.4% means ~48% of churners are missed at default threshold
- Performance varies by customer segment:
  - Better for month-to-month customers (higher base rate)
  - Weaker for long-tenure customers (rare churn events)
- Confidence intervals not quantified; point estimates only

### Deployment Limitations
- Requires all input features; missing features cause errors
- Batch prediction; not optimized for sub-millisecond latency
- Model should be retrained quarterly or when performance degrades

## Ethical Considerations

### Bias Assessment
- **Age:** SeniorCitizen feature used; potential age discrimination
- **Geography:** No geographic features, but InternetService may proxy for area
- **Socioeconomic:** MonthlyCharges/TotalCharges may correlate with income
- **Recommendation:** Regular bias audits across protected groups

### Fairness
- Model optimizes for overall performance, not subgroup parity
- Retention offers should be applied equitably across segments
- Monitor for disparate impact in intervention decisions

### Privacy
- Model uses customer account data (PII)
- No personally identifiable information in predictions
- Ensure compliance with GDPR, CCPA, and local regulations
- Implement appropriate access controls for model outputs

### Transparency
- SHAP explanations available for all predictions
- Feature importance documented and interpretable
- Model card maintained with performance metrics

## Maintenance

### Retraining Schedule
- Quarterly retraining with updated customer data
- Emergency retraining if performance drops >5% on key metrics
- Version control all model artifacts

### Monitoring
- Track prediction distribution drift weekly
- Compare predicted vs actual churn rates monthly
- Alert on significant deviation from baseline

### Feedback Loop
- Incorporate retention campaign outcomes
- Track false positive/negative rates by segment
- Update cost assumptions based on actual CLV data

## Caveats and Recommendations

1. **Threshold Selection:** Default 0.5 threshold optimizes accuracy; use cost-optimized threshold (~0.35) for business decisions

2. **Probability Interpretation:** Predicted probabilities represent relative risk, not absolute churn likelihood over a specific time period

3. **Intervention Timing:** Model provides current risk score; does not predict when churn will occur

4. **Complementary Analysis:** Combine with qualitative customer feedback and support ticket analysis

## References

- Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- XGBoost Documentation: https://xgboost.readthedocs.io/
- SHAP Documentation: https://shap.readthedocs.io/
- MLflow Documentation: https://mlflow.org/docs/latest/index.html

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-XX | Initial model deployment |

---

*This model card follows the format recommended by Mitchell et al. (2019) "Model Cards for Model Reporting"*
