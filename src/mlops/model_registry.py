"""
MLflow Model Registry Module
Registers and versions production models.
"""

import mlflow
from mlflow.tracking import MlflowClient
from typing import Any, Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def register_model(
    run_id: str,
    model_name: str = "churn_prediction_model",
    description: Optional[str] = None
) -> str:
    """
    Register a model from a run to the model registry.

    Parameters
    ----------
    run_id : str
        MLflow run ID containing the model.
    model_name : str
        Name for the registered model.
    description : str, optional
        Model description.

    Returns
    -------
    str
        Model version.
    """
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri, model_name)

    if description:
        client = MlflowClient()
        client.update_model_version(
            name=model_name,
            version=result.version,
            description=description
        )

    logger.info(f"Registered model: {model_name} v{result.version}")

    return result.version


def transition_model_stage(
    model_name: str,
    version: str,
    stage: str = "Production"
) -> None:
    """
    Transition a model version to a new stage.

    Parameters
    ----------
    model_name : str
        Name of the registered model.
    version : str
        Version to transition.
    stage : str
        Target stage: 'Staging', 'Production', 'Archived'.
    """
    client = MlflowClient()

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )

    logger.info(f"Transitioned {model_name} v{version} to {stage}")


def load_production_model(model_name: str = "churn_prediction_model") -> Any:
    """
    Load the production version of a model.

    Returns
    -------
    model
        The loaded model.
    """
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pyfunc.load_model(model_uri)

    logger.info(f"Loaded production model: {model_name}")

    return model


def get_model_info(model_name: str) -> Dict:
    """
    Get information about a registered model.

    Returns
    -------
    Dict
        Model metadata and versions.
    """
    client = MlflowClient()

    try:
        model = client.get_registered_model(model_name)
    except Exception:
        return {'exists': False}

    versions = []
    for mv in model.latest_versions:
        versions.append({
            'version': mv.version,
            'stage': mv.current_stage,
            'status': mv.status,
            'run_id': mv.run_id,
            'description': mv.description
        })

    return {
        'exists': True,
        'name': model.name,
        'description': model.description,
        'versions': versions,
        'latest_versions': {
            mv.current_stage: mv.version
            for mv in model.latest_versions
        }
    }


def create_model_card(
    model_name: str,
    model_info: Dict,
    metrics: Dict,
    save_path: str
) -> str:
    """
    Create a model card (documentation).

    Parameters
    ----------
    model_name : str
        Name of the model.
    model_info : Dict
        Model metadata.
    metrics : Dict
        Performance metrics.
    save_path : str
        Path to save the model card.

    Returns
    -------
    str
        Path to the saved model card.
    """
    card = f"""# Model Card: {model_name}

## Model Description

**Model Type:** XGBoost Classifier
**Task:** Binary Classification (Customer Churn Prediction)
**Version:** {model_info.get('latest_versions', {}).get('Production', 'N/A')}

### Intended Use

This model predicts the probability that a telecom customer will churn (cancel their subscription) based on their account characteristics, usage patterns, and demographic information.

**Primary Use Cases:**
- Identify at-risk customers for proactive retention campaigns
- Calculate optimal intervention thresholds for cost-effective retention
- Provide customer service representatives with churn risk scores

**Users:**
- Marketing and retention teams
- Customer service departments
- Business intelligence analysts

## Training Data

**Dataset:** Telco Customer Churn (Kaggle)
**Size:** 7,043 customers
**Features:** 21 original features + engineered features
**Class Distribution:** ~26.5% churn (imbalanced)

### Data Processing
- Missing values in TotalCharges filled with MonthlyCharges
- Categorical features label-encoded
- Numerical features standardized
- Class imbalance handled via class weighting

## Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics.get('accuracy', 'N/A'):.4f} |
| Precision | {metrics.get('precision', 'N/A'):.4f} |
| Recall | {metrics.get('recall', 'N/A'):.4f} |
| F1 Score | {metrics.get('f1', 'N/A'):.4f} |
| AUC-ROC | {metrics.get('roc_auc', 'N/A'):.4f} |
| AUC-PR | {metrics.get('pr_auc', 'N/A'):.4f} |

**Evaluation Dataset:** 15% held-out test set (stratified)

## Limitations

1. **Data Limitations:**
   - Model trained on single telecom provider's data
   - Historical data may not reflect current market conditions
   - Limited demographic features available

2. **Performance Limitations:**
   - Precision-recall trade-off: optimized for recall (catching churners)
   - May have false positives leading to unnecessary retention spend
   - Performance may vary across customer segments

3. **Deployment Limitations:**
   - Requires all input features to be present
   - Predictions should be refreshed as customer data changes

## Ethical Considerations

1. **Bias Assessment:**
   - Model uses SeniorCitizen as a feature, which may correlate with age discrimination
   - Geographic or socioeconomic biases possible through proxy features
   - Recommend regular bias audits across protected groups

2. **Privacy:**
   - Model uses PII (customer account data)
   - Ensure compliance with data protection regulations (GDPR, CCPA)
   - Implement appropriate access controls

3. **Fairness:**
   - Retention offers should be equitable across customer segments
   - Monitor for disparate impact in intervention decisions

## Maintenance

- **Retraining Frequency:** Quarterly or when performance degrades >5%
- **Monitoring:** Track prediction distribution drift and actual churn rates
- **Feedback Loop:** Incorporate retention campaign outcomes

## References

- Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/

---

*Generated by Customer Intelligence Platform*
"""

    with open(save_path, 'w') as f:
        f.write(card)

    logger.info(f"Created model card at {save_path}")

    return save_path


if __name__ == "__main__":
    # Example usage
    info = get_model_info("churn_prediction_model")
    print(f"Model exists: {info.get('exists', False)}")

    if info.get('exists'):
        print(f"Versions: {info.get('versions', [])}")
