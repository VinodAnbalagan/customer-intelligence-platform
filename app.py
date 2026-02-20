"""
Customer Intelligence Platform - Main Streamlit App
End-to-end churn prediction with interactive dashboard.
"""

import streamlit as st

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("📊 Customer Intelligence Platform")
st.markdown("### AI-Powered Churn Prediction & Customer Analytics")

st.markdown("""
---

Welcome to the **Customer Intelligence Platform**, an end-to-end machine learning system
for predicting customer churn and enabling data-driven retention strategies.

### 🎯 What This Platform Does

- **Identifies at-risk customers** before they churn
- **Quantifies business impact** of retention interventions
- **Provides actionable insights** for customer success teams
- **Enables "What-If" analysis** to optimize retention strategies

### 📑 Navigate the Dashboard

Use the sidebar to explore different sections:

| Page | Description |
|------|-------------|
| **Executive Summary** | Key metrics, churn drivers, and risk distribution |
| **Customer Risk Explorer** | Browse and filter customers by risk level |
| **What-If Simulator** | Test how changes affect churn probability |
| **Model Performance** | ROC curves, confusion matrices, and calibration |
| **Business Impact** | Cost analysis and ROI calculations |

### 🔬 Technical Details

- **Model:** XGBoost with class weighting for imbalanced data
- **Dataset:** Telco Customer Churn (7,043 customers)
- **Features:** 21 original + 13 engineered features
- **Explainability:** SHAP values for global and local explanations

---

*Built with Streamlit, XGBoost, and SHAP*
""")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.info(
    "Use the pages above to explore the platform. "
    "Start with the Executive Summary for a high-level overview."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "This platform demonstrates production-grade "
    "data science for customer churn prediction."
)
