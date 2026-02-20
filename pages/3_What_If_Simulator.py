"""
What-If Simulator Page
Interactive tool to test how changes affect churn probability.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="What-If Simulator", page_icon="🔮", layout="wide")

st.title("🔮 What-If Simulator")
st.markdown(
    "Test how different interventions affect a customer's churn probability. "
    "Adjust the inputs and see the impact in real-time."
)
st.markdown("---")


def calculate_churn_probability(features):
    """
    Calculate churn probability based on features.
    This is a simplified model for demo purposes.
    In production, this would call the actual trained model.
    """
    base_prob = 0.30

    # Contract impact (strongest predictor)
    contract_impact = {
        'Month-to-month': 0.25,
        'One year': 0.0,
        'Two year': -0.15
    }
    prob = base_prob + contract_impact.get(features['contract'], 0)

    # Tenure impact
    prob -= features['tenure'] * 0.004

    # Monthly charges impact
    if features['monthly_charges'] > 70:
        prob += 0.05

    # Support services impact
    if features['tech_support'] == 'No':
        prob += 0.10
    if features['online_security'] == 'No':
        prob += 0.08

    # Internet service impact
    if features['internet_service'] == 'Fiber optic':
        prob += 0.05
    elif features['internet_service'] == 'No':
        prob -= 0.10

    # Payment method impact
    if features['payment_method'] == 'Electronic check':
        prob += 0.05
    elif 'automatic' in features['payment_method'].lower():
        prob -= 0.05

    # Paperless billing
    if features['paperless_billing'] == 'Yes':
        prob += 0.03

    # Clip to valid range
    return np.clip(prob, 0.02, 0.98)


# Create two columns: inputs and results
col_input, col_result = st.columns([1, 1])

with col_input:
    st.markdown("### 📝 Customer Profile")

    # Contract type (most impactful)
    contract = st.selectbox(
        "Contract Type",
        options=['Month-to-month', 'One year', 'Two year'],
        index=0,
        help="Contract type is the #1 predictor of churn"
    )

    # Tenure
    tenure = st.slider(
        "Tenure (months)",
        min_value=1,
        max_value=72,
        value=12,
        help="How long the customer has been with the company"
    )

    # Monthly charges
    monthly_charges = st.slider(
        "Monthly Charges ($)",
        min_value=20.0,
        max_value=120.0,
        value=70.0,
        step=5.0
    )

    # Internet service
    internet_service = st.selectbox(
        "Internet Service",
        options=['DSL', 'Fiber optic', 'No'],
        index=1
    )

    # Tech support
    tech_support = st.selectbox(
        "Tech Support",
        options=['Yes', 'No'],
        index=1,
        help="Tech Support reduces churn risk"
    )

    # Online security
    online_security = st.selectbox(
        "Online Security",
        options=['Yes', 'No'],
        index=1
    )

    # Payment method
    payment_method = st.selectbox(
        "Payment Method",
        options=[
            'Electronic check',
            'Mailed check',
            'Bank transfer (automatic)',
            'Credit card (automatic)'
        ],
        index=0
    )

    # Paperless billing
    paperless_billing = st.selectbox(
        "Paperless Billing",
        options=['Yes', 'No'],
        index=0
    )

# Calculate current probability
current_features = {
    'contract': contract,
    'tenure': tenure,
    'monthly_charges': monthly_charges,
    'internet_service': internet_service,
    'tech_support': tech_support,
    'online_security': online_security,
    'payment_method': payment_method,
    'paperless_billing': paperless_billing
}

current_prob = calculate_churn_probability(current_features)

with col_result:
    st.markdown("### 📊 Prediction Results")

    # Main risk gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=current_prob * 100,
        title={'text': "Churn Probability", 'font': {'size': 24}},
        number={'suffix': '%', 'font': {'size': 48}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkred" if current_prob > 0.5 else "darkorange" if current_prob > 0.2 else "darkgreen"},
            'steps': [
                {'range': [0, 20], 'color': "#d4edda"},
                {'range': [20, 50], 'color': "#fff3cd"},
                {'range': [50, 80], 'color': "#ffe5d0"},
                {'range': [80, 100], 'color': "#f8d7da"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': current_prob * 100
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Risk segment
    if current_prob >= 0.8:
        st.error("🚨 **CRITICAL RISK** - Immediate intervention required")
    elif current_prob >= 0.5:
        st.warning("⚠️ **AT RISK** - Proactive retention recommended")
    elif current_prob >= 0.2:
        st.info("📊 **MONITOR** - Light engagement suggested")
    else:
        st.success("✅ **SAFE** - Low churn risk")

st.markdown("---")

# What-if scenarios
st.markdown("### 🎯 What-If Scenarios")
st.markdown("See how specific changes would affect this customer's churn risk:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📄 Contract Upgrade")

    # Calculate impact of contract changes
    for new_contract in ['Month-to-month', 'One year', 'Two year']:
        test_features = current_features.copy()
        test_features['contract'] = new_contract
        new_prob = calculate_churn_probability(test_features)
        delta = (new_prob - current_prob) * 100

        if new_contract == contract:
            st.markdown(f"**{new_contract}** (current): {new_prob:.1%}")
        else:
            color = "green" if delta < 0 else "red"
            st.markdown(
                f"**{new_contract}**: {new_prob:.1%} "
                f"(<span style='color:{color}'>{delta:+.1f}%</span>)",
                unsafe_allow_html=True
            )

with col2:
    st.markdown("#### 🛠️ Add Services")

    # Tech support impact
    test_features = current_features.copy()
    test_features['tech_support'] = 'Yes'
    new_prob = calculate_churn_probability(test_features)
    delta = (new_prob - current_prob) * 100
    if tech_support == 'No':
        st.markdown(
            f"**Add Tech Support**: {new_prob:.1%} "
            f"(<span style='color:green'>{delta:+.1f}%</span>)",
            unsafe_allow_html=True
        )
    else:
        st.markdown("✅ Tech Support already active")

    # Online security impact
    test_features = current_features.copy()
    test_features['online_security'] = 'Yes'
    new_prob = calculate_churn_probability(test_features)
    delta = (new_prob - current_prob) * 100
    if online_security == 'No':
        st.markdown(
            f"**Add Online Security**: {new_prob:.1%} "
            f"(<span style='color:green'>{delta:+.1f}%</span>)",
            unsafe_allow_html=True
        )
    else:
        st.markdown("✅ Online Security already active")

with col3:
    st.markdown("#### 💳 Payment Method")

    # Auto-payment impact
    for pm in ['Credit card (automatic)', 'Bank transfer (automatic)']:
        test_features = current_features.copy()
        test_features['payment_method'] = pm
        new_prob = calculate_churn_probability(test_features)
        delta = (new_prob - current_prob) * 100

        if pm == payment_method:
            st.markdown(f"✅ {pm} (current)")
        else:
            color = "green" if delta < 0 else "red"
            st.markdown(
                f"**{pm.split('(')[0].strip()}**: {new_prob:.1%} "
                f"(<span style='color:{color}'>{delta:+.1f}%</span>)",
                unsafe_allow_html=True
            )

st.markdown("---")

# Best intervention recommendation
st.markdown("### 💡 Optimal Intervention")

# Calculate best single intervention
best_intervention = None
best_reduction = 0

interventions = [
    ('Upgrade to Two year contract', {'contract': 'Two year'}),
    ('Upgrade to One year contract', {'contract': 'One year'}),
    ('Add Tech Support', {'tech_support': 'Yes'}),
    ('Add Online Security', {'online_security': 'Yes'}),
    ('Switch to automatic payment', {'payment_method': 'Credit card (automatic)'})
]

for name, changes in interventions:
    test_features = current_features.copy()
    test_features.update(changes)
    new_prob = calculate_churn_probability(test_features)
    reduction = current_prob - new_prob

    if reduction > best_reduction:
        best_reduction = reduction
        best_intervention = (name, new_prob, reduction)

if best_intervention and best_reduction > 0.01:
    name, new_prob, reduction = best_intervention

    col1, col2 = st.columns([2, 1])

    with col1:
        st.success(
            f"**Recommended:** {name}\n\n"
            f"This would reduce churn risk from **{current_prob:.1%}** to **{new_prob:.1%}** "
            f"(a **{reduction*100:.1f} percentage point** reduction)."
        )

    with col2:
        # Before/after comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Before', 'After'],
            y=[current_prob * 100, new_prob * 100],
            marker_color=['#dc3545', '#28a745'],
            text=[f'{current_prob:.1%}', f'{new_prob:.1%}'],
            textposition='outside'
        ))
        fig.update_layout(
            yaxis_title="Churn Probability (%)",
            yaxis_range=[0, 100],
            showlegend=False,
            height=250
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("This customer already has optimal settings or is inherently low-risk.")
