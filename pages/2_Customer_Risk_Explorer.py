"""
Customer Risk Explorer Page
Browse and filter customers by risk level with SHAP explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Customer Risk Explorer", page_icon="CRE", layout="wide")

st.title("Customer Risk Explorer")
st.markdown("Browse customers by risk level and view individual explanations.")
st.markdown("---")


@st.cache_data
def load_demo_customers():
    """Load demo customer data."""
    np.random.seed(42)
    n_customers = 500

    # Generate demo data
    data = {
        'CustomerID': [f'CUST-{i:04d}' for i in range(n_customers)],
        'tenure': np.random.randint(1, 72, n_customers),
        'MonthlyCharges': np.random.uniform(20, 120, n_customers),
        'TotalCharges': np.random.uniform(100, 8000, n_customers),
        'Contract': np.random.choice(
            ['Month-to-month', 'One year', 'Two year'],
            n_customers,
            p=[0.55, 0.25, 0.20]
        ),
        'InternetService': np.random.choice(
            ['DSL', 'Fiber optic', 'No'],
            n_customers,
            p=[0.35, 0.45, 0.20]
        ),
        'TechSupport': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
        'OnlineSecurity': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
    }

    df = pd.DataFrame(data)

    # Generate churn probabilities based on features
    base_prob = 0.3
    df['ChurnProbability'] = base_prob

    # Adjust based on contract
    df.loc[df['Contract'] == 'Month-to-month', 'ChurnProbability'] += 0.25
    df.loc[df['Contract'] == 'Two year', 'ChurnProbability'] -= 0.15

    # Adjust based on tenure
    df['ChurnProbability'] -= df['tenure'] * 0.003

    # Adjust based on support
    df.loc[df['TechSupport'] == 'No', 'ChurnProbability'] += 0.1
    df.loc[df['OnlineSecurity'] == 'No', 'ChurnProbability'] += 0.08

    # Add noise and clip
    df['ChurnProbability'] += np.random.uniform(-0.1, 0.1, n_customers)
    df['ChurnProbability'] = df['ChurnProbability'].clip(0.02, 0.98)

    # Assign segments
    df['Segment'] = pd.cut(
        df['ChurnProbability'],
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=['Safe', 'Monitor', 'At Risk', 'Critical']
    )

    # Top risk factors (mock SHAP)
    df['TopRiskFactor1'] = np.where(
        df['Contract'] == 'Month-to-month',
        'Contract: Month-to-month',
        np.where(df['TechSupport'] == 'No', 'No Tech Support', 'Low tenure')
    )
    df['TopRiskFactor2'] = np.where(
        df['OnlineSecurity'] == 'No',
        'No Online Security',
        'High Monthly Charges'
    )

    return df


customers_df = load_demo_customers()

# Sidebar filters
st.sidebar.header("Filters")

# Segment filter
segments = st.sidebar.multiselect(
    "Risk Segment",
    options=['Safe', 'Monitor', 'At Risk', 'Critical'],
    default=['At Risk', 'Critical']
)

# Contract filter
contracts = st.sidebar.multiselect(
    "Contract Type",
    options=['Month-to-month', 'One year', 'Two year'],
    default=['Month-to-month', 'One year', 'Two year']
)

# Tenure range
tenure_range = st.sidebar.slider(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=(0, 72)
)

# Apply filters
filtered_df = customers_df[
    (customers_df['Segment'].isin(segments)) &
    (customers_df['Contract'].isin(contracts)) &
    (customers_df['tenure'] >= tenure_range[0]) &
    (customers_df['tenure'] <= tenure_range[1])
]

# Summary metrics
st.markdown("### Filtered Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Customers Shown", f"{len(filtered_df):,}")
with col2:
    avg_risk = filtered_df['ChurnProbability'].mean() * 100
    st.metric("Avg Churn Risk", f"{avg_risk:.1f}%")
with col3:
    critical_count = len(filtered_df[filtered_df['Segment'] == 'Critical'])
    st.metric("Critical Customers", f"{critical_count:,}")
with col4:
    mtm_count = len(filtered_df[filtered_df['Contract'] == 'Month-to-month'])
    st.metric("Month-to-Month", f"{mtm_count:,}")

st.markdown("---")

# Color coding function
def color_segment(val):
    colors = {
        'Safe': 'background-color: #d4edda',
        'Monitor': 'background-color: #fff3cd',
        'At Risk': 'background-color: #ffe5d0',
        'Critical': 'background-color: #f8d7da'
    }
    return colors.get(val, '')


def color_probability(val):
    if val >= 0.8:
        return 'background-color: #f8d7da; color: #721c24'
    elif val >= 0.5:
        return 'background-color: #ffe5d0; color: #856404'
    elif val >= 0.2:
        return 'background-color: #fff3cd; color: #856404'
    else:
        return 'background-color: #d4edda; color: #155724'


# Display table
st.markdown("### Customer List")

# Sortable columns
sort_by = st.selectbox(
    "Sort by",
    options=['ChurnProbability', 'tenure', 'MonthlyCharges'],
    index=0
)
sort_order = st.radio("Order", ['Descending', 'Ascending'], horizontal=True)

sorted_df = filtered_df.sort_values(
    sort_by,
    ascending=(sort_order == 'Ascending')
)

# Display columns
display_cols = [
    'CustomerID', 'ChurnProbability', 'Segment', 'Contract',
    'tenure', 'MonthlyCharges', 'TechSupport', 'OnlineSecurity',
    'TopRiskFactor1', 'TopRiskFactor2'
]

styled_df = sorted_df[display_cols].head(100).style.map(
    color_probability, subset=['ChurnProbability']
).map(
    color_segment, subset=['Segment']
).format({
    'ChurnProbability': '{:.1%}',
    'MonthlyCharges': '${:.2f}'
})

st.dataframe(styled_df, use_container_width=True, height=400)

st.markdown("---")

# Individual customer detail
st.markdown("### Individual Customer Analysis")

selected_customer = st.selectbox(
    "Select a customer to view detailed explanation",
    options=sorted_df['CustomerID'].tolist()
)

if selected_customer:
    customer = sorted_df[sorted_df['CustomerID'] == selected_customer].iloc[0]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Customer Profile")

        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=customer['ChurnProbability'] * 100,
            title={'text': "Churn Risk"},
            number={'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 20], 'color': "#d4edda"},
                    {'range': [20, 50], 'color': "#fff3cd"},
                    {'range': [50, 80], 'color': "#ffe5d0"},
                    {'range': [80, 100], 'color': "#f8d7da"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': customer['ChurnProbability'] * 100
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Segment:** {customer['Segment']}")
        st.markdown(f"**Contract:** {customer['Contract']}")
        st.markdown(f"**Tenure:** {customer['tenure']} months")
        st.markdown(f"**Monthly Charges:** ${customer['MonthlyCharges']:.2f}")

    with col2:
        st.markdown("#### Risk Factors (SHAP Explanation)")

        # Mock SHAP waterfall data
        factors = pd.DataFrame({
            'Factor': [
                'Contract: Month-to-month' if customer['Contract'] == 'Month-to-month' else f"Contract: {customer['Contract']}",
                f"Tenure: {customer['tenure']} months",
                f"TechSupport: {customer['TechSupport']}",
                f"OnlineSecurity: {customer['OnlineSecurity']}",
                f"InternetService: {customer['InternetService']}"
            ],
            'Impact': [
                0.25 if customer['Contract'] == 'Month-to-month' else -0.15,
                -0.01 * customer['tenure'],
                0.10 if customer['TechSupport'] == 'No' else -0.05,
                0.08 if customer['OnlineSecurity'] == 'No' else -0.03,
                0.05 if customer['InternetService'] == 'Fiber optic' else -0.02
            ]
        })

        fig = px.bar(
            factors,
            x='Impact',
            y='Factor',
            orientation='h',
            color='Impact',
            color_continuous_scale='RdYlGn_r',
            range_color=[-0.3, 0.3]
        )
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title="Impact on Churn Probability",
            yaxis_title="",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.markdown("#### Recommended Actions")

        if customer['Contract'] == 'Month-to-month':
            st.success("**Offer contract upgrade** - 1-year contract could reduce risk by ~30%")
        if customer['TechSupport'] == 'No':
            st.info("**Add Tech Support** - Customers with support churn 15% less")
        if customer['OnlineSecurity'] == 'No':
            st.info("**Add Online Security** - Improves customer stickiness")
        if customer['tenure'] < 12:
            st.warning("**New customer** - Prioritize early engagement")
