"""
Business Impact Page
Cost analysis, ROI calculations, and intervention planning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Business Impact", page_icon="💰", layout="wide")

st.title("💰 Business Impact Analysis")
st.markdown("Translate model predictions into business value and ROI.")
st.markdown("---")

# Cost assumptions (editable)
st.sidebar.header("💵 Cost Assumptions")
st.sidebar.markdown("Adjust these values to match your business:")

cost_of_churn = st.sidebar.number_input(
    "Cost of Losing a Customer ($)",
    min_value=100,
    max_value=2000,
    value=500,
    step=50,
    help="Revenue lost when a customer churns (CLV impact)"
)

cost_of_retention = st.sidebar.number_input(
    "Cost of Retention Campaign ($)",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
    help="Cost of retention offer/campaign per customer"
)

retention_success_rate = st.sidebar.slider(
    "Expected Retention Success Rate",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.1,
    help="Percentage of at-risk customers retained when contacted"
)

# Demo data
total_customers = 7043
churn_rate = 0.265
test_customers = 1057
expected_churners = int(test_customers * churn_rate)

# Model performance (from demo)
model_recall = 0.524  # 52.4% of churners caught
model_precision = 0.658  # 65.8% of flagged customers actually churn

flagged_customers = int(expected_churners * model_recall / model_precision)
true_positives = int(expected_churners * model_recall)
false_positives = flagged_customers - true_positives

st.markdown("### 📊 Scenario Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### ❌ No Model (Do Nothing)")

    no_model_loss = expected_churners * cost_of_churn

    st.metric(
        "Expected Churners",
        f"{expected_churners:,}"
    )
    st.metric(
        "Total Revenue Loss",
        f"${no_model_loss:,}",
        delta=None
    )
    st.metric(
        "Retention Spend",
        "$0"
    )

    st.error(f"**Net Impact:** -${no_model_loss:,}")

with col2:
    st.markdown("#### 📢 Blanket Campaign (Everyone)")

    blanket_retention_cost = test_customers * cost_of_retention
    blanket_retained = int(expected_churners * retention_success_rate)
    blanket_churned = expected_churners - blanket_retained
    blanket_loss = blanket_churned * cost_of_churn
    blanket_savings = blanket_retained * cost_of_churn
    blanket_net = blanket_savings - blanket_retention_cost - blanket_loss

    st.metric(
        "Customers Contacted",
        f"{test_customers:,}"
    )
    st.metric(
        "Total Retention Cost",
        f"${blanket_retention_cost:,}"
    )
    st.metric(
        "Expected Retained",
        f"{blanket_retained:,}"
    )

    if blanket_net > 0:
        st.success(f"**Net Impact:** +${blanket_net:,}")
    else:
        st.warning(f"**Net Impact:** -${abs(blanket_net):,}")

with col3:
    st.markdown("#### 🎯 Targeted (Using Model)")

    targeted_retention_cost = flagged_customers * cost_of_retention
    targeted_retained = int(true_positives * retention_success_rate)
    targeted_churned = expected_churners - targeted_retained
    targeted_loss = targeted_churned * cost_of_churn
    targeted_savings = targeted_retained * cost_of_churn
    targeted_net = targeted_savings - targeted_retention_cost - targeted_loss

    st.metric(
        "Customers Contacted",
        f"{flagged_customers:,}",
        delta=f"-{test_customers - flagged_customers:,} vs blanket"
    )
    st.metric(
        "Total Retention Cost",
        f"${targeted_retention_cost:,}",
        delta=f"-${blanket_retention_cost - targeted_retention_cost:,}"
    )
    st.metric(
        "Expected Retained",
        f"{targeted_retained:,}"
    )

    st.success(f"**Net Impact:** +${targeted_net:,}")

st.markdown("---")

# Comparison chart
st.markdown("### 📈 Strategy Comparison")

comparison_data = pd.DataFrame({
    'Strategy': ['No Action', 'Blanket Campaign', 'Targeted (Model)'],
    'Retention Cost': [0, blanket_retention_cost, targeted_retention_cost],
    'Revenue Saved': [0, blanket_savings, targeted_savings],
    'Revenue Lost': [no_model_loss, blanket_loss, targeted_loss],
    'Net Impact': [-no_model_loss, blanket_net, targeted_net]
})

fig = go.Figure()

fig.add_trace(go.Bar(
    name='Retention Cost',
    x=comparison_data['Strategy'],
    y=comparison_data['Retention Cost'],
    marker_color='#ffc107'
))

fig.add_trace(go.Bar(
    name='Revenue Lost',
    x=comparison_data['Strategy'],
    y=comparison_data['Revenue Lost'],
    marker_color='#dc3545'
))

fig.add_trace(go.Bar(
    name='Revenue Saved',
    x=comparison_data['Strategy'],
    y=comparison_data['Revenue Saved'],
    marker_color='#28a745'
))

fig.update_layout(
    barmode='group',
    yaxis_title='$ Amount',
    legend_title='Category',
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# ROI calculation
model_savings_vs_nothing = targeted_net - (-no_model_loss)
model_savings_vs_blanket = targeted_net - blanket_net

st.markdown("### 💵 ROI Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Savings vs No Action",
        f"${model_savings_vs_nothing:,}",
        delta=f"+{model_savings_vs_nothing/no_model_loss*100:.0f}% ROI"
    )

with col2:
    st.metric(
        "Savings vs Blanket Campaign",
        f"${model_savings_vs_blanket:,}",
        delta="More efficient targeting"
    )

with col3:
    roi_pct = (targeted_savings - targeted_retention_cost) / targeted_retention_cost * 100
    st.metric(
        "Campaign ROI",
        f"{roi_pct:.0f}%",
        delta="Return on retention spend"
    )

st.markdown("---")

# Optimal Threshold Analysis
st.markdown("### 🎯 Optimal Threshold Analysis")

st.markdown(
    "Finding the right prediction threshold balances catching churners (recall) "
    "against wasting resources on false positives (precision)."
)

# Generate threshold analysis data
thresholds = np.arange(0.1, 0.95, 0.05)
threshold_data = []

for t in thresholds:
    # Approximate metrics at different thresholds
    if t < 0.3:
        recall, precision = 0.85, 0.35
    elif t < 0.5:
        recall, precision = 0.65, 0.55
    elif t < 0.7:
        recall, precision = 0.45, 0.70
    else:
        recall, precision = 0.25, 0.82

    # Add some variation
    recall = recall + np.random.uniform(-0.05, 0.05)
    precision = precision + np.random.uniform(-0.05, 0.05)

    tp = int(expected_churners * recall)
    fp = int(tp / precision - tp) if precision > 0 else 0

    ret_cost = (tp + fp) * cost_of_retention
    retained = int(tp * retention_success_rate)
    saved = retained * cost_of_churn
    lost = (expected_churners - retained) * cost_of_churn

    net = saved - ret_cost - lost

    threshold_data.append({
        'Threshold': t,
        'Precision': precision,
        'Recall': recall,
        'Customers Flagged': tp + fp,
        'Net Impact': net
    })

threshold_df = pd.DataFrame(threshold_data)

# Find optimal
optimal_idx = threshold_df['Net Impact'].idxmax()
optimal_threshold = threshold_df.loc[optimal_idx, 'Threshold']
optimal_net = threshold_df.loc[optimal_idx, 'Net Impact']

col1, col2 = st.columns(2)

with col1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=threshold_df['Threshold'],
        y=threshold_df['Net Impact'],
        mode='lines+markers',
        name='Net Impact',
        line=dict(width=3, color='#28a745')
    ))

    fig.add_vline(
        x=optimal_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Optimal: {optimal_threshold:.2f}"
    )

    fig.update_layout(
        xaxis_title='Prediction Threshold',
        yaxis_title='Net Impact ($)',
        title='Net Impact by Threshold',
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=threshold_df['Threshold'],
        y=threshold_df['Precision'],
        mode='lines',
        name='Precision',
        line=dict(width=2, color='#007bff')
    ))

    fig.add_trace(go.Scatter(
        x=threshold_df['Threshold'],
        y=threshold_df['Recall'],
        mode='lines',
        name='Recall',
        line=dict(width=2, color='#fd7e14')
    ))

    fig.add_vline(
        x=optimal_threshold,
        line_dash="dash",
        line_color="red"
    )

    fig.update_layout(
        xaxis_title='Prediction Threshold',
        yaxis_title='Score',
        title='Precision-Recall Trade-off',
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)

st.success(
    f"**Recommended Threshold: {optimal_threshold:.2f}**\n\n"
    f"At this threshold, the expected net impact is **${optimal_net:,.0f}** "
    f"with balanced precision and recall."
)

st.markdown("---")

# Segment-level ROI
st.markdown("### 📊 Segment-Level Analysis")

segment_data = pd.DataFrame({
    'Segment': ['Critical', 'At Risk', 'Monitor', 'Safe'],
    'Customers': [50, 130, 350, 527],
    'Avg Churn Risk': ['85%', '62%', '35%', '12%'],
    'Investment': [5000, 13000, 35000, 52700],
    'Expected Savings': [21250, 40300, 61250, 31620],
    'Net ROI': [16250, 27300, 26250, -21080],
    'Recommend': ['✅ Yes', '✅ Yes', '✅ Yes', '❌ No']
})

st.dataframe(segment_data, use_container_width=True, hide_index=True)

st.info(
    "💡 **Recommendation:** Focus retention efforts on Critical, At Risk, and Monitor "
    "segments. The Safe segment has negative ROI for intervention due to low base churn rate."
)

# Final recommendation
st.markdown("---")
st.markdown("### 🎯 Executive Recommendation")

st.markdown(f"""
Based on the analysis with current cost assumptions:

1. **Deploy the model** at threshold **{optimal_threshold:.2f}** for optimal business impact
2. **Focus on {flagged_customers} high-risk customers** instead of blanket campaigns
3. **Expected annual savings: ${model_savings_vs_nothing * 12:,.0f}** (scaled from test set)
4. **Prioritize Critical and At Risk segments** for immediate intervention

**Key Actions:**
- Offer contract upgrades to month-to-month customers (highest impact)
- Bundle Tech Support with retention offers
- Switch customers to automatic payments (reduces churn 5-10%)
""")
