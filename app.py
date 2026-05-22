import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

st.set_page_config(
    page_title="Olist Customer Analytics",
    page_icon="🛒",
    layout="wide"
)

@st.cache_data
def load_data():
    cohort = pd.read_csv('cohort_retention.csv')
    rfm = pd.read_csv('rfm_segments.csv')
    funnel = pd.read_csv('funnel.csv')
    return cohort, rfm, funnel

@st.cache_resource
def load_model():
    model = joblib.load('churn_model.pkl')
    encoders = joblib.load('encoders.pkl')
    with open('cat_values.json') as f:
        cat_values = json.load(f)
    with open('window_lookup.json') as f:
        windows = json.load(f)
    return model, encoders, cat_values, windows

cohort, rfm, funnel = load_data()
model, encoders, cat_values, windows = load_model()

tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "🔮 Return Probability Predictor"])

with tab1:
    st.title("Olist E-Commerce — Customer Analytics")
    st.markdown("SQL-driven analysis of 93,000+ customers across cohort retention, RFM segmentation, and order funnel.")

    st.header("Cohort Retention Analysis")
    st.markdown("Each row is a group of customers who made their first purchase in that month. Values show % who returned in subsequent months.")

    cohort['cohort_month'] = pd.to_datetime(cohort['cohort_month']).dt.strftime('%Y-%m')
    cohort_pivot = cohort.pivot_table(
        index='cohort_month',
        columns='month_number',
        values='retention_pct'
    )

    fig1, ax1 = plt.subplots(figsize=(16, 8))
    sns.heatmap(cohort_pivot, annot=True, fmt='.1f', cmap='YlOrRd',
                linewidths=0.5, mask=cohort_pivot.isnull(), ax=ax1)
    ax1.set_title('Customer Cohort Retention (%)', fontsize=13)
    ax1.set_xlabel('Months Since First Purchase')
    ax1.set_ylabel('Cohort Month')
    st.pyplot(fig1)
    plt.close()

    st.markdown("**Key insight:** Retention drops below 1% immediately after first purchase across all cohorts — consistent with Olist's durable goods category mix where repurchase cycles are naturally long.")

    st.header("RFM Customer Segmentation")
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    colors_map = {'Champion':'#2ecc71','At Risk':'#e74c3c',
                  'Promising':'#3498db','Churned':'#95a5a6','Needs Attention':'#f39c12'}

    for i, row in rfm.iterrows():
        with cols[i]:
            st.metric(
                label=row['segment'],
                value=f"{row['customer_count']:,}",
                delta=f"{row['pct']}% of base"
            )

    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(14, 5))
    bar_colors = [colors_map[s] for s in rfm['segment']]
    ax2.barh(rfm['segment'], rfm['customer_count'], color=bar_colors)
    ax2.set_xlabel('Number of Customers')
    ax2.set_title('Segment Sizes')

    scatter_colors = [colors_map[s] for s in rfm['segment']]
    ax3.scatter(rfm['avg_recency'], rfm['avg_spend'],
                c=scatter_colors, s=rfm['customer_count']/50, alpha=0.7)
    for _, row in rfm.iterrows():
        ax3.annotate(row['segment'], (row['avg_recency'], row['avg_spend']),
                    textcoords='offset points', xytext=(5,5), fontsize=9)
    ax3.set_xlabel('Avg Days Since Last Purchase')
    ax3.set_ylabel('Avg Spend (R$)')
    ax3.set_title('Segment Value Map')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("**Key insight:** At Risk customers (1.1% of base) have the highest average spend at R$295. Despite their small size, they represent the highest-value win-back opportunity given sufficient time has passed for durable goods repurchase cycles to complete.")

    st.header("Order Funnel Analysis")
    stages = ['Placed', 'Approved', 'Dispatched', 'Delivered']
    values = [funnel['total_orders_placed'].values[0],
              funnel['total_orders_approved'].values[0],
              funnel['total_orders_dispatched'].values[0],
              funnel['total_orders_delivered'].values[0]]

    fig3, ax4 = plt.subplots(figsize=(10, 4))
    colors_funnel = ['#3498db','#2ecc71','#f39c12','#27ae60']
    bars = ax4.barh(stages, values, color=colors_funnel)
    for bar, val in zip(bars, values):
        pct = val / values[0] * 100
        ax4.text(bar.get_width() - 1000, bar.get_y() + bar.get_height()/2,
                f'{val:,} ({pct:.1f}%)', va='center', ha='right',
                color='white', fontweight='bold')
    ax4.set_xlabel('Number of Orders')
    ax4.set_title('Order Funnel — Stage by Stage Drop-off')
    st.pyplot(fig3)
    plt.close()

    st.markdown("**Key insight:** 97% end-to-end delivery rate confirms operations are not the retention problem. The churn challenge is re-engagement, not fulfilment.")

with tab2:
    st.title("Return Probability Predictor")
    st.markdown("Enter a customer's first order details to predict likelihood of making a second purchase.")

    col1, col2 = st.columns(2)

    with col1:
        state = st.selectbox("Customer State", cat_values['customer_state'])
        category = st.selectbox("Product Category", cat_values['first_order_category'])
        payment = st.selectbox("Payment Type", cat_values['payment_type'])
        review = st.slider("Review Score", 1.0, 5.0, 4.0, 0.5)

    with col2:
        spend = st.number_input("First Order Spend (R$)", min_value=0.0, value=150.0)
        items = st.number_input("Number of Items", min_value=1, value=1)
        delay = st.number_input("Delivery Delay (days, negative = early)",
                                 value=-3.0, step=0.5)
        days_since = st.number_input("Days Since Purchase", min_value=1, value=90)

    if st.button("Predict Return Probability", type="primary"):
        expected = windows['lookup'].get(category, windows['global_avg'])
        recency_ratio = days_since / expected

        input_data = pd.DataFrame({
            'customer_state': [state],
            'first_order_spend': [spend],
            'items_in_order': [items],
            'payment_type': [payment],
            'delivery_delay_days': [delay],
            'first_order_category': [category],
            'review_score': [review],
            'recency_ratio': [recency_ratio]
        })

        for col in ['customer_state', 'payment_type', 'first_order_category']:
            try:
                input_data[col] = encoders[col].transform(input_data[col])
            except ValueError:
                input_data[col] = 0

        prob = model.predict_proba(input_data)[0][1]

        st.markdown("---")
        col_result, col_context = st.columns(2)

        with col_result:
            st.metric(
                label="Return Probability",
                value=f"{prob:.1%}",
                delta=f"{(prob - 0.032):.1%} vs base rate"
            )
            if prob > 0.15:
                st.success("High likelihood — worth targeting with retention campaign")
            elif prob > 0.07:
                st.warning("Moderate likelihood — consider low-cost nudge")
            else:
                st.error("Low likelihood — below base rate, deprioritise")

        with col_context:
            st.markdown(f"**Category expected repurchase window:** {expected:.0f} days")
            st.markdown(f"**Recency ratio:** {recency_ratio:.2f}x")
            st.markdown(f"**Base rate (population avg):** 3.2%")
            if recency_ratio > 1:
                st.markdown(f"⚠️ Customer is {recency_ratio:.1f}x past expected repurchase window")
            else:
                st.markdown(f"✓ Customer is within normal repurchase window")