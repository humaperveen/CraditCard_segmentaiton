#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import pickle

with open("kmeans_model.pkl", "rb") as file:
    kmeans_model = pickle.load(file)

st.title("Customer Segmentation with K-means Clustering")

st.header("User Input:")
balance = st.slider("Balance", min_value=0.0, max_value=20000.0, step=100.0, value=1000.0)
purchases = st.slider("Purchases", min_value=0.0, max_value=50000.0, step=100.0, value=1000.0)
oneoff_purchases = st.slider("One-Off Purchases", min_value=0.0, max_value=50000.0, step=100.0, value=1000.0)
installments_purchases = st.slider("Installments Purchases", min_value=0.0, max_value=30000.0, step=100.0, value=1000.0)
cash_advance = st.slider("Cash Advance", min_value=0.0, max_value=50000.0, step=100.0, value=1000.0)
credit_limit = st.slider("Credit Limit", min_value=0.0, max_value=30000.0, step=100.0, value=10000.0)
payments = st.slider("Payments", min_value=0.0, max_value=50000.0, step=100.0, value=1000.0)
prc_full_payment = st.slider("Percentage of Full Payment", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
tenure = st.slider("Tenure", min_value=0, max_value=20, step=1, value=10)
credit_utilization = st.slider("Credit Utilization", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

user_input = {
    'BALANCE': balance,
    'PURCHASES': purchases,
    'ONEOFF_PURCHASES': oneoff_purchases,
    'INSTALLMENTS_PURCHASES': installments_purchases,
    'CASH_ADVANCE': cash_advance,
    'CREDIT_LIMIT': credit_limit,
    'PAYMENTS': payments,
    'PRC_FULL_PAYMENT': prc_full_payment,
    'TENURE': tenure,
    'CREDIT_UTILIZATION': credit_utilization
}

def perform_clustering(user_input):
    def scale_user_input(user_input):
        feature_ranges = {
            'BALANCE': (0.0, 20000.0),
            'PURCHASES': (0.0, 50000.0),
            'ONEOFF_PURCHASES': (0.0, 50000.0),
            'INSTALLMENTS_PURCHASES': (0.0, 30000.0),
            'CASH_ADVANCE': (0.0, 50000.0),
            'CREDIT_LIMIT': (0.0, 30000.0),
            'PAYMENTS': (0.0, 50000.0),
            'PRC_FULL_PAYMENT': (0.0, 1.0),
            'TENURE': (0, 20),
            'CREDIT_UTILIZATION': (0.0, 1.0)
        }

        scaled_user_input = {feature: (user_input[feature] - min_val) / (max_val - min_val)
                              for feature, (min_val, max_val) in feature_ranges.items()}

        return pd.DataFrame(scaled_user_input, index=[0])

    user_df_scaled = scale_user_input(user_input)

    cluster = kmeans_model.predict(user_df_scaled)[0]
    return cluster

def get_segment_description(cluster):
    segment_descriptions = {
        0: "Segment 0: These customers have a moderate balance, make regular purchases, and use credit moderately.",
        1: "Segment 1: These customers have a high balance, make frequent purchases, and utilize credit extensively.",
    }
    return segment_descriptions.get(cluster, "Undefined Segment")

def get_recommendations(cluster):
    recommendations = {
        0: "Recommendation for Segment 0: Encourage customers to make more regular purchases to maximize their credit benefits. Offer personalized product suggestions based on their purchase history.",
        1: "Recommendation for Segment 1: Leverage targeted marketing campaigns to promote high-end products or credit limit upgrades, considering their high credit utilization and purchase frequency.",
    }
    return recommendations.get(cluster, "No specific recommendations for this segment.")

if st.button("Predict"):
    cluster = perform_clustering(user_input)
    st.write("Predicted Cluster:", cluster)
    st.write("Segment Description:", get_segment_description(cluster))
    st.write("Recommendations:", get_recommendations(cluster))

