import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import shap

# ====================================
# LOAD MODEL FILES
# ====================================

model = pickle.load(open(
    'D:/D drive/Project/ML/Customer-Churn-Project/model/model.pkl',
    'rb'
))

scaler = pickle.load(open(
    'D:/D drive/Project/ML/Customer-Churn-Project/model/scaler.pkl',
    'rb'
))

columns = pickle.load(open(
    'D:/D drive/Project/ML/Customer-Churn-Project/model/columns.pkl',
    'rb'
))

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="AI Customer Churn Prediction",
    layout="centered"
)

st.title("🤖 AI Customer Churn Prediction System")
st.subheader("Enter Customer Details")

# ====================================
# USER INPUTS
# ====================================

tenure = st.slider(
    "Tenure (months)",
    min_value=0,
    max_value=72,
    value=24
)

monthly = st.number_input(
    "Monthly Charges",
    value=50.0
)

total = st.number_input(
    "Total Charges",
    value=1000.0
)

contract = st.selectbox(
    "Contract Type",
    [0, 1, 2]
)

# ====================================
# CREATE INPUT DATA
# ====================================

input_dict = {}

for col in columns:
    input_dict[col] = 0

if 'tenure' in columns:
    input_dict['tenure'] = tenure

if 'MonthlyCharges' in columns:
    input_dict['MonthlyCharges'] = monthly

if 'TotalCharges' in columns:
    input_dict['TotalCharges'] = total

if 'Contract' in columns:
    input_dict['Contract'] = contract

input_df = pd.DataFrame([input_dict])

input_df = input_df[columns]

input_scaled = scaler.transform(input_df)

# ====================================
# PREDICTION
# ====================================

if st.button("Predict"):

    prediction = model.predict(input_scaled)[0]

    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Result")

    # ====================================
    # CHURN RISK GAUGE
    # ====================================

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Churn Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ]
        }
    ))

    st.plotly_chart(fig)

    # ====================================
    # PROBABILITY CHART
    # ====================================

    labels = ['Stay', 'Churn']
    values = [1 - prob, prob]

    fig2, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_title("Prediction Probability")

    st.pyplot(fig2)

    # ====================================
    # RISK LEVEL
    # ====================================

    if prob > 0.75:

        st.error("🚨 Very High Risk Customer")

    elif prob > 0.50:

        st.warning("⚠ Moderate Risk Customer")

    else:

        st.success("✅ Low Risk Customer")

    # ====================================
    # BUSINESS RECOMMENDATIONS
    # ====================================

    if prob > 0.50:

        st.subheader("💡 Recommended Actions")

        recommendations = []

        if monthly > 80:
            recommendations.append("💰 Offer discount or promotional pricing")

        if contract == 0:
            recommendations.append("📅 Encourage annual contract plan")

        if tenure < 6:
            recommendations.append("📞 Dedicated onboarding and support")

        if len(recommendations) == 0:
            recommendations.append("📧 Run customer engagement campaign")

        for rec in recommendations:
            st.write(rec)

    # ====================================
    # CUSTOMER SUMMARY
    # ====================================

    st.subheader("📋 Customer Summary")

    summary_df = pd.DataFrame({
        "Feature": [
            "Tenure",
            "Monthly Charges",
            "Total Charges",
            "Contract"
        ],
        "Value": [
            tenure,
            monthly,
            total,
            contract
        ]
    })

    st.table(summary_df)

    # ====================================
    # SHAP EXPLAINABILITY
    # ====================================

    st.subheader("🔍 AI Explanation (SHAP)")

    try:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(input_scaled)

        shap_df = pd.DataFrame({
            "Feature": columns,
            "Impact": shap_values[0]
        })

        shap_df["Abs Impact"] = np.abs(shap_df["Impact"])

        shap_df = shap_df.sort_values(
            by="Abs Impact",
            ascending=False
        )

        st.write("### Top Factors Influencing Churn")

        st.dataframe(
            shap_df[
                ["Feature", "Impact"]
            ].head(10)
        )

        # Bar Chart

        top_features = shap_df.head(10)

        fig3, ax = plt.subplots(figsize=(8,5))

        ax.barh(
            top_features["Feature"],
            top_features["Abs Impact"]
        )

        ax.invert_yaxis()

        ax.set_title(
            "Top Churn Drivers"
        )

        st.pyplot(fig3)

    except Exception as e:

        st.warning(
            f"SHAP explanation unavailable: {e}"
        )

    st.subheader("🧠 Why is this customer at risk?")

    top3 = shap_df.head(3)

    for _, row in top3.iterrows():

        if row["Impact"] > 0:
            st.write(
                f"🔴 {row['Feature']} is increasing churn risk"
            )
        else:
            st.write(
                f"🟢 {row['Feature']} is reducing churn risk"
            )
    # ====================================
    # INTERPRETATION
    # ====================================

    st.subheader("🧠 Business Interpretation")

    if tenure < 6:
        st.write(
            "🔴 Customer tenure is very low, increasing churn risk."
        )

    if contract == 0:
        st.write(
            "🔴 Month-to-month contract increases churn probability."
        )

    if monthly > 80:
        st.write(
            "🔴 High monthly charges may contribute to churn."
        )

    if prob < 0.5:
        st.write(
            "🟢 Customer appears relatively stable."
        )

    # ====================================
    # FEATURE IMPORTANCE
    # ====================================

    if hasattr(model, "feature_importances_"):

        st.subheader("📈 Model Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": columns,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        fig4, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            importance_df["Feature"].head(10),
            importance_df["Importance"].head(10)
        )

        ax.invert_yaxis()

        ax.set_title(
            "Most Important Features"
        )

        st.pyplot(fig4)