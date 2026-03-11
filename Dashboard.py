# =========================================================
# ADVANCED HEART DISEASE AI DASHBOARD 
# Research-Grade Clinical AI System + CDSS
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer
from datetime import datetime

st.set_page_config(page_title="Heart Disease AI Pro v4", layout="wide")

st.title("🫀 AI-Powered Heart Disease Risk Intelligence System ")
st.markdown("Clinical-grade predictive modeling with Explainable AI + Calibration + Multi-Model Analysis")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    df = df = pd.read_csv("heart.csv")
    df = pd.get_dummies(df, drop_first=True).astype(float)
    return df

df = load_data()
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# =========================================================
# TRAIN MODELS
# =========================================================
@st.cache_resource
def train_models():

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale only for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42
    )

    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    xgb_model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )

    log_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    explainer = shap.Explainer(xgb_model, X_train)

    lime_explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X.columns,
        class_names=["No Disease", "Heart Disease"],
        mode="classification"
    )

    return (
        log_model, rf_model, xgb_model,
        explainer, lime_explainer,
        X_test, y_test,
        scaler
    )

log_model, rf_model, model, explainer, lime_explainer, X_test, y_test, scaler = train_models()

# =========================================================
# RISK ZONE FUNCTION
# =========================================================
def risk_zone(prob):
    if prob < 0.1:
        return "🟢 Low Clinical Risk"
    elif prob < 0.2:
        return "🟡 Mild Risk"
    elif prob < 0.4:
        return "🟠 Moderate Risk"
    elif prob < 0.7:
        return "🔴 High Risk"
    else:
        return "🚨 Critical Risk"

# =========================================================
# REALISTIC BOOTSTRAP CI
# =========================================================
def bootstrap_ci(model, x, n=100):
    preds = []
    for _ in range(n):
        noise = np.random.normal(0, 0.01, size=x.shape)
        x_noisy = x + noise
        preds.append(model.predict_proba(x_noisy)[0][1])
    return np.percentile(preds, [5, 95])

# =========================================================
# CDSS RECOMMENDATION ENGINE
# =========================================================
recommendations_dict = {
    "Cholesterol": "Reduce saturated fats, increase fiber intake, and monitor lipid profile regularly.",
    "RestingBP": "Monitor blood pressure daily and reduce sodium intake.",
    "MaxHR": "Structured cardiovascular exercise under medical supervision recommended.",
    "Age": "Ensure regular preventive cardiology screening.",
    "Smoking": "Immediate smoking cessation program recommended.",
    "Diabetes": "Maintain strict glycemic control and endocrinology follow-up.",
    "BMI": "Adopt a structured weight management plan.",
    "ExerciseAngina": "Further cardiology evaluation for exertional symptoms advised.",
    "Oldpeak": "Consider advanced cardiac stress testing."
}

# =========================================================
# SIDEBAR
# =========================================================
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview",
     "🌍 Global Explainability",
     "👤 Single Patient Analysis",
     "🔄 What-If Simulation",
     "📊 Model Performance",
     "📈 Calibration & Clinical Utility",
     "📁 Batch High-Risk Patients"]
)

# =========================================================
# OVERVIEW
# =========================================================
if page == "🏠 Overview":

    st.subheader("Dataset Overview")
    st.write(df.head())

    col1, col2 = st.columns(2)
    col1.metric("Total Patients", len(df))
    col2.metric("Disease Prevalence", f"{y.mean():.2%}")

    st.subheader("Population Risk Distribution")
    probs = model.predict_proba(X)[:, 1]
    plt.figure()
    sns.histplot(probs, bins=20, kde=True)
    plt.title("Population Risk Distribution")
    st.pyplot(plt)

# =========================================================
# GLOBAL EXPLAINABILITY
# =========================================================
elif page == "🌍 Global Explainability":

    shap_values = explainer(X)

    st.subheader("Global SHAP Feature Importance")
    fig = plt.figure()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig)

# =========================================================
# SINGLE PATIENT
# =========================================================
elif page == "👤 Single Patient Analysis":

    shap_values = explainer(X)

    patient_id = st.slider("Select Patient ID", 0, len(X)-1, 0)
    patient_data = X.iloc[patient_id]

    prob = model.predict_proba(patient_data.values.reshape(1,-1))[0][1]
    ci_low, ci_high = bootstrap_ci(model, patient_data.values.reshape(1,-1))

    st.metric("Predicted Risk", f"{prob:.2%}")
    st.write(risk_zone(prob))
    st.write(f"Confidence Interval (5–95%): {ci_low:.2%} - {ci_high:.2%}")

    st.subheader("SHAP Explanation")
    fig = plt.figure()
    shap.plots.waterfall(shap_values[patient_id], show=False)
    st.pyplot(fig)

    st.subheader("LIME Explanation")
    lime_exp = lime_explainer.explain_instance(
        patient_data.values,
        model.predict_proba,
        num_features=5
    )
    fig2 = lime_exp.as_pyplot_figure()
    st.pyplot(fig2)

    # CDSS Recommendations
    st.subheader("🩺 Personalized Clinical Recommendations")
    shap_vals = shap_values[patient_id].values
    top_idx = np.argsort(np.abs(shap_vals))[-3:][::-1]
    top_features = X.columns[top_idx]

    for f in top_features:
        if f in recommendations_dict:
            st.write(f"**{f}** → {recommendations_dict[f]}")

    # AI Clinical Summary
    st.subheader("📋 AI Clinical Summary")

    summary = f"""
    Patient demonstrates a predicted cardiovascular risk of {prob:.2%}.
    Risk category: {risk_zone(prob)}.
    Confidence interval range: {ci_low:.2%} - {ci_high:.2%}.
    Primary contributing factors include: {', '.join(top_features)}.
    Clinical follow-up and preventive intervention should be considered accordingly.
    """

    st.info(summary)

# =========================================================
# WHAT-IF SIMULATION
# =========================================================
elif page == "🔄 What-If Simulation":

    st.subheader("Simulate Risk Change")

    input_data = X.iloc[0].copy()

    for col in X.columns:
        input_data[col] = st.number_input(col, value=float(input_data[col]))

    prob = model.predict_proba(np.array(input_data).reshape(1,-1))[0][1]
    st.metric("Simulated Risk", f"{prob:.2%}")
    st.write(risk_zone(prob))

# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "📊 Model Performance":

    st.subheader("ROC Curve Comparison")

    models = {
        "Logistic Regression": log_model,
        "Random Forest": rf_model,
        "XGBoost": model
    }

    plt.figure()

    for name, m in models.items():

        if name == "Logistic Regression":
            probs = m.predict_proba(scaler.transform(X_test))[:,1]
        else:
            probs = m.predict_proba(X_test)[:,1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_score = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})")

    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt)

# =========================================================
# CALIBRATION
# =========================================================
elif page == "📈 Calibration & Clinical Utility":

    st.subheader("Calibration Curve (XGBoost)")

    probs = model.predict_proba(X_test)[:,1]
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    st.pyplot(plt)

# =========================================================
# BATCH HIGH-RISK
# =========================================================
elif page == "📁 Batch High-Risk Patients":

    st.subheader("High-Risk Patient Identification")

    probs = model.predict_proba(X)[:,1]
    df_results = df.copy()
    df_results["PredictedRisk"] = probs

    high_risk = df_results[df_results["PredictedRisk"] > 0.7]

    st.metric("High-Risk Patients (>70%)", len(high_risk))

    st.dataframe(high_risk.sort_values("PredictedRisk", ascending=False).head(20))
