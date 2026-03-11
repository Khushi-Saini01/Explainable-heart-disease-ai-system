# =========================================================
# 0. REPRODUCIBILITY
# =========================================================
import numpy as np
np.random.seed(42)

# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 2. LOAD DATASET
# =========================================================
df = pd.read_csv("heart.csv")
df = pd.get_dummies(df, drop_first=True).astype(float)
print("\nDataset Preview:")
print(df.head())

# =========================================================
# 3. CLASS DISTRIBUTION
# =========================================================
print("\nClass Distribution:")
print(df["HeartDisease"].value_counts())
print(df["HeartDisease"].value_counts(normalize=True))

# =========================================================
# 4. FEATURE / TARGET SPLIT
# =========================================================
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# =========================================================
# 5. TRAIN–TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 6. FEATURE SCALING (NON-TREE MODELS)
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 7. MODEL TRAINING FUNCTION
# =========================================================
def train_evaluate(model, X_tr, y_tr, X_te, y_te, model_name="Model"):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_te)[:,1]
    else:
        prob = model.decision_function(X_te)
        prob = (prob - prob.min()) / (prob.max() - prob.min())  # scale to 0-1
    print(f"\n=== {model_name} Performance ===")
    print(classification_report(y_te, pred))
    print("ROC-AUC:", roc_auc_score(y_te, prob))
    return pred, prob

# =========================================================
# 8. BASELINE MODELS
# =========================================================
log_pred, log_prob = train_evaluate(LogisticRegression(max_iter=1000), X_train_scaled, y_train, X_test_scaled, y_test, "Logistic Regression")
rf_pred, rf_prob = train_evaluate(RandomForestClassifier(n_estimators=300, random_state=42), X_train, y_train, X_test, y_test, "Random Forest")
svm_pred, svm_prob = train_evaluate(SVC(probability=True, random_state=42), X_train_scaled, y_train, X_test_scaled, y_test, "SVM")
mlp_pred, mlp_prob = train_evaluate(MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42), X_train_scaled, y_train, X_test_scaled, y_test, "Neural Network")

# =========================================================
# 9. MODEL COMPARISON TABLE
# =========================================================
models = {"Logistic Regression": (log_pred, log_prob),
          "Random Forest": (rf_pred, rf_prob),
          "SVM": (svm_pred, svm_prob),
          "Neural Network": (mlp_pred, mlp_prob)}

results = []
for name, (pred, prob) in models.items():
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })

results_df = pd.DataFrame(results)
print("\n=== Model Comparison Table ===")
print(results_df)

# =========================================================
# 10. XGBOOST (FINAL MODEL)
# =========================================================
xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42
)
xgb_pred, xgb_prob = train_evaluate(xgb, X_train, y_train, X_test, y_test, "XGBoost")

# =========================================================
# 11. CONFUSION MATRIX
# =========================================================
cm = confusion_matrix(y_test, xgb_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("XGBoost Confusion Matrix")
plt.show()

# =========================================================
# 12. ROC CURVE
# =========================================================
fpr, tpr, _ = roc_curve(y_test, xgb_prob)
plt.plot(fpr, tpr, label="XGBoost")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve"); plt.legend(); plt.show()

# =========================================================
# 13. FEATURE IMPORTANCE
# =========================================================
importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n=== Top 10 Features ===")
print(importances.head(10))

# =========================================================
# 14. SHAP EXPLAINER (FULL DATA)
# =========================================================
explainer = shap.Explainer(xgb, X)
shap_values = explainer(X)
print("\nSHAP values computed for full dataset ✅")

# =========================================================
# 15. LIME EXPLAINER
# =========================================================
lime_explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=X.columns,
    class_names=["No Disease","Heart Disease"],
    mode="classification"
)

# =========================================================
# 16. PATIENT-SPECIFIC INTERPRETABILITY
# =========================================================
recommendations = {
    "Cholesterol": "Reduce saturated fat, increase fiber, exercise",
    "BloodPressure": "Monitor BP, reduce salt intake, exercise",
    "Age": "Regular health checkups recommended",
    "Smoking": "Enroll in cessation program",
    "Diabetes": "Control blood sugar, follow diet plan",
    "RestingBP": "Monitor BP and consult doctor if high",
    "MaxHR": "Maintain regular cardiovascular exercise"
}

def top_risk_factors(shap_vals, idx, X_columns, top_n=3):
    vals = shap_vals[idx].values
    top_idx = np.argsort(np.abs(vals))[-top_n:][::-1]
    return X_columns[top_idx]

def patient_recommendations(patient_id, prob, shap_vals, X):
    print(f"\nPatient {patient_id} Risk Probability: {prob:.2f}")
    risk_cat = "Low" if prob<0.2 else "Moderate" if prob<0.5 else "High" if prob<0.8 else "Very High"
    print("Risk Category:", risk_cat)
    if prob >= 0.8: print("⚠️ High-risk patient! Immediate consultation recommended.")
    elif prob >= 0.5: print("🔔 Moderate risk: Schedule follow-up soon.")
    else: print("✅ Low risk: Maintain healthy lifestyle.")

    # SHAP Top Features
    top_features = top_risk_factors(shap_vals, patient_id, X.columns)
    print("Top Contributing SHAP Features:", list(top_features))
    for f in top_features:
        if f in recommendations:
            print(f"Recommendation for {f}: {recommendations[f]}")

    # LIME explanation
    lime_exp = lime_explainer.explain_instance(X.iloc[patient_id].values, xgb.predict_proba, num_features=3)
    print("\nTop Contributing LIME Features:")
    for feature, weight in lime_exp.as_list():
        print(f"{feature}: {weight:.4f}")

# =========================================================
# 17. WHAT-IF SCENARIO
# =========================================================
def simulate_scenario(patient_id, changes: dict, model, X):
    patient_data = X.iloc[patient_id].copy()
    original_prob = model.predict_proba(patient_data.values.reshape(1, -1))[0][1]
    print(f"\nOriginal probability: {original_prob:.2f}")
    for f, val in changes.items():
        if f in patient_data.index:
            patient_data[f] = val
            print(f"Changed {f} -> {val}")
    new_prob = model.predict_proba(patient_data.values.reshape(1, -1))[0][1]
    print(f"New predicted risk: {new_prob:.2f}")

# =========================================================
# 18. BATCH SUMMARY FOR HIGH-RISK PATIENTS
# =========================================================
def batch_summary(probs, shap_vals, X, threshold=0.7):
    high_risk_idx = [i for i, p in enumerate(probs) if p >= threshold]
    if not high_risk_idx:
        print("\nNo patients above threshold.")
        return pd.DataFrame()
    
    summary = []
    for i in high_risk_idx:
        prob = probs[i]
        top_feats = X.columns[np.argsort(np.abs(shap_vals[i].values))[-3:][::-1]]
        recs = [recommendations.get(f, "No recommendation") for f in top_feats]
        summary.append({
            "Patient_ID": i,
            "Risk_Prob": prob,
            "Top_Features": list(top_feats),
            "Recommendations": recs
        })
    summary_df = pd.DataFrame(summary)
    print(f"\n=== High-Risk Patients (>{threshold}) Summary ===")
    print(summary_df)
    return summary_df

# =========================================================
# 19. INTERACTIVE CLI DASHBOARD
# =========================================================
def cli_dashboard(model, shap_vals, X):
    while True:
        print("\n--- Heart Disease Risk Dashboard ---")
        print("1️⃣  Analyze single patient")
        print("2️⃣  Simulate what-if scenario for a patient")
        print("3️⃣  View batch summary of high-risk patients")
        print("4️⃣  Exit")
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == "1":
            try:
                pid = int(input(f"Enter Patient ID (0-{len(X)-1}): "))
                if pid < 0 or pid >= len(X):
                    print("❌ Invalid Patient ID")
                    continue
            except:
                print("❌ Invalid input")
                continue
            prob = model.predict_proba(X.iloc[pid].values.reshape(1,-1))[0][1]
            patient_recommendations(pid, prob, shap_vals, X)
        
        elif choice == "2":
            try:
                pid = int(input(f"Enter Patient ID (0-{len(X)-1}): "))
                if pid < 0 or pid >= len(X):
                    print("❌ Invalid Patient ID")
                    continue
            except:
                print("❌ Invalid input")
                continue
            changes = {}
            print("Enter feature changes (press Enter to stop):")
            while True:
                f = input("Feature name (or press Enter to finish): ").strip()
                if f == "":
                    break
                if f not in X.columns:
                    print("❌ Feature not found")
                    continue
                try:
                    val = float(input(f"New value for {f}: "))
                    changes[f] = val
                except:
                    print("❌ Invalid value, must be a number")
            simulate_scenario(pid, changes, model, X)
        
        elif choice == "3":
            try:
                threshold = float(input("Enter risk threshold (0-1, default 0.7): ") or 0.7)
            except:
                threshold = 0.7
            batch_summary(model.predict_proba(X)[:,1], shap_vals, X, threshold)
        
        elif choice == "4":
            print("👋 Exiting dashboard.")
            break
        
        else:
            print("❌ Invalid choice. Enter 1-4.")

# =========================================================
# RUN CLI DASHBOARD
# =========================================================
cli_dashboard(xgb, shap_values, X)
# =========================================================
# 20. PUBLICATION-QUALITY VISUALIZATIONS
# =========================================================

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 300
})

# ---------------------------------------------------------
# 20a. Model Comparison (ROC-AUC)
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.barplot(data=results_df, x="Model", y="ROC-AUC", palette="viridis")
plt.title("ROC-AUC Comparison of Machine Learning Models")
plt.ylabel("ROC-AUC Score")
plt.xlabel("")
plt.ylim(0,1)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_comparison_roc_auc.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 20b. XGBoost Feature Importance (Top 10)
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
top10 = importances.head(10)
sns.barplot(x=top10.values, y=top10.index, palette="magma")
plt.title("Top 10 Important Features (XGBoost)")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("xgboost_feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 20c. SHAP Global Importance (Bar)
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("Global Feature Importance using SHAP")
plt.tight_layout()
plt.savefig("shap_global_bar.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 20d. SHAP Beeswarm Plot
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Feature Impact Distribution")
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 20e. SHAP Waterfall Plot (Single Patient)
# ---------------------------------------------------------
patient_idx = 0
plt.figure(figsize=(6,4))
shap.plots.waterfall(shap_values[patient_idx], max_display=10, show=False)
plt.tight_layout()
plt.savefig("shap_waterfall_patient.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 20f. LIME Explanation Plot (Single Patient)
# ---------------------------------------------------------
lime_exp = lime_explainer.explain_instance(
    X.iloc[patient_idx].values,
    xgb.predict_proba,
    num_features=5
)

fig = lime_exp.as_pyplot_figure()
plt.gcf().set_size_inches(6,4)
plt.title(f"LIME Feature Contributions (Patient {patient_idx})")
plt.tight_layout()
plt.savefig("lime_patient_explanation.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 20g. Predicted Risk Probability Distribution
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.histplot(xgb.predict_proba(X)[:,1], bins=25, kde=True)
plt.title("Distribution of Predicted Heart Disease Risk")
plt.xlabel("Predicted Risk Probability")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("risk_probability_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# 21. ADVANCED RESEARCH VISUALIZATIONS
# =========================================================
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 300
})

# ---------------------------------------------------------
# 21a. Precision–Recall Curve
# ---------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test, xgb_prob)
pr_auc = auc(recall, precision)

plt.figure(figsize=(5,4))
plt.plot(recall, precision, label=f"XGBoost (AUC = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("precision_recall_curve.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 21b. Calibration Curve
# ---------------------------------------------------------
prob_true, prob_pred = calibration_curve(y_test, xgb_prob, n_bins=10)

plt.figure(figsize=(5,4))
plt.plot(prob_pred, prob_true, marker='o', label="XGBoost")
plt.plot([0,1], [0,1], linestyle='--', label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Curve")
plt.legend()
plt.tight_layout()
plt.savefig("calibration_curve.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 21c. Decision Curve Analysis (DCA)
# ---------------------------------------------------------
thresholds = np.linspace(0.01, 0.99, 100)
net_benefit = []

for t in thresholds:
    preds = (xgb_prob >= t).astype(int)
    tp = np.sum((preds == 1) & (y_test == 1))
    fp = np.sum((preds == 1) & (y_test == 0))
    nb = (tp / len(y_test)) - (fp / len(y_test)) * (t / (1 - t))
    net_benefit.append(nb)

plt.figure(figsize=(5,4))
plt.plot(thresholds, net_benefit, label="XGBoost")
plt.plot(thresholds, np.zeros_like(thresholds), linestyle='--', label="Treat None")
plt.xlabel("Threshold Probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis")
plt.legend()
plt.tight_layout()
plt.savefig("decision_curve_analysis.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 21d. SHAP Interaction Plot (Top Feature)
# ---------------------------------------------------------
top_feature = importances.index[0]

plt.figure(figsize=(5,4))
shap.plots.scatter(shap_values[:, top_feature], color=shap_values)
plt.title(f"SHAP Interaction Plot: {top_feature}")
plt.tight_layout()
plt.savefig("shap_interaction_plot.png", dpi=300, bbox_inches="tight")
plt.show()


# ---------------------------------------------------------
# 21e. Statistical Comparison Table
# ---------------------------------------------------------
advanced_results = results_df.copy()
advanced_results.loc[len(advanced_results)] = {
    "Model": "XGBoost",
    "Accuracy": accuracy_score(y_test, xgb_pred),
    "Precision": precision_score(y_test, xgb_pred),
    "Recall": recall_score(y_test, xgb_pred),
    "F1": f1_score(y_test, xgb_pred),
    "ROC-AUC": roc_auc_score(y_test, xgb_prob)
}

print("\n=== Final Model Performance Comparison ===")

print(advanced_results)
