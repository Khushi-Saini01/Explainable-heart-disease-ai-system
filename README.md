🫀 Heart Disease AI Pro v4

Clinical-Grade AI Dashboard for Heart Disease Risk Prediction & Explainable Decision Support

This project is an advanced AI-powered heart disease risk intelligence system, combining multiple machine learning models, explainable AI (SHAP & LIME), and clinical decision support. It is designed for researchers, clinicians, and data enthusiasts to understand, predict, and visualize cardiovascular risk in patients.

🔍 Features
1. Multi-Model Risk Prediction

Implements Logistic Regression, Random Forest, SVM, Neural Network, and XGBoost.

XGBoost selected as the final high-performance model.

Provides probabilities, risk categories, and confidence intervals for each patient.

2. Explainable AI (XAI)

SHAP: Global and patient-specific feature importance.

LIME: Instance-level explanations for personalized predictions.

Highlights top contributing risk factors for each patient.

3. Interactive Clinical Dashboard

Single patient risk analysis with SHAP & LIME explanations.

What-If simulation: Test interventions or lifestyle changes and see impact on predicted risk.

Batch analysis: Identify high-risk patients for preventive interventions.

4. Publication-Quality Visualizations

ROC-AUC comparison for multiple models.

Feature importance plots (XGBoost, SHAP, LIME).

Risk distribution histograms, calibration curves, decision curve analysis (DCA), and SHAP interaction plots.

5. Clinical Decision Support

Personalized recommendations for key risk factors:

Cholesterol, Blood Pressure, Max Heart Rate, Smoking, Diabetes, BMI, Exercise Angina, etc.

Risk categories: Low → Critical Risk with actionable guidance.

Designed to integrate seamlessly into research workflows or clinical pilot studies.

🛠️ Technologies Used

Python 3.10+

Machine Learning: scikit-learn, XGBoost, MLPClassifier

Explainable AI: SHAP, LIME

Visualization: Matplotlib, Seaborn

Web Dashboard: Streamlit

Data Handling: pandas, numpy

🚀 Installation & Setup

Clone the repository:

git clone https://github.com/yourusername/HeartDiseaseAIPro.git
cd HeartDiseaseAIPro

Install dependencies:

pip install -r requirements.txt

Run the Streamlit dashboard:

streamlit run streamlit_app.py
🧠 How It Works

Load dataset (heart.csv) and preprocess features.

Split data into training and testing sets.

Train multiple models (Logistic Regression, Random Forest, SVM, Neural Network, XGBoost).

Evaluate models using accuracy, precision, recall, F1, ROC-AUC.

Select XGBoost as the final model for explainable AI analysis.

Compute SHAP values for global and local interpretability.

Compute LIME explanations for patient-level insights.

Generate interactive dashboards with patient-specific predictions and recommendations.

Enable what-if simulations to test potential interventions.

Provide batch analysis for high-risk patient identification.

📊 Dashboard Pages
Page	Features
🏠 Overview	Dataset preview, population risk distribution
🌍 Global Explainability	SHAP global feature importance
👤 Single Patient Analysis	Patient-level risk prediction, SHAP & LIME explanations, clinical recommendations
🔄 What-If Simulation	Test changes in patient features to simulate risk reduction
📊 Model Performance	ROC curves & AUC comparison for all models
📈 Calibration & Clinical Utility	Calibration curves and decision curve analysis
📁 Batch High-Risk Patients	Identify and summarize high-risk patients
📈 Visual Outputs

ROC-AUC model comparison

SHAP global & individual feature importance

LIME patient-level explanations

Risk probability distribution

Precision–Recall & calibration curves

Decision curve analysis

SHAP interaction plots for top features

🎯 Target Users

Cardiologists & healthcare professionals

Clinical data scientists

Researchers in cardiovascular risk modeling

ML enthusiasts exploring explainable AI in healthcare

⚡ Future Enhancements

Integration with real-time EHR systems.

Mobile-friendly interactive dashboard.

Automated risk stratification alerts for high-risk patients.

Incorporate longitudinal patient data for predictive modeling over time.

📁 Dataset

Source: Heart Disease Dataset CSV

Features include demographics, vitals, lab results, and lifestyle indicators.

Target: HeartDisease (binary classification)

📌 References

SHAP: Lundberg & Lee, 2017

LIME: Ribeiro et al., 2016

XGBoost: Chen & Guestrin, 2016

Streamlit documentation: https://docs.streamlit.io



