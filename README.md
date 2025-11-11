❤️ Heart Disease Prediction & Dashboard

🩺 Machine Learning Web App built using Python and Streamlit that predicts the likelihood of heart disease based on medical parameters.
Includes a fully interactive dashboard, model performance metrics, ROC curve, confusion matrix, feature importance, and auto-generated EDA reports (HTML + PDF).

🚀 Live App

🔗 Launch the app on Streamlit Cloud
https://heart-disease-aiml.streamlit.app/


🧠 Project Description

This project predicts the risk of heart disease using machine learning techniques on a medical dataset.
The trained Random Forest Classifier achieves ~93% accuracy and 0.98 ROC-AUC.

The Streamlit-based web dashboard provides:

Model Performance Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

Visual Insights – ROC Curve, Confusion Matrix, Feature Importance

Interactive Predictions – enter patient details to get real-time results

EDA Report Generator – one-click export of detailed HTML and PDF reports

🧩 Features

✅ Heart disease risk prediction using Random Forest
✅ Model performance dashboard (ROC, CM, Feature Importance)
✅ Downloadable EDA Reports (HTML + PDF)
✅ Data visualization using Matplotlib & Seaborn
✅ Interactive Streamlit UI with form-based prediction
✅ Adjustable decision threshold for sensitivity tuning

📊 Model Performance
Metric	Score
Accuracy	0.93
Precision	0.94
Recall	0.93
F1-Score	0.93
ROC-AUC	0.98

Best Model: RandomForestClassifier
Features Used:

Numeric: age, resting_bp_s, cholesterol, max_heart_rate, oldpeak

Categorical: sex, chest_pain_type, fasting_blood_sugar, resting_ecg, exercise_angina, st_slope

🏗️ Tech Stack
Layer	Tools
Frontend	Streamlit
Machine Learning	Scikit-learn
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
PDF / Reports	ReportLab
Deployment	Streamlit Cloud

📂 Project Structure
HeartDiseaseML/
│
├── app_heart.py         # Streamlit app (Dashboard + Predict + Data Preview)
├── eda_report.py        # Generates EDA HTML & PDF reports
├── train_heart.py       # Model training & artifact generation
├── dataset.csv          # Input dataset
├── requirements.txt     # Python dependencies
├── artifacts/           # Saved model, metrics, plots
│   ├── heart_model.joblib
│   ├── metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
└── README.md

📈 Example Output

Prediction Example:
Probability of Heart Disease: 72.0%
Model predicts: Heart Disease (1)

EDA Report:

Correlation heatmap

Feature distributions

Categorical breakdown by target

Box-plots, histograms, and summary tables

Reports are downloadable in:

📄 HTML – interactive version

📘 PDF – formatted for reports & submissions

🌐 Deployment

Deployed on Streamlit Cloud.
Visit the live app at:
https://heart-disease-aiml.streamlit.app/
👨‍💻 Author
ACHAL URS S
