# app.py
# ------------------------------------------------------------
# Streamlit app with:
#   • 📊 Dashboard (metrics, ROC, CM, feature importance, CV table)
#   • 📄 "Generate & Download EDA Report" (HTML + PDF)  ← uses edareport.py
#   • 🔮 Predict (single patient form)
#   • 📁 Data Preview
#
# Run:  streamlit run app.py
# ------------------------------------------------------------

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

# EDA report helpers
from eda_report import build_eda_report_html, build_pdf_from_images

# ---------- App setup ----------
st.set_page_config(page_title="❤️ Heart Disease Prediction & Dashboard",
                   page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Prediction & Dashboard")

# ---------- Paths ----------
ART = Path("artifacts")
MODEL_PATH = ART / "heart_model.joblib"
METRICS_PATH = ART / "metrics.json"
TEST_DATA_PATH = ART / "test_data.csv"
ROC_IMG = ART / "roc_curve.png"
CM_IMG = ART / "confusion_matrix.png"
FI_IMG = ART / "feature_importance.png"
DATASET = Path("dataset.csv")

# ---------- Load assets ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None

@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_DATA_PATH) if TEST_DATA_PATH.exists() else None

@st.cache_data
def load_dataset():
    if DATASET.exists():
        df = pd.read_csv(DATASET)
        # normalize to snake_case (matches training)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df
    return None

pipe = load_model()
metrics = load_metrics()
test_df = load_test_data()
data_df = load_dataset()

if pipe is None:
    st.error("Model not found. Run `python train_heart.py` to generate ./artifacts first.")
    st.stop()

# Feature schema (must match training script)
NUMERIC = ["age", "resting_bp_s", "cholesterol", "max_heart_rate", "oldpeak"]
CATEG  = ["sex", "chest_pain_type", "fasting_blood_sugar", "resting_ecg", "exercise_angina", "st_slope"]

# ---------- Navigation (Dashboard first, Predict below) ----------
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Predict", "📁 Data Preview"])

# ==============================================================
# 📊 DASHBOARD
# ==============================================================
if page == "📊 Dashboard":
    st.subheader("Model Performance Dashboard")

    # Summary KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    if metrics and "test" in metrics:
        c1.metric("Best Model", metrics.get("best_model", "-"))
        c2.metric("ROC-AUC", f"{metrics['test']['roc_auc']:.3f}")
        c3.metric("Accuracy", f"{metrics['test']['accuracy']:.3f}")
        c4.metric("Precision", f"{metrics['test']['precision']:.3f}")
        c5.metric("Recall", f"{metrics['test']['recall']:.3f}")
    else:
        st.info("Metrics not found. Run training first.")

    st.markdown("---")
    cL, cR = st.columns(2)
    with cL:
        st.markdown("**ROC Curve**")
        if ROC_IMG.exists():
            st.image(str(ROC_IMG))
        else:
            st.info("roc_curve.png not found.")
    with cR:
        st.markdown("**Confusion Matrix**")
        if CM_IMG.exists():
            st.image(str(CM_IMG))
        else:
            st.info("confusion_matrix.png not found.")

    st.markdown("---")
    cL2, cR2 = st.columns(2)
    with cL2:
        st.markdown("**Feature Importance**")
        if FI_IMG.exists():
            st.image(str(FI_IMG))
        else:
            st.caption("Feature importance available for tree-based models (e.g., RandomForest).")
    with cR2:
        st.markdown("**Cross-Validation (ROC-AUC)**")
        cv_path = ART / "cv_results.csv"
        if cv_path.exists():
            cv_df = pd.read_csv(cv_path).sort_values("roc_auc_mean", ascending=False)
            st.dataframe(cv_df, use_container_width=True)
        else:
            st.info("cv_results.csv not found.")

    st.markdown("---")
    st.markdown("### Threshold Tuning (on the saved test split)")
    if test_df is None or test_df.empty:
        st.info("Test split not found. Re-run training to generate artifacts/test_data.csv.")
    else:
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve

        X_t = test_df.drop(columns=["target"])
        y_t = test_df["target"].astype(int).values
        y_prob = pipe.predict_proba(X_t)[:, 1]

        thr = st.slider("Decision threshold", min_value=0.0, max_value=1.0,
                        value=0.50, step=0.01)
        y_hat = (y_prob >= thr).astype(int)

        p = precision_score(y_t, y_hat, zero_division=0)
        r = recall_score(y_t, y_hat, zero_division=0)
        f1 = f1_score(y_t, y_hat, zero_division=0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{p:.3f}")
        m2.metric("Recall", f"{r:.3f}")
        m3.metric("F1", f"{f1:.3f}")

        fpr, tpr, thresholds = roc_curve(y_t, y_prob)
        fig = plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--")
        # mark chosen threshold
        import numpy as _np
        idx = int(_np.argmin(_np.abs(thresholds - thr)))
        plt.scatter([fpr[idx]], [tpr[idx]])
        plt.title("ROC Curve (test) with threshold marker")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.legend()
        st.pyplot(fig, clear_figure=True)

    # ---------- EDA REPORT (HTML + PDF) ----------
    st.markdown("---")
    st.markdown("### 📄 Generate & Download EDA Report")
    st.caption("Creates a single-file HTML report, and a multi-page PDF. (Requires seaborn, reportlab)")

    if data_df is None:
        st.info("dataset.csv not found — place it next to this script.")
    else:
        if st.button("Generate EDA Report"):
            with st.spinner("Building EDA report..."):
                html, ordered_images = build_eda_report_html(data_df)      # from edareport.py
                pdf_bytes = build_pdf_from_images(ordered_images,
                                                  title="Heart Disease — EDA Report")
                st.session_state["eda_html"] = html
                st.session_state["eda_pdf"] = pdf_bytes
            st.success("EDA report generated!")

        if "eda_html" in st.session_state:
            st.download_button(
                label="⬇️ Download eda_report.html",
                data=st.session_state["eda_html"].encode("utf-8"),
                file_name="eda_report.html",
                mime="text/html"
            )
        if "eda_pdf" in st.session_state:
            st.download_button(
                label="⬇️ Download eda_report.pdf",
                data=st.session_state["eda_pdf"],
                file_name="eda_report.pdf",
                mime="application/pdf"
            )

# ==============================================================
# 🔮 PREDICT
# ==============================================================
elif page == "🔮 Predict":
    st.subheader("Single Patient Prediction")
    st.caption("Provide values below and click **Predict**. Outputs are for educational purposes only.")

    # Encodings
    sex_map = {"Female (0)": 0, "Male (1)": 1}
    cp_map = {"Typical angina (1)": 1, "Atypical angina (2)": 2,
              "Non-anginal pain (3)": 3, "Asymptomatic (4)": 4}
    fbs_map = {"≤ 120 mg/dL (0)": 0, "> 120 mg/dL (1)": 1}
    rest_ecg_map = {"Normal (0)": 0, "ST-T abnormality (1)": 1, "LVH by Estes (2)": 2}
    ex_angina_map = {"No (0)": 0, "Yes (1)": 1}
    slope_map = {"Upward (1)": 1, "Flat (2)": 2, "Downward (3)": 3}

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", 1, 120, 50, 1, key="age")
            resting_bp_s = st.number_input("Resting BP (mm Hg)", 60, 260, 130, 1, key="resting_bp_s")
            cholesterol = st.number_input("Cholesterol (mg/dL)", 80, 800, 240, 1, key="cholesterol")
            max_hr = st.number_input("Max Heart Rate", 60, 230, 150, 1, key="max_heart_rate")
            oldpeak = st.number_input("Oldpeak (ST depression)", -5.0, 10.0, 1.0, 0.1, key="oldpeak")
        with c2:
            sex = sex_map[st.selectbox("Sex", list(sex_map.keys()), index=1)]
            chest_pain_type = cp_map[st.selectbox("Chest Pain Type", list(cp_map.keys()), index=3)]
            fasting_blood_sugar = fbs_map[st.selectbox("Fasting Blood Sugar", list(fbs_map.keys()), index=0)]
            resting_ecg = rest_ecg_map[st.selectbox("Resting ECG", list(rest_ecg_map.keys()), index=0)]
            exercise_angina = ex_angina_map[st.selectbox("Exercise-induced Angina", list(ex_angina_map.keys()), index=0)]
            st_slope = slope_map[st.selectbox("ST Slope", list(slope_map.keys()), index=1)]
        submitted = st.form_submit_button("Predict ❤️")

    if submitted:
        row = {
            "age": age, "resting_bp_s": resting_bp_s, "cholesterol": cholesterol,
            "max_heart_rate": max_hr, "oldpeak": oldpeak, "sex": sex,
            "chest_pain_type": chest_pain_type, "fasting_blood_sugar": fasting_blood_sugar,
            "resting_ecg": resting_ecg, "exercise_angina": exercise_angina, "st_slope": st_slope,
        }
        X_input = pd.DataFrame([row])
        proba = float(pipe.predict_proba(X_input)[0, 1])
        pred = int(proba >= 0.5)

        st.markdown(f"### Probability of Heart Disease: **{proba:.2%}**")
        if pred == 1:
            st.error("Model predicts: **Heart Disease (1)**")
        else:
            st.success("Model predicts: **No Heart Disease (0)**")
        st.caption("⚠️ Not a medical diagnosis. Educational use only.")

# ==============================================================
# 📁 DATA PREVIEW
# ==============================================================
elif page == "📁 Data Preview":
    st.subheader("Dataset Preview")
    if data_df is None:
        st.error("dataset.csv not found.")
    else:
        st.write(f"Rows: **{len(data_df):,}** | Columns: **{len(data_df.columns)}**")
        st.dataframe(data_df.head(20), use_container_width=True)
        with st.expander("Column schema"):
            st.json({c: str(data_df[c].dtype) for c in data_df.columns})
    st.markdown("---")
    st.caption("Note: Column names are normalized to snake_case during training.")
