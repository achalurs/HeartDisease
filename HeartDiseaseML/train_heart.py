# train_heart.py
# ------------------------------------------------------------
# Train, compare, and save the best ML pipeline for Heart Disease prediction.
# Also saves evaluation artifacts for the Streamlit dashboard:
# - artifacts/heart_model.joblib
# - artifacts/roc_curve.png
# - artifacts/confusion_matrix.png
# - artifacts/feature_importance.png (if available)
# - artifacts/metrics.json
# - artifacts/cv_results.csv
# - artifacts/test_data.csv (X_test + y_test for dashboard analytics)
# ------------------------------------------------------------

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import joblib


# ---------- Helpers ----------
def to_snake(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_feature_names(preprocessor, numeric_features, categorical_features):
    num_names = list(numeric_features)
    cat = preprocessor.named_transformers_["cat"]
    cat_names = list(cat.get_feature_names_out(categorical_features))
    return num_names + cat_names


# ---------- 1) Load data ----------
csv_path = Path("dataset.csv")
if not csv_path.exists():
    raise FileNotFoundError("dataset.csv not found. Place it next to this script.")

df_raw = pd.read_csv(csv_path)
df = df_raw.copy()
df.columns = [to_snake(c) for c in df.columns]

expected_cols = {
    "age", "sex", "chest_pain_type", "resting_bp_s", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_angina", "oldpeak", "st_slope", "target"
}
missing = expected_cols - set(df.columns)
if missing:
    raise ValueError(
        f"Your CSV is missing columns: {sorted(missing)}\n"
        f"Found columns: {list(df.columns)}"
    )

# Quick NA handling (dataset should be clean, but we guard anyway)
if df.isna().sum().sum() > 0:
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# ---------- 2) Split ----------
X = df.drop("target", axis=1)
y = df["target"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ---------- 3) Preprocess ----------
categorical_features = [
    "sex", "chest_pain_type", "fasting_blood_sugar",
    "resting_ecg", "exercise_angina", "st_slope"
]
numeric_features = ["age", "resting_bp_s", "cholesterol", "max_heart_rate", "oldpeak"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="drop",
)

# ---------- 4) Candidate models ----------
candidates = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVC": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
best_name, best_score, best_pipeline = None, -np.inf, None

print("\n=== Cross-validated model comparison (ROC AUC) ===")
for name, model in candidates.items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    mean_auc, std_auc = scores.mean(), scores.std()
    cv_rows.append({"model": name, "roc_auc_mean": mean_auc, "roc_auc_std": std_auc})
    print(f"{name:>18}: ROC-AUC = {mean_auc:.4f} ± {std_auc:.4f}")
    if mean_auc > best_score:
        best_score, best_name, best_pipeline = mean_auc, name, pipe

print(f"\n✅ Best CV model: {best_name} (ROC-AUC {best_score:.4f})")

# ---------- 5) Fit best and evaluate ----------
best_pipeline.fit(X_train, y_train)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("\n=== Test Set Metrics ===")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}\n")

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))

print("\nConfusion Matrix:")
print(cm)

# ---------- 6) Save artifacts ----------
ART = Path("artifacts")
ensure_dir(ART)

# Model
joblib.dump(best_pipeline, ART / "heart_model.joblib")

# CV table
pd.DataFrame(cv_rows).sort_values("roc_auc_mean", ascending=False)\
  .to_csv(ART / "cv_results.csv", index=False)

# ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title(f"ROC Curve - {best_name}")
plt.tight_layout()
plt.savefig(ART / "roc_curve.png", dpi=150)
plt.close()

# Confusion matrix plot
plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()
classes = ["No Disease (0)", "Disease (1)"]
tick_marks = np.arange(2)
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()
plt.savefig(ART / "confusion_matrix.png", dpi=150)
plt.close()

# Feature importance (if model exposes it)
try:
    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        top_k = min(20, len(importances))

        plt.figure(figsize=(8, 6))
        plt.barh([feature_names[i] for i in order[:top_k]][::-1],
                 importances[order[:top_k]][::-1])
        plt.title(f"Top Feature Importances - {best_name}")
        plt.tight_layout()
        plt.savefig(ART / "feature_importance.png", dpi=150)
        plt.close()
except Exception as e:
    print(f"(Feature importance skipped: {e})")

# Metrics JSON
metrics = {
    "best_model": best_name,
    "cv_best_roc_auc": round(best_score, 6),
    "test": {
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "roc_auc": round(auc, 6),
        "confusion_matrix": cm.tolist(),
    },
    "features": {
        "numeric": numeric_features,
        "categorical": categorical_features
    },
}
with open(ART / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save test data for dashboard (raw, snake_case, includes target)
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(ART / "test_data.csv", index=False)

print("\n💾 Saved artifacts in ./artifacts")
print(" - heart_model.joblib")
print(" - roc_curve.png")
print(" - confusion_matrix.png")
print(" - feature_importance.png (if available)")
print(" - metrics.json")
print(" - cv_results.csv")
print(" - test_data.csv")
