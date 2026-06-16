"""
train.py — Train a Random Forest model for customer churn prediction.

Usage:
    python train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

from preprocess import load_data, preprocess

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/E_Commerce_Dataset.xlsx"
MODEL_PATH  = "models/random_forest_churn.pkl"
OUTPUT_DIR  = "outputs"
RANDOM_SEED = 42
TEST_SIZE   = 0.20

os.makedirs("models",  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load & preprocess ──────────────────────────────────────────────────────
print("=" * 60)
print("  Customer Churn Prediction — Random Forest")
print("=" * 60)

print("\n[1/6] Loading and preprocessing data ...")
df_raw = load_data(DATA_PATH)
print(f"      Raw shape  : {df_raw.shape}")
print(f"      Churn rate : {df_raw['Churn'].mean():.2%}")

X, y, feature_names = preprocess(df_raw)

# ── 2. Train / test split ─────────────────────────────────────────────────────
print("\n[2/6] Splitting data ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"      Train : {X_train.shape[0]} rows")
print(f"      Test  : {X_test.shape[0]} rows")

# ── 3. SMOTE — handle class imbalance ────────────────────────────────────────
print("\n[3/6] Applying SMOTE to balance training set ...")
sm = SMOTE(random_state=RANDOM_SEED)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)
print(f"      After SMOTE — class counts: {dict(zip(*np.unique(y_train_bal, return_counts=True)))}")

# ── 4. Hyper-parameter tuning (light grid) ────────────────────────────────────
print("\n[4/6] Tuning hyperparameters (GridSearchCV) ...")
param_grid = {
    "n_estimators":      [100, 200],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5],
    "class_weight":      ["balanced", None],
}

base_rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
grid_search = GridSearchCV(
    base_rf, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=1
)
grid_search.fit(X_train_bal, y_train_bal)

best_params = grid_search.best_params_
print(f"      Best params : {best_params}")
print(f"      Best CV AUC : {grid_search.best_score_:.4f}")

# ── 5. Final model ────────────────────────────────────────────────────────────
print("\n[5/6] Training final model ...")
rf = grid_search.best_estimator_

# Cross-val on balanced training data
cv_scores = cross_val_score(rf, X_train_bal, y_train_bal, cv=5, scoring="roc_auc")
print(f"      5-Fold CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── 6. Evaluation ─────────────────────────────────────────────────────────────
print("\n[6/6] Evaluating on held-out test set ...")
y_pred      = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_prob)
report = classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"])

print(f"\n      ROC-AUC : {auc:.4f}")
print("\n" + report)

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(rf, MODEL_PATH)
print(f"\n  Model saved → {MODEL_PATH}")

# ── Save metrics JSON ─────────────────────────────────────────────────────────
metrics = {
    "roc_auc":    round(auc, 4),
    "best_params": best_params,
    "cv_auc_mean": round(cv_scores.mean(), 4),
    "cv_auc_std":  round(cv_scores.std(), 4),
    "test_size":   len(y_test),
    "train_size":  len(y_train_bal),
}
with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ── Plot 1: Confusion Matrix ───────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Churned", "Churned"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
plt.close()

# ── Plot 2: ROC Curve ─────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"Random Forest (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Churn Prediction", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150)
plt.close()

# ── Plot 3: Feature Importances ───────────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 7))
colors = ["#2563EB" if v >= importances.quantile(0.75) else "#93C5FD" for v in importances]
importances.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Feature Importances", fontsize=14, fontweight="bold")
ax.set_xlabel("Mean Decrease in Impurity")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importances.png", dpi=150)
plt.close()

# ── Plot 4: Churn Distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
y_test_series = pd.Series(y_test.values, name="Churn")

counts = y_test_series.value_counts()
axes[0].bar(["Not Churned", "Churned"], counts.values, color=["#93C5FD", "#2563EB"])
axes[0].set_title("Test Set — Actual Churn", fontweight="bold")
axes[0].set_ylabel("Count")

pred_counts = pd.Series(y_pred).value_counts()
axes[1].bar(["Not Churned", "Churned"], pred_counts.values, color=["#86EFAC", "#16A34A"])
axes[1].set_title("Test Set — Predicted Churn", fontweight="bold")
axes[1].set_ylabel("Count")

plt.suptitle("Actual vs Predicted Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/churn_distribution.png", dpi=150)
plt.close()

print(f"\n  Plots saved → {OUTPUT_DIR}/")
print("\n✅  Training complete!\n")
