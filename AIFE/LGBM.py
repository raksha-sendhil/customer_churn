import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import optuna
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, fbeta_score, precision_recall_curve
)
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ========================= CONFIG =========================
RANDOM_STATE = 42
N_OPTUNA_TRIALS = 100
DATA_PATH = "ecommerce_churn/E Commerce Dataset.xlsx"
SHEET_NAME = "E Comm"
OUTPUT_DIR = "."

np.random.seed(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

# ─────────────────────────────────────────────
# 1. Load & clean
# ─────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{DATA_PATH}'.\n"
        "Download from: https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction"
    )

df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
df = df.drop(columns=["CustomerID"])
df = df.drop_duplicates().reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['Churn'].mean():.4f}")

target_col = "Churn"
X = df.drop(columns=[target_col])
y = df[target_col]

# ─────────────────────────────────────────────
# 2. Missing values + Feature Engineering
# ─────────────────────────────────────────────
# Handle missing values
numeric_cols = X.select_dtypes(include=["number"]).columns
for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

# Missing indicators for important columns
for col in ['Tenure', 'DaySinceLastOrder', 'WarehouseToHome']:
    if col in X.columns:
        X[col + '_missing'] = X[col].isna().astype(int)

# Feature Engineering
X = X.copy()
X["spend_per_order"] = X["CashbackAmount"] / (X["OrderCount"] + 1)
X["days_per_order"] = X["DaySinceLastOrder"] / (X["OrderCount"] + 1)
X["complaint_x_inactive"] = X["Complain"] * X["DaySinceLastOrder"]
X["tenure_per_order"] = X["Tenure"] / (X["OrderCount"] + 1)
X["cashback_ratio"] = X["CashbackAmount"] / (X["OrderAmountHikeFromlastYear"] + 1)
X["total_orders_value"] = X["OrderCount"] * X["CashbackAmount"]

print("Added missing value handling + 6 engineered features.")

# ─────────────────────────────────────────────
# 3. Encode categoricals
# ─────────────────────────────────────────────
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype("category")

print(f"Categorical columns: {cat_cols}")

# ─────────────────────────────────────────────
# 4. Three-way split
# ─────────────────────────────────────────────
X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.2, random_state=RANDOM_STATE, stratify=y_dev
)

print(f"\nDev: {len(X_dev)} | Holdout: {len(X_holdout)}")
print(f"Train: {len(X_train)} | Val: {len(X_val)}")

# ─────────────────────────────────────────────
# 5. Class imbalance weight
# ─────────────────────────────────────────────
neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.3f} (neg={neg}, pos={pos})")

# ─────────────────────────────────────────────
# 6. Optuna Hyperparameter Tuning
# ─────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "random_state": RANDOM_STATE,
        "verbose": -1,
    }

    model = LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
        categorical_feature=cat_cols,
    )

    proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, proba)


print(f"\nRunning Optuna ({N_OPTUNA_TRIALS} trials)...")
study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
)
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_params.update({
    "n_estimators": 1000,
    "scale_pos_weight": scale_pos_weight,
    "random_state": RANDOM_STATE,
    "verbose": -1,
})

print(f"\nBest val AUC: {study.best_value:.4f}")
print(f"Best params: {best_params}")

# ─────────────────────────────────────────────
# 7. Cross-validation on dev set
# ─────────────────────────────────────────────
print("\nRunning 5-fold CV on dev set...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_dev, y_dev), 1):
    X_cv_train = X_dev.iloc[train_idx]
    X_cv_val = X_dev.iloc[val_idx]
    y_cv_train = y_dev.iloc[train_idx]
    y_cv_val = y_dev.iloc[val_idx]

    fold_model = LGBMClassifier(**best_params)
    fold_model.fit(
        X_cv_train, y_cv_train,
        eval_set=[(X_cv_val, y_cv_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
        categorical_feature=cat_cols,
    )
    auc = roc_auc_score(y_cv_val, fold_model.predict_proba(X_cv_val)[:, 1])
    cv_scores.append(auc)
    print(f" Fold {fold}: AUC = {auc:.4f}")

cv_scores = np.array(cv_scores)
print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 8. Final model
# ─────────────────────────────────────────────
print("\nTraining final model on full dev set...")
final_model = LGBMClassifier(**best_params)
final_model.fit(
    X_dev, y_dev,
    eval_set=[(X_holdout, y_holdout)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(50),
    ],
    categorical_feature=cat_cols,
)

# Save model
model_path = os.path.join(OUTPUT_DIR, "lgbm_churn_final.model")
final_model.booster_.save_model(model_path)
print(f"Model saved to: {model_path}")

# ─────────────────────────────────────────────
# 9. Threshold tuning on validation set (no leakage)
# ─────────────────────────────────────────────
val_proba = final_model.predict_proba(X_val)[:, 1]
prec, rec, thresh = precision_recall_curve(y_val, val_proba)
f1 = 2 * prec * rec / (prec + rec + 1e-8)
best_idx = f1[:-1].argmax()
best_threshold = thresh[best_idx]

print(f"\nOptimal threshold (from val): {best_threshold:.3f}")

# ─────────────────────────────────────────────
# 10. Holdout Evaluation
# ─────────────────────────────────────────────
holdout_proba = final_model.predict_proba(X_holdout)[:, 1]
y_pred_default = (holdout_proba >= 0.5).astype(int)
y_pred_tuned = (holdout_proba >= best_threshold).astype(int)

print("\n" + "="*60)
print("HOLDOUT RESULTS")
print("="*60)

for name, preds in [("Default (0.50)", y_pred_default),
                    (f"Tuned ({best_threshold:.3f})", y_pred_tuned)]:
    print(f"\n── {name} ──")
    print(classification_report(y_holdout, preds, target_names=["No Churn", "Churn"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_holdout, preds))
    print(f"F2 Score: {fbeta_score(y_holdout, preds, beta=2):.4f}")

print("\n── Threshold-independent metrics ──")
print(f"ROC AUC : {roc_auc_score(y_holdout, holdout_proba):.4f}")
print(f"PR  AUC : {average_precision_score(y_holdout, holdout_proba):.4f}")

# ─────────────────────────────────────────────
# 11. SHAP Analysis
# ─────────────────────────────────────────────
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_holdout)

if isinstance(shap_values, list):
    sv = shap_values[1]
elif len(shap_values.shape) == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

# Summary plot
shap.summary_plot(sv, X_holdout, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=200, bbox_inches="tight")
plt.close()

# Bar plot
shap.summary_plot(sv, X_holdout, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_importance_bar.png"), dpi=200, bbox_inches="tight")
plt.close()

print("SHAP plots saved: shap_summary.png and shap_importance_bar.png")