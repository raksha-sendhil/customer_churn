import json
import math
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'E Commerce Dataset.xlsx'
SAMPLE_FILE = BASE_DIR / 'demo_sample.csv'
MODEL_DIR = BASE_DIR / 'artifacts'
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / 'lgbm_model.joblib'
METADATA_PATH = MODEL_DIR / 'lgbm_metadata.json'

_model_cache = None
_metadata_cache = None


def _add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Six interaction features from the LightGBM training script."""
    frame = frame.copy()
    try:
        frame['spend_per_order'] = frame['CashbackAmount'] / (frame['OrderCount'] + 1)
        frame['days_per_order'] = frame['DaySinceLastOrder'] / (frame['OrderCount'] + 1)
        frame['complaint_x_inactive'] = frame['Complain'] * frame['DaySinceLastOrder']
        frame['tenure_per_order'] = frame['Tenure'] / (frame['OrderCount'] + 1)
        frame['cashback_ratio'] = frame['CashbackAmount'] / (frame['OrderAmountHikeFromlastYear'] + 1)
        frame['total_orders_value'] = frame['OrderCount'] * frame['CashbackAmount']
    except (KeyError, TypeError):
        pass
    return frame


def load_raw_dataset():
    return pd.read_excel(DATA_FILE, sheet_name='E Comm')


def build_features(frame: pd.DataFrame):
    frame = frame.copy()
    frame = frame.drop(columns=['CustomerID'], errors='ignore')
    target = frame.pop('Churn').astype(int)

    numeric_columns = frame.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = frame.select_dtypes(exclude=['number']).columns.tolist()

    # Missing-value indicators captured before imputation (training only)
    for col in ['Tenure', 'DaySinceLastOrder', 'WarehouseToHome']:
        if col in frame.columns:
            frame[col + '_missing'] = frame[col].isna().astype(int)

    numeric_medians = frame[numeric_columns].median()
    frame[numeric_columns] = frame[numeric_columns].fillna(numeric_medians)
    frame[categorical_columns] = frame[categorical_columns].fillna('Unknown').astype(str)

    frame = _add_engineered_features(frame)
    encoded = pd.get_dummies(frame, columns=categorical_columns, dummy_na=False)

    return encoded, target, {
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'medians': numeric_medians.to_dict(),
        'feature_columns': list(encoded.columns),
    }


def train_model():
    global _model_cache, _metadata_cache
    print('[LightGBM] Training model …')

    df = load_raw_dataset()

    if not SAMPLE_FILE.exists():
        df.sample(n=25, random_state=42).to_csv(SAMPLE_FILE, index=False)

    X, y, metadata = build_features(df)

    # Three-way split: 64% train / 16% val (early stopping) / 20% test (final eval)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval,
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

    # High n_estimators ceiling; early stopping on the val set decides when to stop
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    print(f'[LightGBM] Early stopping at tree {model.best_iteration_}')

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    metadata.update({
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'label': 'Churn',
    })

    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'[LightGBM] Done — F1: {f1:.4f}  Acc: {accuracy:.4f}')

    _model_cache = model
    _metadata_cache = metadata
    sample = pd.read_csv(SAMPLE_FILE) if SAMPLE_FILE.exists() else None
    return model, metadata, sample


def ensure_model_ready():
    global _model_cache, _metadata_cache

    if _model_cache is not None and _metadata_cache is not None:
        sample = pd.read_csv(SAMPLE_FILE) if SAMPLE_FILE.exists() else None
        return _model_cache, _metadata_cache, sample

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return train_model()

    _model_cache = joblib.load(MODEL_PATH)
    _metadata_cache = json.loads(METADATA_PATH.read_text(encoding='utf-8'))
    sample = pd.read_csv(SAMPLE_FILE) if SAMPLE_FILE.exists() else None
    return _model_cache, _metadata_cache, sample


def prepare_input(frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame.drop(columns=['CustomerID'], errors='ignore')

    numeric_columns = metadata.get('numeric_columns', [])
    categorical_columns = metadata.get('categorical_columns', [])
    medians = metadata.get('medians', {})

    # Missing indicators must be created before imputation
    for col in ['Tenure', 'DaySinceLastOrder', 'WarehouseToHome']:
        if col in frame.columns:
            frame[col + '_missing'] = frame[col].isna().astype(int)
        else:
            frame[col + '_missing'] = 0

    if numeric_columns:
        numeric_frame = frame[numeric_columns].copy()
        for col in numeric_columns:
            numeric_frame[col] = pd.to_numeric(numeric_frame[col], errors='coerce')
            numeric_frame[col] = numeric_frame[col].fillna(float(medians.get(col, 0.0)))
        frame = pd.concat([numeric_frame, frame.drop(columns=numeric_columns)], axis=1)

    for col in categorical_columns:
        frame[col] = frame[col].fillna('Unknown').astype(str)

    frame = _add_engineered_features(frame)
    encoded = pd.get_dummies(frame, columns=categorical_columns, dummy_na=False)
    encoded = encoded.reindex(columns=metadata.get('feature_columns', []), fill_value=0)
    return encoded


def predict_probabilities(frame: pd.DataFrame):
    model, metadata, _ = ensure_model_ready()
    encoded = prepare_input(frame, metadata)
    probabilities = model.predict_proba(encoded)[:, 1]
    return probabilities, model, metadata, encoded


def compute_shap_values(model, encoded: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    shap_obj = explainer(encoded)
    values = np.asarray(shap_obj.values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def explain_row(
    frame: pd.DataFrame,
    index: int = 0,
    *,
    model=None,
    metadata=None,
    encoded=None,
    shap_values=None,
    row_probability=None,
):
    from xgBoost import build_suggestions, human_explanation

    if model is None or metadata is None:
        model, metadata, _ = ensure_model_ready()
    if encoded is None:
        encoded = prepare_input(frame, metadata)
    if shap_values is None:
        shap_values = compute_shap_values(model, encoded)

    row_vals = shap_values[index]
    feature_names = list(encoded.columns)

    top_features = sorted(
        [
            {'feature': feature_names[i], 'impact': float(row_vals[i])}
            for i in range(len(feature_names))
            if not math.isnan(row_vals[i])
        ],
        key=lambda x: abs(x['impact']),
        reverse=True,
    )[:6]

    positive_features = [f for f in top_features if f['impact'] > 0]
    negative_features = [f for f in top_features if f['impact'] < 0]

    summary = [human_explanation(f['feature'], f['impact']) for f in positive_features]
    summary += [human_explanation(f['feature'], f['impact'], positive=False) for f in negative_features[:2]]

    if not summary:
        summary.append(
            'The model finds this customer is generally stable, with no single factor pushing churn risk sharply higher.'
        )

    risk = row_probability if row_probability is not None else float(model.predict_proba(encoded)[index, 1])
    return {
        'risk_score': round(risk, 4),
        'top_features': top_features,
        'summary': summary,
        'suggestions': build_suggestions(positive_features, frame.iloc[index]),
        'feature_values': frame.iloc[index].to_dict(),
    }


if __name__ == '__main__':
    train_model()
