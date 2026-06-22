import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'E Commerce Dataset.xlsx'
SAMPLE_FILE = BASE_DIR / 'demo_sample.csv'
MODEL_DIR = BASE_DIR / 'artifacts'
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / 'random_forest_model.joblib'
METADATA_PATH = MODEL_DIR / 'random_forest_metadata.json'

_model_cache = None
_metadata_cache = None


def load_raw_dataset():
    return pd.read_excel(DATA_FILE, sheet_name='E Comm')


def build_features(frame: pd.DataFrame):
    frame = frame.copy()
    frame = frame.drop(columns=['CustomerID'], errors='ignore')
    target = frame.pop('Churn').astype(int)

    numeric_columns = frame.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = frame.select_dtypes(exclude=['number']).columns.tolist()

    numeric_medians = frame[numeric_columns].median()
    frame[numeric_columns] = frame[numeric_columns].fillna(numeric_medians)
    frame[categorical_columns] = frame[categorical_columns].fillna('Unknown').astype(str)

    encoded = pd.get_dummies(frame, columns=categorical_columns, dummy_na=False)
    return encoded, target, {
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'medians': numeric_medians.to_dict(),
        'feature_columns': list(encoded.columns),
    }


def train_model():
    global _model_cache, _metadata_cache
    print('[Random Forest] Training model …')

    df = load_raw_dataset()

    if not SAMPLE_FILE.exists():
        df.sample(n=25, random_state=42).to_csv(SAMPLE_FILE, index=False)

    X, y, metadata = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

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
    print(f'[Random Forest] Done — F1: {f1:.4f}  Acc: {accuracy:.4f}')

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

    if numeric_columns:
        numeric_frame = frame[numeric_columns].copy()
        for col in numeric_columns:
            numeric_frame[col] = pd.to_numeric(numeric_frame[col], errors='coerce')
            numeric_frame[col] = numeric_frame[col].fillna(float(medians.get(col, 0.0)))
        frame = pd.concat([numeric_frame, frame.drop(columns=numeric_columns)], axis=1)

    for col in categorical_columns:
        frame[col] = frame[col].fillna('Unknown').astype(str)

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

    _exclude = {'MaritalStatus', 'NumberOfAddress'}
    positive_for_text = [f for f in positive_features if not any(x in f['feature'] for x in _exclude)]
    negative_for_text = [f for f in negative_features if not any(x in f['feature'] for x in _exclude)]

    summary = [human_explanation(f['feature'], f['impact']) for f in positive_for_text]
    summary += [human_explanation(f['feature'], f['impact'], positive=False) for f in negative_for_text[:2]]
    summary = list(dict.fromkeys(summary))

    if not summary:
        summary.append(
            'The model finds this customer is generally stable, with no single factor pushing churn risk sharply higher.'
        )

    risk = row_probability if row_probability is not None else float(model.predict_proba(encoded)[index, 1])
    return {
        'risk_score': round(risk, 4),
        'top_features': top_features,
        'summary': summary,
        'suggestions': build_suggestions(positive_for_text, frame.iloc[index]),
        'feature_values': frame.iloc[index].to_dict(),
    }


if __name__ == '__main__':
    train_model()
