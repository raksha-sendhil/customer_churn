import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / 'E Commerce Dataset.xlsx'
SAMPLE_FILE = BASE_DIR / 'demo_sample.csv'
MODEL_DIR = BASE_DIR / 'artifacts'
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / 'xgboost_model.joblib'
METADATA_PATH = MODEL_DIR / 'xgboost_metadata.json'
SCORES_PATH = BASE_DIR / 'model_scores.json'

# In-memory cache — populated on first request, avoids repeated disk reads.
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

    df = load_raw_dataset()
    sample = df.sample(n=25, random_state=42).copy()
    sample.to_csv(SAMPLE_FILE, index=False)

    X, y, metadata = build_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    metadata['accuracy'] = float(accuracy)
    metadata['f1_score'] = float(f1)
    metadata['precision'] = float(precision)
    metadata['recall'] = float(recall)
    metadata['label'] = 'Churn'

    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    scores = {
        'XGBoost': round(float(accuracy), 4),
        'Random Forest': 0.0,
        'LightGBM': 0.0,
        'Decision Tree': 0.0,
        'best_model': 'XGBoost',
    }
    SCORES_PATH.write_text(json.dumps(scores, indent=2), encoding='utf-8')

    _model_cache = model
    _metadata_cache = metadata
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


def prepare_input(frame: pd.DataFrame, metadata: dict):
    frame = frame.copy()
    frame = frame.drop(columns=['CustomerID'], errors='ignore')

    numeric_columns = metadata.get('numeric_columns', [])
    categorical_columns = metadata.get('categorical_columns', [])
    medians = metadata.get('medians', {})

    if numeric_columns:
        numeric_frame = frame[numeric_columns].copy()
        for col in numeric_columns:
            numeric_frame[col] = pd.to_numeric(numeric_frame[col], errors='coerce')
            fill = float(medians[col]) if col in medians else 0.0
            numeric_frame[col] = numeric_frame[col].fillna(fill)
        frame = pd.concat([numeric_frame, frame.drop(columns=numeric_columns)], axis=1)

    for col in categorical_columns:
        frame[col] = frame[col].fillna('Unknown').astype(str)

    encoded = pd.get_dummies(frame, columns=categorical_columns, dummy_na=False)
    feature_columns = metadata.get('feature_columns', [])
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded


def predict_probabilities(frame: pd.DataFrame):
    model, metadata, _ = ensure_model_ready()
    encoded = prepare_input(frame, metadata)
    probabilities = model.predict_proba(encoded)[:, 1]
    return probabilities, model, metadata, encoded


def compute_shap_values(model, encoded: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for all rows in one pass."""
    explainer = shap.TreeExplainer(model)
    shap_obj = explainer(encoded)
    values = np.asarray(shap_obj.values)
    # For multi-output models shap returns (n_samples, n_features, n_classes);
    # take the positive-class slice to get (n_samples, n_features).
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
):
    """Generate a human-readable explanation for one customer row.

    Pass pre-computed model, metadata, encoded, and shap_values to avoid
    redundant work when calling this in a loop over many rows.
    """
    if model is None or metadata is None:
        model, metadata, _ = ensure_model_ready()
    if encoded is None:
        encoded = prepare_input(frame, metadata)
    if shap_values is None:
        shap_values = compute_shap_values(model, encoded)

    row_values = shap_values[index]
    feature_names = list(encoded.columns)

    top_features = sorted(
        [
            {'feature': feature_names[i], 'impact': float(row_values[i])}
            for i in range(len(feature_names))
            if not math.isnan(row_values[i])
        ],
        key=lambda item: abs(item['impact']),
        reverse=True,
    )[:6]

    positive_features = [item for item in top_features if item['impact'] > 0]
    negative_features = [item for item in top_features if item['impact'] < 0]

    summary = []
    for item in positive_features:
        summary.append(human_explanation(item['feature'], item['impact']))
    for item in negative_features[:2]:
        summary.append(human_explanation(item['feature'], item['impact'], positive=False))

    if not summary:
        summary.append(
            'The model finds this customer is generally stable, with no single factor pushing churn risk sharply higher.'
        )

    suggestions = build_suggestions(positive_features, frame.iloc[index])
    return {
        'risk_score': round(float(model.predict_proba(encoded)[index, 1]), 4),
        'top_features': top_features,
        'summary': summary,
        'suggestions': suggestions,
        'feature_values': frame.iloc[index].to_dict(),
    }


def human_explanation(feature_name: str, impact: float, positive: bool = True):
    """Translate a SHAP contribution into plain-English guidance."""
    label = feature_name.replace('_', ' ').replace('  ', ' ')

    if positive:
        if 'Satisfaction' in feature_name:
            return 'Low satisfaction levels are raising churn risk. The customer seems less confident in the service, which can make them more likely to leave.'
        if 'Complain' in feature_name:
            return 'Recent complaints are pushing the churn risk up. Fast support follow-up and issue resolution would help rebuild trust.'
        if 'Tenure' in feature_name:
            return 'This customer has a shorter time with the brand, and that can make them more willing to switch. A welcome or loyalty touchpoint would help.'
        if 'WarehouseToHome' in feature_name or 'Distance' in feature_name:
            return 'Delivery distance is making the experience less convenient, which increases the odds of churn. Better delivery options or faster fulfilment could help.'
        if 'OrderAmountHike' in feature_name or 'Hike' in feature_name:
            return 'A recent increase in order spend is making the customer feel less comfortable, which can raise churn risk. Review pricing or offer value-based incentives.'
        if 'Coupon' in feature_name or 'Cashback' in feature_name:
            return 'The customer is not getting enough value from rewards right now, so churn risk is climbing. A targeted offer or loyalty reward may help.'
        if 'DaySinceLastOrder' in feature_name:
            return 'The customer has been inactive for a while, which is increasing churn risk. A re-engagement message or reminder could bring them back.'
        if 'PreferredPayment' in feature_name or 'Payment' in feature_name:
            return 'Payment preferences are a clue here, and the current setup may be reducing confidence. A smoother checkout experience could improve retention.'
        return f'{label} is one of the main reasons this customer looks at risk. Improving this area could reduce the chance of churn.'

    else:
        if 'Satisfaction' in feature_name:
            return "Higher satisfaction is working in the customer's favour and helping lower churn risk."
        if 'Coupon' in feature_name or 'Cashback' in feature_name:
            return "A stronger rewards pattern is helping the customer stay engaged and reducing churn risk."
        return f"{label} is helping the customer stay more stable, which is offsetting some churn pressure."


def build_suggestions(positive_features, row):
    suggestions = []
    features = {item['feature']: item['impact'] for item in positive_features}

    if any('Satisfaction' in key for key in features):
        suggestions.append('Reach out with a personalised support follow-up and ask what is affecting satisfaction.')
    if any('Complain' in key for key in features):
        suggestions.append('Resolve recent complaints quickly and send a reassurance message about the next steps.')
    if any('Tenure' in key for key in features):
        suggestions.append('Offer a loyalty or onboarding incentive to strengthen the customer relationship early.')
    if any('WarehouseToHome' in key or 'Distance' in key for key in features):
        suggestions.append('Improve delivery convenience with faster fulfilment, better tracking, or nearby pickup options.')
    if any('Coupon' in key or 'Cashback' in key for key in features):
        suggestions.append('Provide a targeted reward or discount to increase customer value perception.')
    if any('DaySinceLastOrder' in key for key in features):
        suggestions.append('Send a re-engagement campaign with recommendations or a limited-time offer.')
    if not suggestions:
        suggestions.append("Offer a tailored retention offer and monitor this customer's experience over the next few visits.")

    return suggestions


def get_model_scores():
    """Return model comparison scores, dynamically computing the current best_model."""
    model_names = ['XGBoost', 'Random Forest', 'LightGBM', 'Decision Tree']
    if SCORES_PATH.exists():
        scores = json.loads(SCORES_PATH.read_text(encoding='utf-8'))
        trained = [n for n in model_names if scores.get(n, 0) > 0]
        scores['best_model'] = max(trained, key=lambda n: scores[n]) if trained else 'XGBoost'
        return scores
    return {name: 0.0 for name in model_names} | {'best_model': 'XGBoost'}


if __name__ == '__main__':
    train_model()
    print('Model trained successfully.')
    print('Demo sample saved to', SAMPLE_FILE)
