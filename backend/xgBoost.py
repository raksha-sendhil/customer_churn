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

_model_cache = None
_metadata_cache = None


def load_raw_dataset():
    return pd.read_excel(DATA_FILE, sheet_name='E Comm')


def _add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Same six interaction features used by LightGBM — keeps the comparison fair."""
    frame = frame.copy()
    try:
        frame['spend_per_order']      = frame['CashbackAmount'] / (frame['OrderCount'] + 1)
        frame['days_per_order']       = frame['DaySinceLastOrder'] / (frame['OrderCount'] + 1)
        frame['complaint_x_inactive'] = frame['Complain'] * frame['DaySinceLastOrder']
        frame['tenure_per_order']     = frame['Tenure'] / (frame['OrderCount'] + 1)
        frame['cashback_ratio']       = frame['CashbackAmount'] / (frame['OrderAmountHikeFromlastYear'] + 1)
        frame['total_orders_value']   = frame['OrderCount'] * frame['CashbackAmount']
    except (KeyError, TypeError):
        pass
    return frame


def build_features(frame: pd.DataFrame):
    frame = frame.copy()
    frame = frame.drop(columns=['CustomerID'], errors='ignore')
    target = frame.pop('Churn').astype(int)

    numeric_columns = frame.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = frame.select_dtypes(exclude=['number']).columns.tolist()

    # Missing indicators before imputation (same as LightGBM)
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

    df = load_raw_dataset()
    sample = df.sample(n=25, random_state=42).copy()
    sample.to_csv(SAMPLE_FILE, index=False)

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

    model = XGBClassifier(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=100,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f'[XGBoost] Early stopping at tree {model.best_iteration}')

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

    # Write model_scores.json using F1 as the ranking metric
    scores = {
        'XGBoost': round(float(f1), 4),
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

    # Missing indicators before imputation (same order as training)
    for col in ['Tenure', 'DaySinceLastOrder', 'WarehouseToHome']:
        if col in frame.columns:
            frame[col + '_missing'] = frame[col].isna().astype(int)
        else:
            frame[col + '_missing'] = 0

    if numeric_columns:
        numeric_frame = frame[numeric_columns].copy()
        for col in numeric_columns:
            numeric_frame[col] = pd.to_numeric(numeric_frame[col], errors='coerce')
            fill = float(medians[col]) if col in medians else 0.0
            numeric_frame[col] = numeric_frame[col].fillna(fill)
        frame = pd.concat([numeric_frame, frame.drop(columns=numeric_columns)], axis=1)

    for col in categorical_columns:
        frame[col] = frame[col].fillna('Unknown').astype(str)

    frame = _add_engineered_features(frame)
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

    risk = row_probability if row_probability is not None else float(model.predict_proba(encoded)[index, 1])
    return {
        'risk_score': round(risk, 4),
        'top_features': top_features,
        'summary': summary,
        'suggestions': build_suggestions(positive_features, frame.iloc[index]),
        'feature_values': frame.iloc[index].to_dict(),
    }


def human_explanation(feature_name: str, _impact: float, positive: bool = True):
    """Translate a SHAP contribution into plain-English guidance."""
    label = feature_name.replace('_', ' ').replace('  ', ' ')

    if positive:
        if 'Satisfaction' in feature_name:
            return 'Low satisfaction levels are raising churn risk. The customer seems less confident in the service, which can make them more likely to leave.'
        if 'Complain' in feature_name or 'complaint_x_inactive' in feature_name:
            return 'Recent complaints are pushing the churn risk up. Fast support follow-up and issue resolution would help rebuild trust.'
        if 'Tenure' in feature_name or 'tenure_per_order' in feature_name:
            return 'This customer has a shorter time with the brand, and that can make them more willing to switch. A welcome or loyalty touchpoint would help.'
        if 'WarehouseToHome' in feature_name or 'Distance' in feature_name:
            return 'Delivery distance is making the experience less convenient, which increases the odds of churn. Better delivery options or faster fulfilment could help.'
        if 'OrderAmountHike' in feature_name or 'Hike' in feature_name:
            return 'A recent increase in order spend is making the customer feel less comfortable, which can raise churn risk. Review pricing or offer value-based incentives.'
        if 'Coupon' in feature_name or 'Cashback' in feature_name or 'spend_per_order' in feature_name or 'cashback_ratio' in feature_name or 'total_orders_value' in feature_name:
            return 'The customer is not getting enough value from rewards right now, so churn risk is climbing. A targeted offer or loyalty reward may help.'
        if 'DaySinceLastOrder' in feature_name or 'days_per_order' in feature_name:
            return 'The customer has been inactive for a while, which is increasing churn risk. A re-engagement message or reminder could bring them back.'
        if 'PreferredPayment' in feature_name or 'Payment' in feature_name:
            return 'Payment preferences are a clue here, and the current setup may be reducing confidence. A smoother checkout experience could improve retention.'
        return f'{label} is one of the main reasons this customer looks at risk. Improving this area could reduce the chance of churn.'

    else:
        if 'Satisfaction' in feature_name:
            return "Higher satisfaction is working in the customer's favour and helping lower churn risk."
        if 'Coupon' in feature_name or 'Cashback' in feature_name or 'spend_per_order' in feature_name or 'cashback_ratio' in feature_name:
            return "A stronger rewards pattern is helping the customer stay engaged and reducing churn risk."
        if 'Tenure' in feature_name or 'tenure_per_order' in feature_name:
            return "A longer relationship with the brand is working in this customer's favour, helping keep churn risk low."
        return f"{label} is helping the customer stay more stable, which is offsetting some churn pressure."


def build_suggestions(positive_features, _row):
    suggestions = []
    features = {item['feature']: item['impact'] for item in positive_features}

    if any('Satisfaction' in key for key in features):
        suggestions.append('Reach out with a personalised support follow-up and ask what is affecting satisfaction.')
    if any('Complain' in key or 'complaint_x_inactive' in key for key in features):
        suggestions.append('Resolve recent complaints quickly and send a reassurance message about the next steps.')
    if any('Tenure' in key or 'tenure_per_order' in key for key in features):
        suggestions.append('Offer a loyalty or onboarding incentive to strengthen the customer relationship early.')
    if any('WarehouseToHome' in key or 'Distance' in key for key in features):
        suggestions.append('Improve delivery convenience with faster fulfilment, better tracking, or nearby pickup options.')
    if any('Coupon' in key or 'Cashback' in key or 'spend_per_order' in key or 'cashback_ratio' in key or 'total_orders_value' in key for key in features):
        suggestions.append('Provide a targeted reward or discount to increase customer value perception.')
    if any('DaySinceLastOrder' in key or 'days_per_order' in key for key in features):
        suggestions.append('Send a re-engagement campaign with recommendations or a limited-time offer.')
    if not suggestions:
        suggestions.append("Offer a tailored retention offer and monitor this customer's experience over the next few visits.")

    return suggestions


def get_model_scores():
    """Return scores and full metrics for all models by reading their metadata files.

    Uses accuracy as the main ranking metric (also exposes F1, precision, recall).
    """
    model_names = ['XGBoost', 'Random Forest', 'LightGBM', 'Decision Tree']
    metadata_paths = {
        'XGBoost': MODEL_DIR / 'xgboost_metadata.json',
        'Decision Tree': MODEL_DIR / 'decision_tree_metadata.json',
        'LightGBM': MODEL_DIR / 'lgbm_metadata.json',
        'Random Forest': MODEL_DIR / 'random_forest_metadata.json',
    }

    scores = {}
    all_metrics = {}

    for name in model_names:
        path = metadata_paths.get(name)
        if path and path.exists():
            meta = json.loads(path.read_text(encoding='utf-8'))
            accuracy = float(meta.get('accuracy', 0.0))
            scores[name] = round(accuracy, 4)
            all_metrics[name] = {
                'accuracy': round(accuracy, 4),
                'f1': round(float(meta.get('f1_score', 0.0)), 4),
                'precision': round(float(meta.get('precision', 0.0)), 4),
                'recall': round(float(meta.get('recall', 0.0)), 4),
            }
        else:
            scores[name] = 0.0

    trained = [n for n in model_names if scores.get(n, 0) > 0]
    scores['best_model'] = max(trained, key=lambda n: scores[n]) if trained else 'XGBoost'
    scores['metrics'] = all_metrics
    return scores


if __name__ == '__main__':
    train_model()
    print('Model trained successfully.')
    print('Demo sample saved to', SAMPLE_FILE)
