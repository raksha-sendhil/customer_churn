import math
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from xgBoost import (
    compute_shap_values,
    ensure_model_ready,
    explain_row,
    get_model_scores,
    predict_probabilities,
)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / 'frontend'

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')


def to_serializable(value):
    # NaN / Inf are not valid JSON — convert to None (→ null).
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [to_serializable(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


@app.route('/')
def index():
    return send_from_directory(str(FRONTEND_DIR), 'index.html')


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/model-scores')
def model_scores():
    return jsonify(get_model_scores())


@app.route('/api/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if file is None or file.filename == '':
        return jsonify({'error': 'Please upload a CSV file.'}), 400

    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return jsonify({'error': f'Unable to read CSV file: {exc}'}), 400

    if df.empty:
        return jsonify({'error': 'The uploaded file does not contain any rows.'}), 400

    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])

    # Compute probabilities, encode, and SHAP values once for the whole batch.
    probabilities, model, metadata, encoded = predict_probabilities(df)
    shap_vals = compute_shap_values(model, encoded)

    scores = get_model_scores()
    best_model_name = scores.get('best_model', 'XGBoost')

    results = []
    for idx, risk in enumerate(probabilities):
        explanation = explain_row(
            df, idx,
            model=model,
            metadata=metadata,
            encoded=encoded,
            shap_values=shap_vals,
        )
        risk_score = float(risk)
        if risk_score >= 0.75:
            level = 'Critical'
        elif risk_score >= 0.50:
            level = 'High'
        elif risk_score >= 0.30:
            level = 'Moderate'
        else:
            level = 'Low'

        customer_id = df.iloc[idx].get('CustomerID', f'Customer {idx + 1}')
        results.append(to_serializable({
            'row': idx + 1,
            'customerId': str(customer_id),
            'riskScore': round(risk_score, 4),
            'riskLevel': level,
            'summary': explanation['summary'],
            'suggestions': explanation['suggestions'],
            'topFeatures': explanation['top_features'],
            'featureValues': explanation['feature_values'],
        }))

    results.sort(key=lambda item: item['riskScore'], reverse=True)

    return jsonify({
        'message': 'Prediction complete.',
        'totalRows': len(results),
        'bestModel': best_model_name,
        'predictions': results,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
