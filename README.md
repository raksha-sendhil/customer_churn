# ChurnCare AI

A customer churn detection platform with explainable AI. Upload a CSV of customer records, get churn risk scores ranked by priority, and receive plain-English explanations plus actionable retention advice for each customer — all powered by XGBoost and SHAP.

---

## Features

- **Risk-ranked predictions** — customers sorted from highest to lowest churn probability
- **Plain-English SHAP explanations** — no beeswarms or jargon; just "low satisfaction is raising churn risk"
- **Retention suggestions** — practical next steps generated from the top churn drivers
- **Model comparison page** — benchmarks XGBoost against Random Forest, LightGBM, and Decision Tree; automatically highlights whichever model has the best score
- **Demo sample** — a 25-row CSV (`backend/demo_sample.csv`) is included for quick testing

---

## Project structure

```
customer_churn/
├── backend/
│   ├── app.py                    # Flask API server
│   ├── xgBoost.py                # Model training, inference, SHAP explanations
│   ├── requirements.txt
│   ├── E Commerce Dataset.xlsx   # Training data
│   ├── demo_sample.csv           # 25-row sample for demos
│   ├── model_scores.json         # Model comparison scores (teammates update this)
│   └── artifacts/
│       ├── xgboost_model.joblib  # Trained model binary
│       └── xgboost_metadata.json # Feature schema and model metrics
└── frontend/
    ├── index.html
    ├── app.js
    └── styles.css
```

---

## Setup and running

### 1. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Start the server

```bash
cd backend
python app.py
```

The app will be available at **http://localhost:5000**.

> The model artifact is already trained and included in `backend/artifacts/`. The server loads it on the first request. If the artifact is missing for any reason, the server will retrain automatically from `E Commerce Dataset.xlsx`.

---

## Using the app

1. Open **http://localhost:5000** in your browser.
2. Click **Choose CSV file** and select `backend/demo_sample.csv` (or any customer CSV that matches the column schema).
3. Click **Run risk analysis**.
4. Customers appear ranked from highest to lowest churn risk.
5. Click any row to see the SHAP-based explanation and suggested retention actions.
6. Switch to **Model comparison** in the sidebar to see benchmark scores.

### Expected CSV columns

| Column | Type | Description |
|--------|------|-------------|
| CustomerID | string/int | Unique customer identifier |
| Tenure | numeric | Months with the service |
| CityTier | numeric | City tier (1/2/3) |
| WarehouseToHome | numeric | Distance from warehouse to home |
| HourSpendOnApp | numeric | Hours spent on app per month |
| NumberOfDeviceRegistered | numeric | Devices registered |
| SatisfactionScore | numeric | Customer satisfaction (1–5) |
| NumberOfAddress | numeric | Addresses saved |
| Complain | numeric | 1 if complained in last month |
| OrderAmountHikeFromlastYear | numeric | % increase in order amount YoY |
| CouponUsed | numeric | Coupons used last month |
| OrderCount | numeric | Orders placed last month |
| DaySinceLastOrder | numeric | Days since last order |
| CashbackAmount | numeric | Cashback received last month |
| PreferredLoginDevice | categorical | Computer / Mobile Phone / Phone |
| PreferredPaymentMode | categorical | CC / COD / Credit Card / Debit Card / E wallet / UPI |
| Gender | categorical | Male / Female |
| PreferedOrderCat | categorical | Fashion / Grocery / Laptop & Accessory / Mobile / Others |
| MaritalStatus | categorical | Single / Married / Divorced |

The `Churn` column is optional — it will be dropped automatically if present.

---

## Adding a teammate's model scores

When a teammate finishes training their model, they update `backend/model_scores.json`:

```json
{
  "XGBoost": 0.9423,
  "Random Forest": 0.9150,
  "LightGBM": 0.0,
  "Decision Tree": 0.0
}
```

The app dynamically picks whichever trained model has the highest score and displays it as the leading model in the comparison view. No server restart needed — scores are read on each page load.

---

## Model performance (XGBoost)

| Metric | Score |
|--------|-------|
| Accuracy | 94.23% |
| F1-score | 0.8148 |
| Precision | 0.8882 |
| Recall | 0.7526 |

Trained on the E-Commerce Dataset with an 80/20 stratified split, 250 estimators, max depth 4, learning rate 0.05.

---

## Tech stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3, Flask |
| ML model | XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Data processing | Pandas, NumPy, scikit-learn |
| Frontend | Vanilla HTML / CSS / JavaScript |
