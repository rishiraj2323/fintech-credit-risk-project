# Fintech Credit Risk — Loan Default Prediction

A machine learning system that predicts the probability a loan applicant will default, built on the "Give Me Some Credit" dataset (~150K records). Includes a tuned XGBoost model, a Flask REST API, and a live web demo.

**Live demo:** https://fintech-credit-risk-project.onrender.com
**API endpoint:** `POST /predict`

> Hosted on Render's free tier — if the app hasn't been visited recently, the first request can take 30–60 seconds while the server wakes up. Subsequent requests are fast.

---

## Overview

This project walks the full path from raw data to a deployed, usable tool:

1. Cleaned and prepared ~150,000 rows of consumer credit data
2. Engineered new features to capture repayment behaviour
3. Compared multiple models under class imbalance
4. Tuned a final XGBoost classifier
5. Served it behind a REST API
6. Deployed it publicly with a browser-based demo

## Dataset

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle) — ~150,000 anonymized borrower records with a binary label for serious delinquency within two years.

**Cleaning steps:**
- Handled missing values
- Capped outliers
- Removed placeholder-coded rows (e.g. sentinel values used for missing data)

## Feature Engineering

Three features were added on top of the original 10:

| Feature | Description |
|---|---|
| `TotalTimesLate` | Sum of all late-payment counts (30–59, 60–89, 90+ days) |
| `IncomePerDependent` | Monthly income divided by number of dependents |
| `EverSeriouslyLate` | Binary flag for any 90+ day late payment on record |

## Model

Three models were compared under class-imbalance handling (`class_weight`, threshold tuning):

| Model | Notes |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | — |
| **XGBoost (tuned)** | **Final model** |

**Final performance:**
- ROC-AUC: **~0.86**
- Recall (defaulters): **~0.77**

Artifacts: `fintech_credit_model.pkl` (model), `fintech_scaler.pkl` (StandardScaler).

## API

### `POST /predict`

**Request body:**
```json
{
  "features": [0.5, 45, 0, 0.3, 5000, 8, 0, 1, 0, 2, 0, 2500, 0]
}
```

Features must be provided in this exact order:

1. `RevolvingUtilizationOfUnsecuredLines`
2. `age`
3. `NumberOfTime30-59DaysPastDueNotWorse`
4. `DebtRatio`
5. `MonthlyIncome`
6. `NumberOfOpenCreditLinesAndLoans`
7. `NumberOfTimes90DaysLate`
8. `NumberRealEstateLoansOrLines`
9. `NumberOfTime60-89DaysPastDueNotWorse`
10. `NumberOfDependents`
11. `TotalTimesLate`
12. `IncomePerDependent`
13. `EverSeriouslyLate`

**Response:**
```json
{
  "default_probability": 0.3487,
  "prediction": 1
}
```

`prediction` is `1` (likely to default) if `default_probability >= 0.25`, else `0`.

**Example (curl):**
```bash
curl -X POST https://fintech-credit-risk-project.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 45, 0, 0.3, 5000, 8, 0, 1, 0, 2, 0, 2500, 0]}'
```

## Web Demo

The homepage (`/`) serves a browser-based form — enter an applicant's details and get a live risk verdict without touching the API directly. It computes the three engineered features automatically from the base inputs.

## Tech Stack

- **Modeling:** scikit-learn, XGBoost, pandas, NumPy
- **API:** Flask, Gunicorn
- **Deployment:** Render (free tier)
- **Frontend:** vanilla HTML/CSS/JS (no framework)

## Project Structure

```
├── app.py                        # Flask API + web demo route
├── templates/
│   └── index.html                # Browser demo page
├── fintech_credit_model.pkl      # Trained XGBoost model
├── fintech_scaler.pkl            # Fitted StandardScaler
├── fintech_credit_risk.ipynb     # Data cleaning, feature engineering, model training
├── requirements.txt
├── Procfile                      # Render/Heroku-style start command
├── cs-training.csv / cs-test.csv # Source data
└── Data Dictionary.xls           # Feature descriptions
```

## Running Locally

```bash
git clone https://github.com/rishiraj2323/-fintech-credit-risk-project.git
cd -fintech-credit-risk-project
pip install -r requirements.txt
python app.py
```

The app will run at `http://127.0.0.1:5000`.

## Limitations & Notes

- Hosted on a free-tier instance — it sleeps after 15 minutes of inactivity.
- The `fintech_scaler.pkl` file was saved with an older scikit-learn version than some environments run; a harmless `InconsistentVersionWarning` may appear in logs but does not affect predictions.
- This is a portfolio/demonstration project and is not intended for real lending decisions.
