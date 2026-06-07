#  Monitoring A Financial Fraud Detection Model

> **MLOps · Model Monitoring · Drift Detection**

A real-world MLOps task: diagnosing *why* a fraud detection model's accuracy degrades in production — using statistical monitoring techniques on live transaction data from a fictional UK bank, **Poundbank**.

---

## 🧩 Problem Statement

Banks rely on ML models to detect fraud in real-time. But data patterns shift over time — user behavior changes, transaction volumes fluctuate — and models silently become less reliable.

This project answers:
- **When** did the model start underperforming?
- **Which features** drifted most significantly?
- **What** changed in the production data to cause it?

---

## 📊 Dataset

| Dataset | Records | Period |
|---|---|---|
| `reference.csv` (test/baseline) | 50,207 transactions | Jan – Oct 2018 |
| `analysis.csv` (production) | 39,967 transactions | Nov 2018 – Jun 2019 |

**Features:** `timestamp`, `transaction_amount`, `transaction_type`, `time_since_login_min`, `user_tenure_months`, `is_first_transaction`, `is_fraud`, `predicted_fraud_proba`, `predicted_fraud`

- Baseline model accuracy: **94.36%**
- Fraud rate in reference data: **~50%** (balanced dataset)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NannyML](https://img.shields.io/badge/NannyML-MLOps-blueviolet?style=flat)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

## 🔬 Methodology

### 1. Estimated Performance (CBPE)
Used **Confidence-Based Performance Estimation (CBPE)** from `nannyml` to estimate model accuracy *without ground truth labels* — simulating a real production environment where true labels arrive with a delay.

### 2. Realized Performance
Calculated actual accuracy once true labels (`is_fraud`) became available, and compared against CBPE estimates to validate the monitoring approach.

### 3. Univariate Drift Detection
Applied:
- **Kolmogorov-Smirnov test** for continuous features (`transaction_amount`, `time_since_login_min`, `user_tenure_months`)
- **Chi-squared test** for categorical features (`transaction_type`, `is_first_transaction`)

Drift was calculated monthly and cross-referenced with the performance drop period.

### 4. Statistical Summary Monitoring
Tracked monthly average `transaction_amount` using `nannyml.SummaryStatsAvgCalculator` to identify statistically significant anomalies.

---

## 📈 Key Findings

- 📉 **Performance alerts fired** in multiple months during Apr–Jun 2019, confirmed by both estimated and realized accuracy
- 🔄 **`transaction_amount`** showed the highest correlation with performance degradation — drift alerts fired consistently during the critical period
- ⚠️ A statistically anomalous spike in average transaction amount was detected in a specific month, indicating unusual production behavior

---

## 📁 Project Structure

```
Monitoring A Financial Fraud Detection Model/
├── notebook.ipynb       # Full analysis & monitoring pipeline
├── reference.csv        # Baseline / test data (Jan–Oct 2018)
├── analysis.csv         # Production data (Nov 2018–Jun 2019)
└── cover_image.jpg      # Project cover
```

---

## 💡 What I Learned

- How to use **CBPE** to monitor model performance without waiting for ground truth labels
- Applying **statistical drift tests** (KS, Chi²) to identify feature distribution shifts
- Interpreting monitoring alerts to trace root causes of model degradation in production
- Working with the `nannyml` library for end-to-end ML monitoring workflows
