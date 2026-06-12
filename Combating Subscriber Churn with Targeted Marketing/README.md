# Combating Subscriber Churn with Targeted Marketing

Predicting which subscribers are likely to churn on **AZ Watch** — a video streaming platform for educational content — using supervised classification and unsupervised clustering to support data-driven marketing strategies.

---

## Problem Statement

AZ Watch wants to reduce subscriber churn by identifying at-risk users before they leave, and by segmenting the subscriber base into meaningful personas to guide targeted retention campaigns.

---

## Dataset

**File:** `data/AZWatch_subscribers.csv` — 1,000 subscriber records

| Column | Description |
|---|---|
| `subscriber_id` | Unique subscriber identifier |
| `age_group` | Subscriber's age group |
| `engagement_time` | Avg. session duration (minutes) |
| `engagement_frequency` | Avg. weekly login sessions over the past year |
| `subscription_status` | Target label: `subscribed` or `churned` |

---

## Approach

### 1. Preprocessing
- Ordinal encoding for `age_group` (`18-24` → `55+`)
- Label encoding for the target variable
- Standard scaling for Logistic Regression features

### 2. Churn Prediction (Classification)

Three models trained on an 80/20 train-test split:

| Model | Accuracy |
|---|---|
| Logistic Regression | **91.50%** ✅ |
| Random Forest | 89.50% |
| Decision Tree | 87.50% |

> **Best model: Logistic Regression** with 91.50% accuracy.

### 3. Customer Segmentation (Clustering)

K-Means clustering (`k=3`) on `engagement_time` and `engagement_frequency`:

| Cluster | Avg. Session (min) | Avg. Weekly Sessions | Profile |
|---|---|---|---|
| 0 | 9 | 9 | High time, moderate frequency — deep learners |
| 1 | 4 | 5 | Low engagement — churn risk |
| 2 | 7 | 18 | High frequency, moderate time — power users |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Viz-4C72B0)

- **Pandas** — data loading and feature engineering
- **scikit-learn** — classification models, K-Means, preprocessing, evaluation
- **Seaborn / Matplotlib** — visualization

---

## Project Structure

```
Combating Subscriber Churn with Targeted Marketing/
│
├── data/
│   └── AZWatch_subscribers.csv
├── notebook.ipynb
└── README.md
```

---

## Key Takeaways

- Logistic Regression outperforms tree-based models on this dataset despite its simplicity, suggesting the churn signal is largely linear.
- Cluster 1 (low engagement on both dimensions) is the highest churn-risk segment — a natural target for re-engagement campaigns.
- Cluster 2 (power users) represents AZ Watch's most loyal base and may be leveraged for referral or ambassador programs.
