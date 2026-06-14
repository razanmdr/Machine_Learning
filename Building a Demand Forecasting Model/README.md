# Demand Forecasting for End-of-Year Sales Planning

Forecasting product demand for a multinational e-commerce company using **PySpark** and a **Random Forest Regressor** — helping the Sales & Operations Planning (S&OP) team manage inventory and plan promotions ahead of peak season.

---

## Problem Statement

Uncertainty in supply chains leads to stockouts, delayed deliveries, and rising operational costs. This project builds a scalable demand forecasting pipeline that predicts weekly sales quantities per product and country, enabling smarter inventory decisions before the year-end sales surge.

---

## Dataset

**File:** `Online_Retail.csv` — ~384,000 transaction records

| Column | Description |
|---|---|
| `InvoiceNo` | Unique 6-digit transaction ID |
| `StockCode` | Unique 5-digit product code |
| `Description` | Product name |
| `Quantity` | Units sold per transaction |
| `UnitPrice` | Price per unit |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer's country |
| `InvoiceDate` | Transaction timestamp |
| `Year` / `Month` / `Week` / `Day` / `DayOfWeek` | Extracted time features |

---

## Approach

### 1. Data Aggregation
Daily quantity per `Country` × `StockCode` aggregated using PySpark, then time features re-extracted from `InvoiceDate`.

### 2. Train / Test Split
Temporal split on **2011-09-25** — all transactions before this date form the training set (175,452 records), everything after is the test set.

### 3. Feature Engineering
- Categorical encoding: `Country` and `StockCode` via `StringIndexer`
- Features assembled: `CountryIndex`, `StockCodeIndex`, `Year`, `Month`, `Week`, `Day`, `DayOfWeek`

### 4. Modeling — Random Forest Regressor (PySpark ML Pipeline)

| Parameter | Value |
|---|---|
| `numTrees` | 100 |
| `maxBins` | 4000 |
| `labelCol` | `Quantity` |

### 5. Results

| Metric | Value |
|---|---|
| MAE (test set) | **9.36** |
| Estimated quantity sold — Week 39, 2011 | **88,223 units** |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
![PySpark](https://img.shields.io/badge/PySpark-ML-E25A1C?logo=apachespark&logoColor=white)
![Spark MLlib](https://img.shields.io/badge/Spark_MLlib-Pipeline-orange)

- **PySpark** — distributed data processing and ML pipeline
- **Spark MLlib** — `RandomForestRegressor`, `VectorAssembler`, `StringIndexer`, `RegressionEvaluator`
- **Pandas** — intermediate inspection of training data

---

## Project Structure

```
Demand Forecasting/
│
├── Online Retail.csv
├── notebook.ipynb
└── README.md
```

---

## Key Takeaways

- A PySpark ML Pipeline keeps preprocessing and modeling reproducible and scalable across large retail datasets.
- Temporal train/test splitting (vs. random) is critical for time-series forecasting to avoid data leakage.
- The model estimates ~88K units sold in Week 39 of 2011, giving the S&OP team a concrete planning baseline for stock procurement and logistics.
