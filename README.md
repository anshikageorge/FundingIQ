# Startup Funding Forecasting & Trend Analysis Dashboard

## Overview

Startup Funding Forecasting & Trend Analysis is an end-to-end machine learning and time-series forecasting project that analyzes historical startup funding data, identifies investment trends, evaluates multiple forecasting models, and predicts future funding activity through an interactive Streamlit dashboard.

The project is designed to simulate a real-world analytics platform used by venture capital firms, startup analysts, investors, and business intelligence teams.

---

## Features

### Data Processing

* Automated data cleaning and preprocessing
* Missing value handling
* Funding amount standardization
* Date normalization
* Industry and location cleaning

### Feature Engineering

* Year, Month, Quarter extraction
* Lag Features
* Rolling Mean Features
* Rolling Standard Deviation Features
* Growth Rate Features

### Machine Learning Models

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* Prophet Time-Series Forecasting

### Model Evaluation

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score
* Model comparison dashboard

### Forecasting

* Future funding prediction
* Historical vs Forecast comparison
* Trend analysis
* Growth projections

### Interactive Dashboard

* Executive Overview
* Data Exploration
* Funding Trends
* Model Comparison
* Model Explainability
* Future Forecasts
* Download Center
* About Project

---

## Project Structure

```text
startup_funding_forecaster/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── startup_funding.csv
│
├── models/
│   ├── best_model.pkl
│   └── prophet_model.pkl
│
├── outputs/
│   ├── funding_forecast.csv
│   ├── model_metrics.csv
│   ├── feature_importance.csv
│   └── charts/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluation.py
│   └── utils.py
│
└── api/
    └── api.py
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/startup-funding-forecasting.git

cd startup-funding-forecasting
```

### Create Virtual Environment

Windows:

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS:

```bash
python3 -m venv venv

source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

Place the dataset inside:

```text
data/startup_funding.csv
```

Expected important columns:

```text
Date
AmountInUSD
IndustryVertical
CityLocation
```

---

## Training Models

Train all machine learning models:

```bash
python src/train.py
```

Outputs:

```text
models/best_model.pkl

outputs/model_metrics.csv

outputs/feature_importance.csv
```

---

## Generate Forecasts

Run:

```bash
python src/predict.py
```

Outputs:

```text
outputs/funding_forecast.csv
```

---

## Run Dashboard

Launch Streamlit:

```bash
streamlit run app.py
```

Default URL:

```text
http://localhost:8501
```

---

## Run API

Start FastAPI server:

```bash
uvicorn api.api:app --reload
```

API URL:

```text
http://127.0.0.1:8000
```

### Available Endpoints

#### Health Check

```text
GET /health
```

#### Model Metrics

```text
GET /metrics
```

#### Forecast Data

```text
GET /forecast
```

---

## Dashboard Sections

### Executive Overview

* KPI Cards
* Total Funding
* Total Records
* Best Model
* Forecast Horizon

### Data Exploration

* Dataset Preview
* Missing Values Analysis
* Descriptive Statistics

### Funding Trends

* Historical Funding Trend
* Monthly Trends
* Quarterly Trends
* Moving Averages

### Model Comparison

* Performance Metrics
* Model Ranking
* Accuracy Comparison

### Model Explainability

* Feature Importance Analysis

### Future Forecasts

* Funding Predictions
* Forecast Visualization
* Download Forecasts

---

## Technology Stack

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* XGBoost

### Time-Series Forecasting

* Prophet

### Visualization

* Plotly
* Matplotlib
* Seaborn

### Web Frameworks

* Streamlit
* FastAPI

### Model Persistence

* Joblib

---

## Workflow

```text
Raw Startup Funding Data
            │
            ▼
Data Preprocessing
            │
            ▼
Feature Engineering
            │
            ▼
Model Training
            │
            ▼
Model Evaluation
            │
            ▼
Forecast Generation
            │
            ▼
Interactive Dashboard
```

---

## Future Improvements

* SHAP Explainability
* Industry-Level Forecasting
* Geographic Funding Analysis
* AI-Generated Insights
* PDF Report Generation
* Automated Model Retraining
* Cloud Deployment Automation

---

## Author

Developed as a Machine Learning, Forecasting, and Business Analytics portfolio project demonstrating:

* Time-Series Forecasting
* Machine Learning Model Selection
* Data Visualization
* Streamlit Dashboard Development
* FastAPI Integration
* End-to-End ML Workflow
