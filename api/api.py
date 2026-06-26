from fastapi import FastAPI
import pandas as pd

from src.config import METRICS_PATH, FORECAST_PATH

app = FastAPI(title="Startup Funding Forecast API")

@app.get("/health")
def health():
    return {"status": "online"}

@app.get("/metrics")
def metrics():
    return pd.read_csv(
        METRICS_PATH,
        index_col=0
    ).to_dict(orient="index")

@app.get("/forecast")
def forecast():
    return pd.read_csv(
        FORECAST_PATH
    ).to_dict(orient="records")