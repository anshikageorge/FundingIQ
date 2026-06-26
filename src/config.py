from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"

DATASET_NAME = "startup_funding.csv"

DATA_PATH = DATA_DIR / DATASET_NAME

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
PROPHET_MODEL_PATH = MODELS_DIR / "prophet_model.pkl"

METRICS_PATH = OUTPUTS_DIR / "model_metrics.csv"
FORECAST_PATH = OUTPUTS_DIR / "funding_forecast.csv"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"

AGGREGATION_LEVEL = "ME"
TARGET_COLUMN = "AmountInUSD"
DATE_COLUMN = "Date"

RANDOM_STATE = 42
TEST_SIZE_MONTHS = 6