#writefile startup_funding_forecaster/src/train.py
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from prophet import Prophet
from startup_funding_forecaster.src.config import BEST_MODEL_PATH, PROPHET_MODEL_PATH, METRICS_PATH, CHARTS_DIR, FEATURE_IMPORTANCE_PATH

class ModelTrainer:
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        y_true_safe = np.where(y_true == 0, 1e-5, y_true)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

    def run_pipeline(self, ts_data):
        feature_cols = [c for c in ts_data.columns if c != "AmountInUSD"]
        split_idx = len(ts_data) - 6

        train, test = ts_data.iloc[:split_idx], ts_data.iloc[split_idx:]
        X_train, y_train = train[feature_cols], train["AmountInUSD"]
        X_test, y_test = test[feature_cols], test["AmountInUSD"]

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
        }

        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            report[name] = self.calculate_metrics(y_test.values, preds)

        # Prophet execution
        m_df = train[["AmountInUSD"]].reset_index().rename(columns={"Date": "ds", "AmountInUSD": "y"})
        p_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        p_model.fit(m_df)
        p_preds = p_model.predict(test.reset_index()[["Date"]].rename(columns={"Date": "ds"}))["yhat"].values
        report["Prophet"] = self.calculate_metrics(y_test.values, p_preds)

        metrics_df = pd.DataFrame(report).T
        metrics_df.to_csv(METRICS_PATH)

        best_model_name = metrics_df["RMSE"].idxmin()
        if best_model_name == "Prophet":
            joblib.dump({"name": "Prophet", "model": p_model, "features": []}, BEST_MODEL_PATH)
        else:
            joblib.dump({"name": best_model_name, "model": models[best_model_name], "features": feature_cols}, BEST_MODEL_PATH)

        # Generate and save feature importance
        imp_model_name = best_model_name if best_model_name in ["RandomForest", "XGBoost"] else "XGBoost"
        try:
            imp_model = models[imp_model_name]
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": imp_model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
        except Exception as e:
            print(f"Error saving feature importance: {e}")

        print(f"Champion Model Selected: {best_model_name}")
        self._plot(y_test, models, p_preds, X_test)

    def _plot(self, y_test, models, p_preds, X_test):
        plt.figure(figsize=(10, 5))
        plt.plot(y_test.index, y_test.values, label="Actual", color="black", marker="o")
        for name, m in models.items():
            plt.plot(y_test.index, m.predict(X_test), label=name, linestyle="--")
        plt.plot(y_test.index, p_preds, label="Prophet", linestyle=":")
        plt.legend(); plt.grid(True)
        plt.savefig(CHARTS_DIR / "actual_vs_predicted.png")
        plt.close()