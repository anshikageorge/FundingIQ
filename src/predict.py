#writefile startup_funding_forecaster/src/predict.py
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from startup_funding_forecaster.src.config import BEST_MODEL_PATH, FORECAST_PATH, CHARTS_DIR

class ForecastEngine:
    def __init__(self):
        self.artifact = joblib.load(BEST_MODEL_PATH)

    def forecast_horizon(self, base_data, months=12):
        last_date = base_data.index.max()
        future_dates = pd.date_range(start=last_date + pd.offsets.MonthEnd(1), periods=months, freq="ME")

        if self.artifact["name"] == "Prophet":
            future_df = pd.DataFrame({"ds": future_dates})
            prophet_res = self.artifact["model"].predict(future_df)
            preds = prophet_res["yhat"].clip(lower=0).values
            preds_lower = prophet_res["yhat_lower"].clip(lower=0).values
            preds_upper = prophet_res["yhat_upper"].clip(lower=0).values
        else:
            working = base_data.copy()
            preds = []
            for d in future_dates:
                row = {
                    "Year": d.year, "Month": d.month, "Quarter": d.quarter,
                    "Lag_1": working.iloc[-1]["AmountInUSD"],
                    "Lag_3": working.iloc[-3]["AmountInUSD"],
                    "Lag_6": working.iloc[-6]["AmountInUSD"],
                    "Rolling_Mean_3": working.iloc[-3:]["AmountInUSD"].mean(),
                    "Rolling_Mean_6": working.iloc[-6:]["AmountInUSD"].mean(),
                    "Rolling_Std_3": working.iloc[-3:]["AmountInUSD"].std(),
                    "Growth_Rate": (working.iloc[-1]["AmountInUSD"] - working.iloc[-2]["AmountInUSD"]) / (working.iloc[-2]["AmountInUSD"] + 1e-5)
                }
                x = pd.DataFrame([row])[self.artifact["features"]]
                p_val = max(0.0, float(self.artifact["model"].predict(x)[0]))
                preds.append(p_val)
                working = pd.concat([working, pd.DataFrame(index=[d], data={"AmountInUSD": p_val, **row})])
            
            hist_std = base_data["AmountInUSD"].iloc[-12:].std()
            if pd.isna(hist_std) or hist_std == 0:
                hist_std = base_data["AmountInUSD"].std()
            if pd.isna(hist_std) or hist_std == 0:
                hist_std = 1e7
            
            preds_lower = [max(0.0, p - 1.96 * hist_std * np.sqrt(i + 1) / 3.0) for i, p in enumerate(preds)]
            preds_upper = [p + 1.96 * hist_std * np.sqrt(i + 1) / 3.0 for i, p in enumerate(preds)]

        f_df = pd.DataFrame({
            "Date": future_dates.strftime("%Y-%m-%d"), 
            "Predicted_Funding_Amount": preds,
            "Predicted_Funding_Amount_Lower": preds_lower,
            "Predicted_Funding_Amount_Upper": preds_upper
        })
        f_df.to_csv(FORECAST_PATH, index=False)

        # Save structural visualization
        plt.figure(figsize=(10, 4))
        plt.plot(base_data.index[-12:], base_data["AmountInUSD"].iloc[-12:], label="Historical")
        plt.plot(future_dates, preds, label="Forecast", marker="s", color="red")
        plt.legend(); plt.grid(True)
        plt.savefig(CHARTS_DIR / "forecast_curve.png")
        plt.close()
        return f_df