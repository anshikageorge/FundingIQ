#writefile startup_funding_forecaster/src/feature_engineering.py
import pandas as pd
import numpy as np

class TimeSeriesFeatureEngineer:
    def __init__(self, agg_level="ME"):
        self.agg_level = agg_level

    def transform(self, df):
        ts_df = df.set_index("Date").resample(self.agg_level)["AmountInUSD"].sum().to_frame()
        ts_df["Year"] = ts_df.index.year
        ts_df["Month"] = ts_df.index.month
        ts_df["Quarter"] = ts_df.index.quarter

        ts_df["Lag_1"] = ts_df["AmountInUSD"].shift(1)
        ts_df["Lag_3"] = ts_df["AmountInUSD"].shift(3)
        ts_df["Lag_6"] = ts_df["AmountInUSD"].shift(6)

        ts_df["Rolling_Mean_3"] = ts_df["AmountInUSD"].shift(1).rolling(3).mean()
        ts_df["Rolling_Mean_6"] = ts_df["AmountInUSD"].shift(1).rolling(6).mean()
        ts_df["Rolling_Std_3"] = ts_df["AmountInUSD"].shift(1).rolling(3).std()
        ts_df["Growth_Rate"] = ts_df["Lag_1"].pct_change(1).replace([np.inf, -np.inf], 0)

        return ts_df.bfill().ffill()