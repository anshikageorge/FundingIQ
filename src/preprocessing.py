#writefile startup_funding_forecaster/src/preprocessing.py
import pandas as pd
import numpy as np
import re

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path

    @staticmethod
    def _clean_amount(amount):
        if pd.isna(amount): return np.nan
        amount_str = str(amount).lower().strip()
        if any(x in amount_str for x in ["undisclosed", "unknown", "nan"]): return np.nan
        amount_str = re.sub(r"[,\$\+ ]", "", amount_str)
        try: return float(amount_str)
        except ValueError: return np.nan

    @staticmethod
    def _clean_date(date_str):
        if pd.isna(date_str): return pd.NaT
        date_str = str(date_str).strip().replace("//", "/")
        match = re.search(r"(\d{1,2})[/\.-](\d{1,2})[/\.-](\d{4})", date_str)
        if match:
            day, month, year = match.groups()
            return pd.to_datetime(f"{year}-{month}-{day}", errors="coerce")
        return pd.to_datetime(date_str, errors="coerce")

    def load_and_clean(self):
        df = pd.read_csv(self.file_path)
        df.columns = [col.replace(" ", "") for col in df.columns]

        # Mapping variations
        df.rename(columns={"AmountinUSD": "AmountInUSD", "Datedd/mm/yyyy": "Date"}, inplace=True, errors="ignore")

        df["AmountInUSD"] = df["AmountInUSD"].apply(self._clean_amount)
        df["Date"] = df["Date"].apply(self._clean_date)
        df = df.dropna(subset=["Date"]).copy()

        df["IndustryVertical"] = df["IndustryVertical"].fillna("Unknown").str.title().str.strip()
        df["CityLocation"] = df["CityLocation"].fillna("Unknown").str.title().str.strip().apply(lambda x: str(x).split("/")[0].strip())

        median_funding = df["AmountInUSD"].median()
        df["AmountInUSD"] = df["AmountInUSD"].fillna(median_funding if not pd.isna(median_funding) else 100000)

        return df.sort_values(by="Date").reset_index(drop=True)