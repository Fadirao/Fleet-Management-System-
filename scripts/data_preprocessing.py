import os, pandas as pd, numpy as np

RAW_TELE = "data/raw_data/telemetry.csv"
RAW_TRIPS = "data/raw_data/trips.csv"
CLEAN_TELE = "data/processed_data/telemetry_clean.csv"
CLEAN_TRIPS = "data/processed_data/trips_clean.csv"

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.select_dtypes(include=[float,int]).columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if df[c].dtype.kind in "biufc":
            df[c] = df[c].fillna(df[c].median())
    return df

if __name__ == "__main__":
    tele = pd.read_csv(RAW_TELE)
    trips = pd.read_csv(RAW_TRIPS)
    tele_c = clean(tele); trips_c = clean(trips)
    os.makedirs("data/processed_data", exist_ok=True)
    tele_c.to_csv(CLEAN_TELE, index=False)
    trips_c.to_csv(CLEAN_TRIPS, index=False)
    print("[OK] cleaned")
