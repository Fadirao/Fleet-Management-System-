import os, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

CLEAN_TELE = "data/processed_data/telemetry_clean.csv"
CLEAN_TRIPS = "data/processed_data/trips_clean.csv"
FE_TELE = "data/processed_data/telemetry_features.csv"
FE_TRIPS = "data/processed_data/trips_features.csv"

def engineer_tele(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour"].isin([0,1,2,3,4,5,22,23]).astype(int)
    # rolling stats by vehicle
    df = df.sort_values(["vehicle_id","timestamp"])
    df["speed_ma"] = df.groupby("vehicle_id")["speed_kmh"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    return df

def engineer_trips(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # efficiency proxy: km per litre
    df["km_per_litre"] = (df["total_km"] / (df["fuel_l"] + 1e-6)).clip(0, 50)
    scaler = StandardScaler()
    df[["total_km","avg_speed","idle_time_h","fuel_l"]] = scaler.fit_transform(df[["total_km","avg_speed","idle_time_h","fuel_l"]])
    return df

if __name__ == "__main__":
    tele = pd.read_csv(CLEAN_TELE); trips = pd.read_csv(CLEAN_TRIPS)
    ftele = engineer_tele(tele); ftrips = engineer_trips(trips)
    os.makedirs("data/processed_data", exist_ok=True)
    ftele.to_csv(FE_TELE, index=False)
    ftrips.to_csv(FE_TRIPS, index=False)
    print("[OK] features saved")
