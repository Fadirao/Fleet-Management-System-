import pandas as pd

TRIPS = "data/processed_data/trips_features.csv"

def validate(df: pd.DataFrame) -> bool:
    must = {"total_km","avg_speed","idle_time_h","fuel_l","km_per_litre"}
    return must.issubset(set(df.columns)) and (df[["total_km","avg_speed","fuel_l"]] .isna().sum().sum()==0)

if __name__ == "__main__":
    df = pd.read_csv(TRIPS)
    print("[OK] validation passed" if validate(df) else "[FAIL] validation failed")
