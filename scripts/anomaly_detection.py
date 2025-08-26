import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

INP = "data/processed_data/trips_features.csv"
OUT = "data/processed_data/anomalies.csv"
FIG = "visualization/graphs/anomalies.png"

if __name__ == "__main__":
    df = pd.read_csv(INP)
    X = df[["total_km","avg_speed","idle_time_h","fuel_l","km_per_litre"]]
    iso = IsolationForest(contamination=0.05, random_state=42).fit(X)
    df["anomaly_score"] = iso.decision_function(X)
    df["is_anomaly"] = (iso.predict(X) == -1).astype(int)
    df.to_csv(OUT, index=False)
    os.makedirs("visualization/graphs", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.scatter(df["total_km"], df["fuel_l"], s=10, c=df["is_anomaly"])
    plt.title("Anomaly detection (red=anomaly)")
    plt.tight_layout(); plt.savefig(FIG, dpi=150); plt.close()
    print("[OK] anomalies saved")
