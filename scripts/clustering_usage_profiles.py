import os, pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans

INP = "data/processed_data/trips_features.csv"
FIG = "visualization/graphs/usage_clusters.png"

if __name__ == "__main__":
    df = pd.read_csv(INP)
    X = df[["total_km","avg_speed","idle_time_h","fuel_l","km_per_litre"]]
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    df["cluster"] = km.labels_
    os.makedirs("visualization/graphs", exist_ok=True)
    plt.figure(figsize=(6,4))
    # simple 2D projection
    plt.scatter(df["total_km"], df["fuel_l"], s=10, c=df["cluster"])
    plt.xlabel("total_km"); plt.ylabel("fuel_l"); plt.title("Usage clusters (2D projection)")
    plt.tight_layout(); plt.savefig(FIG, dpi=150); plt.close()
    df.to_csv("data/processed_data/trips_clustered.csv", index=False)
    print("[OK] clusters saved")
