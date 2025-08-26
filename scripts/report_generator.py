import os, json, pandas as pd, numpy as np, matplotlib.pyplot as plt

TRIPS = "data/processed_data/trips_features.csv"
CV = "data/processed_data/model_cv.csv"
BEST = "data/processed_data/best_params.json"
OUT = "data/processed_data/summary_report.md"
FIG = "visualization/graphs/mileage_distribution.png"

if __name__ == "__main__":
    df = pd.read_csv(TRIPS)
    cv = pd.read_csv(CV)
    best = {}
    if os.path.exists(BEST):
        with open(BEST) as f: best = json.load(f)
    # histogram
    os.makedirs("visualization/graphs", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.hist(df["total_km"], bins=30)
    plt.title("Daily total km distribution")
    plt.tight_layout(); plt.savefig(FIG, dpi=150); plt.close()
    md = ["# Fleet Report",
          f"- Rows: {len(df)}",
          f"- Mean km_per_litre: {df['km_per_litre'].mean():.2f}",
          f"- CV metrics: {cv['metric'].unique().tolist()}",
          f"- Best params (if tuned): {best}"]
    os.makedirs("data/processed_data", exist_ok=True)
    with open(OUT,"w") as f: f.write("\n".join(md))
    print("[OK] report saved")
