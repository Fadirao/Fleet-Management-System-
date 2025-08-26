import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

INP = "data/processed_data/trips_features.csv"
FIG = "visualization/graphs/fuel_fit.png"
OUT = "data/processed_data/fuel_pred.csv"

if __name__ == "__main__":
    df = pd.read_csv(INP)
    X = df[["total_km","avg_speed","idle_time_h"]]
    y = df["fuel_l"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(Xtr,ytr)
    yp = model.predict(Xte)
    rmse = mean_squared_error(yte, yp, squared=False); r2 = r2_score(yte, yp)
    os.makedirs("visualization/graphs", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.scatter(yte, yp, s=10)
    plt.xlabel("Actual fuel_l"); plt.ylabel("Predicted fuel_l")
    plt.title(f"Fuel model (RMSE={rmse:.2f}, R2={r2:.2f})")
    plt.tight_layout(); plt.savefig(FIG, dpi=150); plt.close()
    pd.DataFrame({"y_true":yte,"y_pred":yp}).to_csv(OUT, index=False)
    print("[OK] fuel model metrics:", rmse, r2)
