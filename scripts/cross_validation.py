import os, pandas as pd, numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

TRIPS = "data/processed_data/trips_features.csv"
OUT = "data/processed_data/model_cv.csv"

if __name__ == "__main__":
    df = pd.read_csv(TRIPS)
    Xc = df[["total_km","avg_speed","idle_time_h","fuel_l","km_per_litre"]]
    yc = df["maint_flag"]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    aucs = cross_val_score(clf, Xc, yc, scoring="roc_auc", cv=kf)
    # regressor for fuel_l
    Xr = df[["total_km","avg_speed","idle_time_h"]]; yr = df["fuel_l"]
    rfr = RandomForestRegressor(n_estimators=200, random_state=42)
    rmes = -cross_val_score(rfr, Xr, yr, scoring="neg_root_mean_squared_error", cv=kf)
    out = pd.DataFrame({"metric":["AUC"]*len(aucs)+["RMSE"]*len(rmes),
                        "fold":list(range(1,len(aucs)+1))+list(range(1,len(rmes)+1)),
                        "score":np.concatenate([aucs, rmes])})
    os.makedirs("data/processed_data", exist_ok=True)
    out.to_csv(OUT, index=False)
    print("[OK] CV saved")
