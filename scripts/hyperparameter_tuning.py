import os, json, pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

TRIPS = "data/processed_data/trips_features.csv"
OUT = "data/processed_data/best_params.json"

if __name__ == "__main__":
    df = pd.read_csv(TRIPS)
    X = df[["total_km","avg_speed","idle_time_h","fuel_l","km_per_litre"]]
    y = df["maint_flag"]
    grid = {"n_estimators":[100,200], "max_depth":[None,10,20], "min_samples_leaf":[1,2]}
    gs = GridSearchCV(RandomForestClassifier(random_state=42), grid, scoring="roc_auc", cv=3, n_jobs=-1)
    gs.fit(X,y)
    best = gs.best_params_
    os.makedirs("data/processed_data", exist_ok=True)
    with open(OUT,"w") as f: json.dump(best, f, indent=2)
    print("[OK] tuning saved", best)
