import os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

INP = "data/processed_data/trips_features.csv"
FIG = "visualization/graphs/maintenance_roc.png"
OUT = "data/processed_data/maintenance_scores.csv"

if __name__ == "__main__":
    df = pd.read_csv(INP)
    X = df[["total_km","avg_speed","idle_time_h","fuel_l","km_per_litre"]]
    y = df["maint_flag"] if "maint_flag" in df.columns else (df["avg_speed"]<0).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr,ytr)
    proba = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba)
    fpr, tpr, _ = roc_curve(yte, proba)
    os.makedirs("visualization/graphs", exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.title(f"Maintenance risk ROC (AUC={auc:.3f})"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout(); plt.savefig(FIG, dpi=150); plt.close()
    out = pd.DataFrame({"y_true":yte, "score":proba})
    out.to_csv(OUT, index=False)
    print("[OK] maintenance model AUC=", auc)
