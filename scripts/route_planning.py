# Simple nearest-neighbor route heuristic per day per vehicle
import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

INP = "data/raw_data/telemetry.csv"
OUT = "data/processed_data/routes_planned.csv"
FIG = "visualization/graphs/route_map.png"

def plan_routes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    routes = []
    for vid, day_df in df.groupby(["vehicle_id","date"]):
        d = day_df.sort_values("timestamp")
        pts = d[["lat","lon"]].to_numpy()
        if len(pts) < 2:
            continue
        visited = [0]; cur = 0
        while len(visited) < len(pts):
            rem = [i for i in range(len(pts)) if i not in visited]
            dists = [np.linalg.norm(pts[cur]-pts[j]) for j in rem]
            nxt = rem[int(np.argmin(dists))]
            visited.append(nxt); cur = nxt
        seq = d.iloc[visited].assign(order=range(len(visited)))
        routes.append(seq[["vehicle_id","date","timestamp","lat","lon","order"]])
    if routes:
        routes_df = pd.concat(routes).reset_index(drop=True)
    else:
        routes_df = pd.DataFrame(columns=["vehicle_id","date","timestamp","lat","lon","order"])
    return routes_df

if __name__ == "__main__":
    df = pd.read_csv(INP)
    routes_df = plan_routes(df)
    os.makedirs("data/processed_data", exist_ok=True)
    routes_df.to_csv(OUT, index=False)
    # plot last day aggregate
    os.makedirs("visualization/graphs", exist_ok=True)
    plt.figure(figsize=(6,5))
    for vid, r in routes_df.groupby("vehicle_id"):
        r2 = r.sort_values(["date","order"]).tail(50)  # keep recent
        plt.plot(r2["lon"], r2["lat"])
    plt.title("Planned routes (recent segments)")
    plt.tight_layout(); plt.savefig(FIG, dpi=150); plt.close()
    print("[OK] routes planned")
