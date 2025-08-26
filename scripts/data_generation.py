import os, random
from datetime import datetime, timedelta
import numpy as np, pandas as pd

RAW_TELE = "data/raw_data/telemetry.csv"
RAW_TRIPS = "data/raw_data/trips.csv"

def simulate_data(n_vehicles=50, hours=240):  # 10 days at 1‑hour pings
    random.seed(42); np.random.seed(42)
    start = datetime(2024,1,1,8,0,0)
    rows_t = []
    for vid in range(1, n_vehicles+1):
        lat, lon = 51.5 + np.random.normal(0, 0.05), -0.1 + np.random.normal(0, 0.05)
        fuel_eff = np.random.normal(12.0, 2.0)  # km/L
        for h in range(hours):
            ts = start + timedelta(hours=h)
            speed = max(0, np.random.normal(40, 15))  # km/h
            # idle probability higher at night
            idle = 1 if (ts.hour in [0,1,2,3,4,23] and np.random.rand()<0.4) else (1 if np.random.rand()<0.15 else 0)
            # move position
            if idle==0:
                bearing = np.random.uniform(0, 2*np.pi)
                dist_km = speed / 60.0  # 1 hour ~ distance scaled for realism
                lat += (dist_km/111) * np.cos(bearing)
                lon += (dist_km/85) * np.sin(bearing)
            fuel_rate = (speed/ fuel_eff) * (1.25 if idle else 1.0)
            harsh = 1 if np.random.rand()<0.05 else 0
            rows_t.append([ts.strftime("%Y-%m-%d %H:%M:%S"), vid, lat, lon, speed, idle, fuel_rate, harsh])
    tele = pd.DataFrame(rows_t, columns=["timestamp","vehicle_id","lat","lon","speed_kmh","idle","fuel_lph","harsh_event"])
    # trips summary per vehicle/day
    tele["date"] = pd.to_datetime(tele["timestamp"]).dt.date
    grp = tele.groupby(["vehicle_id","date"])
    trips = grp.agg(total_km=("speed_kmh", lambda s: s.sum()/60.0),
                    avg_speed=("speed_kmh","mean"),
                    idle_time_h=("idle","sum"),
                    fuel_l=("fuel_lph","sum")).reset_index()
    # add label for maintenance need (proxy from high fuel and harsh)
    trips["maint_flag"] = ((trips["fuel_l"]>trips["fuel_l"].median()*1.3) | (trips["avg_speed"]<30)).astype(int)
    os.makedirs("data/raw_data", exist_ok=True)
    tele.to_csv(RAW_TELE, index=False)
    trips.to_csv(RAW_TRIPS, index=False)
    print("[OK] generated", RAW_TELE, RAW_TRIPS)

if __name__ == "__main__":
    simulate_data()
