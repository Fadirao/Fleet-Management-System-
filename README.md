# Fleet Management System

End‑to‑end pipeline for fleet analytics: telemetry simulation, preprocessing, feature engineering,
usage clustering, route planning heuristic, anomaly detection, predictive maintenance, fuel efficiency modeling,
cross‑validation, tuning, and ready‑made visuals.

## Expected raw files
After running the provided generator (already executed for you), the repo contains:
- `data/raw_data/telemetry.csv` (GPS pings & vehicle status)
- `data/raw_data/trips.csv` (trip start/stop summaries)

## Quickstart (optional)
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt

# Regenerate outputs if desired
python scripts/data_generation.py
python scripts/data_preprocessing.py
python scripts/feature_engineering.py
python scripts/clustering_usage_profiles.py
python scripts/route_planning.py
python scripts/anomaly_detection.py
python scripts/maintenance_model.py
python scripts/fuel_efficiency_model.py
python scripts/cross_validation.py
python scripts/hyperparameter_tuning.py
python scripts/report_generator.py
```

## Tests
```bash
pytest -q
```

## Outputs (already generated under `visualization/graphs/`)
- `usage_clusters.png`, `mileage_distribution.png`, `route_map.png`, `anomalies.png`,
- `maintenance_roc.png`, `fuel_fit.png`, `model_cv.png`.
