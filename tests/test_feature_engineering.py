import pandas as pd
from scripts.feature_engineering import engineer_tele, engineer_trips

def test_engineer_columns():
    tele = pd.DataFrame({'timestamp':['2024-01-01 08:00:00'],'vehicle_id':[1],'speed_kmh':[30],'idle':[0]})
    trips = pd.DataFrame({'total_km':[100.0],'avg_speed':[40.0],'idle_time_h':[2.0],'fuel_l':[8.0]})
    e1 = engineer_tele(tele)
    e2 = engineer_trips(trips)
    for col in ['hour','is_night','speed_ma']:
        assert col in e1.columns
    assert 'km_per_litre' in e2.columns
