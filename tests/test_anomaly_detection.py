import pandas as pd
from sklearn.ensemble import IsolationForest

def test_iso():
    df = pd.DataFrame({'total_km':[1,2,100,101],'avg_speed':[30,31,5,6],'idle_time_h':[0,0,5,6],'fuel_l':[5,6,20,22],'km_per_litre':[10,11,2,2]})
    X = df[['total_km','avg_speed','idle_time_h','fuel_l','km_per_litre']]
    iso = IsolationForest(contamination=0.25, random_state=42).fit(X)
    pred = iso.predict(X)
    assert (pred==-1).sum() >= 1
