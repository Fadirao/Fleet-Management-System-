import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def test_linreg_small():
    df = pd.DataFrame({'total_km':[10,20,30,40],'avg_speed':[40,45,50,55],'idle_time_h':[1,1,2,2],'fuel_l':[5,7,9,11]})
    X = df[['total_km','avg_speed','idle_time_h']]; y = df['fuel_l']
    m = LinearRegression().fit(X,y)
    r2 = r2_score(y, m.predict(X))
    assert r2 > 0.8
