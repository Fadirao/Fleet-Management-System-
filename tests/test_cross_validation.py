import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

def test_cv_runs():
    df = pd.DataFrame({'a':[1,2,3,4,5],'b':[5,4,3,2,1],'y':[1,2,3,4,5]})
    X = df[['a','b']]; y = df['y']
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    m = RandomForestRegressor(n_estimators=10, random_state=42)
    scores = cross_val_score(m, X, y, scoring='neg_root_mean_squared_error', cv=kf)
    assert len(scores) == 3
