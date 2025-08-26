import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def test_auc_pipeline():
    df = pd.DataFrame({'total_km':[1,2,3,4,5,6,7,8],
                       'avg_speed':[40,42,39,41,20,18,22,21],
                       'idle_time_h':[0,1,0,1,2,3,2,3],
                       'fuel_l':[5,5,5,5,9,9,8,8],
                       'km_per_litre':[10,10,10,10,5,5,6,6],
                       'maint_flag':[0,0,0,0,1,1,1,1]})
    X = df[['total_km','avg_speed','idle_time_h','fuel_l','km_per_litre']]
    y = df['maint_flag']
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(Xtr,ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:,1])
    assert auc >= 0.5
