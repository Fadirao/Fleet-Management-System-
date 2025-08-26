import pandas as pd
from sklearn.cluster import KMeans

def test_dummy_kmeans():
    df = pd.DataFrame({'total_km':[1,2,3,10,11,12],'avg_speed':[30,31,29,60,62,58],'idle_time_h':[0,1,0,0,1,1],'fuel_l':[5,6,5,7,8,7],'km_per_litre':[10,9,11,8,7,8]})
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(df[['total_km','fuel_l']])
    assert hasattr(km, 'cluster_centers_')
