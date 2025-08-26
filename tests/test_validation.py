import pandas as pd
from scripts.validation import validate

def test_validate():
    df = pd.DataFrame({'total_km':[1.0],'avg_speed':[40.0],'idle_time_h':[0.0],'fuel_l':[5.0],'km_per_litre':[10.0]})
    assert validate(df) is True
