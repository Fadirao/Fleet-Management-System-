import pandas as pd
from scripts.route_planning import plan_routes

def test_plan_routes_empty_safe():
    df = pd.DataFrame({'timestamp':[],'vehicle_id':[],'lat':[],'lon':[],'speed_kmh':[],'idle':[],'fuel_lph':[],'harsh_event':[]})
    out = plan_routes(df)
    assert isinstance(out, pd.DataFrame)
