from scripts.fuel_baselines import expected_fuel_l
from scripts.mileage_utils import km_from_speed_hours

def test_utils():
    assert abs(expected_fuel_l(100, 10) - 10) < 1e-6
    assert abs(km_from_speed_hours([30, 30, 30, 30, 30, 30]) - 3.0) < 1e-6
