import os, pandas as pd, numpy as np
from scripts.data_generation import simulate_data

def test_generate():
    simulate_data(n_vehicles=5, hours=24)
    assert os.path.exists('data/raw_data/telemetry.csv')
    assert os.path.exists('data/raw_data/trips.csv')
