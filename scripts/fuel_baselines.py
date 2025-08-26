# Utility module (used by tests to sanity-check formulae)
def expected_fuel_l(km, eff_km_per_l):
    return km / max(eff_km_per_l, 1e-6)
