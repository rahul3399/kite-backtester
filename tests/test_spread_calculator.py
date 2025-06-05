import numpy as np
import pytest
from collections import deque
from app.services.spread_calculator import SpreadCalculator


def test_calculate_spread_computes_indicators_correctly():
    calc = SpreadCalculator()
    # simulate 50 ticks with increasing spread value
    last_data = None
    for i in range(1, 51):
        last_data = calc.calculate_spread("A", 100 + i, "B", 100)

    assert last_data.spread == 50
    assert last_data.spread_percentage == pytest.approx(50.0)

    expected_ma20 = np.mean(np.arange(31, 51))
    expected_ma50 = np.mean(np.arange(1, 51))
    expected_z = (50 - np.mean(np.arange(31, 51))) / np.std(np.arange(31, 51))

    assert last_data.ma_20 == pytest.approx(expected_ma20)
    assert last_data.ma_50 == pytest.approx(expected_ma50)
    assert last_data.z_score == pytest.approx(expected_z)


def test__calculate_indicators_returns_expected_values():
    calc = SpreadCalculator()
    pair_key = "X_Y"
    data = list(range(1, 51))
    calc.spread_history[pair_key] = deque(data, maxlen=calc.lookback_period)

    ma20, ma50, z = calc._calculate_indicators(pair_key)

    assert ma20 == pytest.approx(np.mean(data[-20:]))
    assert ma50 == pytest.approx(np.mean(data[-50:]))
    expected_z = (data[-1] - np.mean(data[-20:])) / np.std(data[-20:])
    assert z == pytest.approx(expected_z)


def test__calculate_indicators_insufficient_data():
    calc = SpreadCalculator()
    pair_key = "A_B"
    data = [1, 2, 3]
    calc.spread_history[pair_key] = deque(data, maxlen=calc.lookback_period)

    ma20, ma50, z = calc._calculate_indicators(pair_key)

    assert ma20 is None and ma50 is None and z is None
