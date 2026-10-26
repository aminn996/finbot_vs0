import math

import pandas as pd
import pytest

from engine.backtester import run_backtest
from engine.costs import CostModel
from engine.risk import MarginConfig
from engine.strategy import Strategy


def market(closes):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC"),
        "open": closes, "high": [max(value, closes[i - 1] if i else value) + 1 for i, value in enumerate(closes)],
        "low": [min(value, closes[i - 1] if i else value) - 1 for i, value in enumerate(closes)], "close": closes,
    })


def strategy(side="long", quantity=1, take_profit=None):
    document = {
        "id": "margin", "version": "1.0.0", "pair": "EURUSD", "timeframe": "1h",
        "created_at": "2024-01-01T00:00:00Z", "entry_rules": [{"type": "always"}],
        "exit_rules": [], "position_size": quantity, "metadata": {"side": side},
    }
    if take_profit is not None:
        document["take_profit"] = take_profit
    return Strategy.from_dict(document)


def test_trade_fits_margin_and_tracks_state():
    result = run_backtest(strategy(), market([100, 100, 100, 105]), 100, margin=MarginConfig(leverage=2))
    state = result.equity_curve.iloc[2]
    assert len(result.trades) == 1
    assert state.exposure == 100
    assert state.used_margin == 50
    assert state.free_margin == 50
    assert state.margin_level == 2


def test_insufficient_margin_rejects_entry():
    result = run_backtest(strategy(), market([100, 100, 100, 105]), 50, margin=MarginConfig())
    assert not result.trades
    assert any(event["kind"] == "insufficient_margin" for event in result.events)


def test_leverage_changes_required_margin():
    one_to_one = run_backtest(strategy(), market([100, 100, 100, 105]), 200, margin=MarginConfig(leverage=1))
    four_to_one = run_backtest(strategy(), market([100, 100, 100, 105]), 200, margin=MarginConfig(leverage=4))
    assert one_to_one.equity_curve.iloc[2].used_margin == 100
    assert four_to_one.equity_curve.iloc[2].used_margin == 25


@pytest.mark.parametrize("side", ["long", "short"])
def test_exposure_limit_rejects_long_and_short(side):
    result = run_backtest(strategy(side), market([100, 100, 100, 105]), 1000, margin=MarginConfig(max_exposure=99))
    assert not result.trades
    assert any(event["kind"] == "exposure_limit" for event in result.events)


def test_exact_margin_and_exposure_boundaries_are_allowed():
    result = run_backtest(strategy(), market([100, 100, 100, 105]), 100, margin=MarginConfig(max_exposure=100))
    assert len(result.trades) == 1
    assert result.equity_curve.iloc[2].free_margin == 0


def test_unrealized_equity_changes_free_margin_and_costs_are_included():
    data = market([100, 100, 100, 105, 105])
    costs = CostModel(spread=2, commission=2, slippage=1)
    result = run_backtest(strategy(), data, 200, costs, margin=MarginConfig(leverage=2))
    entry_state = result.equity_curve.iloc[2]
    moved_state = result.equity_curve.iloc[3]
    assert entry_state.equity == 192
    assert moved_state.equity == 197
    assert moved_state.free_margin == 147


def test_closing_position_releases_used_margin():
    result = run_backtest(strategy(take_profit=5), market([100, 100, 100, 105]), 100, margin=MarginConfig())
    state = result.equity_curve.iloc[-1]
    assert result.trades[0].exit_reason == "take_profit"
    assert state.used_margin == 0
    assert state.exposure == 0
    assert state.free_margin == state.equity


@pytest.mark.parametrize("field,value", [
    ("leverage", 0), ("leverage", -1), ("leverage", math.nan), ("leverage", math.inf),
    ("max_exposure", -1), ("max_exposure", math.nan), ("max_exposure", math.inf), ("max_exposure", "invalid"),
])
def test_invalid_margin_configuration_is_rejected(field, value):
    with pytest.raises(ValueError):
        MarginConfig(**{field: value})


@pytest.mark.parametrize("capital", [0, -1, math.nan, math.inf, "invalid"])
def test_invalid_initial_capital_is_rejected(capital):
    with pytest.raises(ValueError):
        run_backtest(strategy(), market([100, 100, 100, 105]), capital)
