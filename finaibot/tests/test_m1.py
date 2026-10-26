import pandas as pd
import pytest
from engine.backtester import run_backtest
from engine.costs import CostModel
from engine.data import DataValidationError, validate_market_data
from engine.risk import RiskConfig
from engine.strategy import Strategy, StrategyValidationError


def bars(closes):
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC"), "open": closes, "high": [x + 1 for x in closes], "low": [x - 1 for x in closes], "close": closes})


def ohlc(exit_bar):
    rows = [{"open": 100, "high": 100, "low": 100, "close": 100} for _ in range(3)]
    rows.append(exit_bar)
    frame = pd.DataFrame(rows)
    frame.insert(0, "timestamp", pd.date_range("2024-01-01", periods=len(frame), freq="h", tz="UTC"))
    return frame


def strategy(**extra):
    doc = {"id": "synthetic", "version": "1.0.0", "pair": "EURUSD", "timeframe": "1h", "created_at": "2024-01-01T00:00:00Z", "entry_rules": [{"type": "always"}], "exit_rules": [], "position_size": 1, "metadata": {"side": "long"}}
    doc.update(extra)
    return Strategy.from_dict(doc)


def test_strategy_requires_core_fields():
    with pytest.raises(StrategyValidationError):
        Strategy.from_dict({"id": "x"})


def test_data_validation_rejects_duplicate_and_bad_ohlc():
    frame = bars([100, 101])
    frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    with pytest.raises(DataValidationError):
        validate_market_data(frame)
    frame = bars([100, 101])
    frame.loc[0, "high"] = 99
    with pytest.raises(DataValidationError):
        validate_market_data(frame)


def test_entry_uses_next_bar_open_and_costs_are_logged():
    result = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000, CostModel(commission=2))
    assert len(result.trades) == 1
    assert result.trades[0].entry_price == 100
    assert result.trades[0].costs == 4
    assert result.trades[0].net_pnl == 1


def test_declarative_bar_exit_is_supported():
    result = run_backtest(strategy(exit_rules=[{"type": "bars", "value": 1}]), bars([100, 100, 100, 105]), 1000)
    assert result.trades[0].exit_reason == "rule_exit"


def test_hard_total_risk_breach_is_recorded():
    result = run_backtest(strategy(), bars([100, 100, 100, 89]), 100, CostModel(), RiskConfig(total_preventive_pct=.01, total_hard_pct=.02))
    assert any(event["kind"] == "total_hard" for event in result.events)


def test_short_stop_loss_is_triggered_at_stop_price():
    result = run_backtest(strategy(metadata={"side": "short"}, stop_loss=5), ohlc({"open": 100, "high": 105, "low": 99, "close": 100}), 1000)
    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].exit_price == 105


@pytest.mark.parametrize(
    ("side", "field", "exit_bar", "expected_price", "expected_pnl"),
    [
        ("long", "stop_loss", {"open": 100, "high": 101, "low": 95, "close": 100}, 95, -5),
        ("short", "stop_loss", {"open": 100, "high": 105, "low": 99, "close": 100}, 105, -5),
        ("long", "take_profit", {"open": 100, "high": 105, "low": 99, "close": 100}, 105, 5),
        ("short", "take_profit", {"open": 100, "high": 101, "low": 95, "close": 100}, 95, 5),
    ],
)
def test_long_and_short_exit_boundaries(side, field, exit_bar, expected_price, expected_pnl):
    result = run_backtest(strategy(metadata={"side": side}, **{field: 5}), ohlc(exit_bar), 1000)
    trade = result.trades[0]
    assert trade.exit_price == expected_price
    assert trade.net_pnl == expected_pnl
    assert trade.exit_reason == field.replace("take_profit", "take_profit")


@pytest.mark.parametrize(
    ("side", "exit_bar", "expected_price"),
    [
        ("long", {"open": 94, "high": 96, "low": 93, "close": 95}, 94),
        ("short", {"open": 106, "high": 107, "low": 104, "close": 105}, 106),
    ],
)
def test_stop_loss_gap_executes_at_candle_open(side, exit_bar, expected_price):
    result = run_backtest(strategy(metadata={"side": side}, stop_loss=5), ohlc(exit_bar), 1000)
    assert result.trades[0].exit_reason == "stop_loss"
    assert result.trades[0].exit_price == expected_price


@pytest.mark.parametrize("side", ["long", "short"])
def test_ambiguous_candle_uses_stop_loss_conservatively(side):
    result = run_backtest(
        strategy(metadata={"side": side}, stop_loss=5, take_profit=5),
        ohlc({"open": 100, "high": 106, "low": 94, "close": 100}),
        1000,
    )
    trade = result.trades[0]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == (95 if side == "long" else 105)
    assert trade.net_pnl == -5

