import json

import pandas as pd
import pytest

from engine.backtester import run_backtest
from engine.costs import CostModel
from engine.data import DataValidationError, validate_market_data
from engine.persistence import BacktestInterrupted, RunPersistenceConfig, RunPersistenceError
from engine.strategy import Strategy


def strategy(side="long"):
    return Strategy.from_dict({
        "id": "variable-spread",
        "version": "1.0.0",
        "pair": "EURUSD",
        "timeframe": "1h",
        "created_at": "2024-01-01T00:00:00Z",
        "entry_rules": [{"type": "always"}],
        "exit_rules": [],
        "position_size": 1,
        "metadata": {"side": side},
    })


def midpoint_bars(closes):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC"),
        "open": closes,
        "high": [value + 1 for value in closes],
        "low": [value - 1 for value in closes],
        "close": closes,
    })


def bid_ask_bars(side, closes):
    midpoint = midpoint_bars(closes)
    for field in ("open", "high", "low", "close"):
        width = 2.0
        midpoint[f"bid_{field}"] = midpoint[field] - width / 2
        midpoint[f"ask_{field}"] = midpoint[field] + width / 2
    return midpoint


def test_per_bar_spread_changes_directional_execution_and_is_not_double_counted():
    data = midpoint_bars([100, 100, 100, 105])
    data["spread"] = [2.0, 2.0, 2.0, 4.0]

    result = run_backtest(strategy(), data, 1000, CostModel(spread=99))
    trade = result.trades[0]

    assert trade.entry_price == 101
    assert trade.exit_price == 103
    assert trade.cost_breakdown.spread == 3
    assert trade.gross_pnl == 5
    assert trade.net_pnl == 2
    assert result.equity_curve.iloc[-1].equity == 1002


@pytest.mark.parametrize("side,expected_entry,expected_exit", [
    ("long", 101.0, 104.0),
    ("short", 99.0, 96.0),
])
def test_bid_ask_ohlc_uses_directional_prices_without_scalar_spread(side, expected_entry, expected_exit):
    closes = [100, 100, 100, 105 if side == "long" else 95]
    data = bid_ask_bars(side, closes)

    result = run_backtest(strategy(side), data, 1000, CostModel(spread=50))
    trade = result.trades[0]

    assert trade.entry_price == expected_entry
    assert trade.exit_price == expected_exit
    assert trade.cost_breakdown.spread == 2
    assert trade.gross_pnl == 5
    assert trade.net_pnl == 3
    assert result.equity_curve.iloc[-1].equity == 1003


def test_zero_bid_ask_spread_matches_midpoint_execution():
    data = midpoint_bars([100, 100, 100, 105])
    for field in ("open", "high", "low", "close"):
        data[f"bid_{field}"] = data[field]
        data[f"ask_{field}"] = data[field]

    result = run_backtest(strategy(), data, 1000)

    assert result.trades[0].entry_price == 100
    assert result.trades[0].exit_price == 105
    assert result.trades[0].cost_breakdown.spread == 0
    assert result.trades[0].net_pnl == 5


@pytest.mark.parametrize("mutator,code", [
    (lambda frame: frame.assign(spread=[0, float("nan"), 1, 1]), {"non_finite_spread", "missing_spread"}),
    (lambda frame: frame.assign(spread=[0, float("inf"), 1, 1]), {"non_finite_spread"}),
    (lambda frame: frame.assign(spread=[0, -1, 1, 1]), {"negative_spread"}),
    (lambda frame: frame.assign(spread=[0, "bad", 1, 1]), {"invalid_numeric_spread"}),
    (lambda frame: frame.assign(bid_open=[99, 99, 99, 99]), {"incomplete_bid_ask"}),
])
def test_invalid_variable_spread_data_is_rejected_before_simulation(mutator, code):
    data = mutator(midpoint_bars([100, 100, 100, 105]))

    with pytest.raises(DataValidationError) as caught:
        validate_market_data(data, timeframe="1h", minimum_rows=3)

    assert {issue.code for issue in caught.value.report.issues} & code


def test_inconsistent_bid_ask_data_is_rejected():
    data = bid_ask_bars("long", [100, 100, 100, 105])
    data.loc[1, "bid_close"] = data.loc[1, "ask_close"] + 1

    with pytest.raises(DataValidationError, match="bid_close"):
        validate_market_data(data, timeframe="1h", minimum_rows=3)


def test_execution_fields_are_normalized_without_mutating_input_and_change_dataset_hash():
    data = midpoint_bars([100, 100, 100, 105])
    data["spread"] = ["1", "2", "2", "3"]
    original = data.copy(deep=True)

    result = run_backtest(strategy(), data, 1000)
    changed = data.copy(deep=True)
    changed.loc[1, "spread"] = "2.5"
    changed_result = run_backtest(strategy(), changed, 1000)

    pd.testing.assert_frame_equal(data, original)
    assert result.metadata["dataset_hash"] != changed_result.metadata["dataset_hash"]
    json.dumps(result.metadata)


def test_identical_variable_spread_inputs_are_deterministic():
    data = midpoint_bars([100, 100, 100, 105])
    data["spread"] = [1.0, 1.0, 1.5, 2.0]

    first = run_backtest(strategy(), data, 1000)
    second = run_backtest(strategy(), data, 1000)

    assert first.metadata["input_hash"] == second.metadata["input_hash"]
    assert first.trades == second.trades
    pd.testing.assert_frame_equal(first.equity_curve, second.equity_curve)


def test_checkpoint_resume_rejects_changed_spread_data(tmp_path):
    data = midpoint_bars([100, 100, 100, 105, 105])
    data["spread"] = [1.0] * len(data)
    config = RunPersistenceConfig(tmp_path, interrupt_after_bars=2)

    with pytest.raises(BacktestInterrupted) as caught:
        run_backtest(strategy(), data, 1000, persistence=config)

    changed = data.copy(deep=True)
    changed.loc[3, "spread"] = 2.0
    with pytest.raises(RunPersistenceError, match="incompatible"):
        run_backtest(
            strategy(), changed, 1000,
            persistence=RunPersistenceConfig(tmp_path),
            resume_run_id=caught.value.run_id,
        )


def test_month_boundary_marks_open_position_and_does_not_double_charge_costs():
    data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-31T21:00:00Z",
            "2024-01-31T22:00:00Z",
            "2024-01-31T23:00:00Z",
            "2024-02-01T00:00:00Z",
            "2024-02-01T01:00:00Z",
        ], utc=True),
        "open": [100, 100, 100, 101, 102],
        "high": [101, 101, 102, 103, 103],
        "low": [99, 99, 99, 100, 101],
        "close": [100, 100, 101, 102, 102],
    })
    costs = CostModel(spread=2, commission=2)

    result = run_backtest(strategy(), data, 1000, costs)
    month_end = result.equity_curve.iloc[2]
    final = result.equity_curve.iloc[-1]

    assert month_end.timestamp == pd.Timestamp("2024-01-31T23:00:00Z")
    assert month_end.equity == 995
    assert month_end.total_costs == 6
    assert final.equity == 996
    assert result.trades[0].costs == 6
    assert final.total_costs == result.trades[0].costs
