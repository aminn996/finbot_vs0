import math

import pandas as pd
import pytest

from engine.backtester import run_backtest
from engine.costs import CostModel
from engine.strategy import Strategy


def bars(closes):
    return pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC"), "open": closes, "high": [x + 1 for x in closes], "low": [x - 1 for x in closes], "close": closes})


def open_close_move(side):
    closes = [100, 100, 100, 105, 105] if side == "long" else [100, 100, 100, 95, 95]
    opens = [100, 100, 100, 100, closes[-1]]
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC"),
        "open": opens, "high": [max(o, c) + 1 for o, c in zip(opens, closes)],
        "low": [min(o, c) - 1 for o, c in zip(opens, closes)], "close": closes,
    })


def strategy(side="long"):
    return Strategy.from_dict({
        "id": "costs", "version": "1.0.0", "pair": "EURUSD", "timeframe": "1h",
        "created_at": "2024-01-01T00:00:00Z", "entry_rules": [{"type": "always"},],
        "exit_rules": [], "position_size": 1, "metadata": {"side": side},
    })


def test_zero_cost_baseline_and_reconciliation():
    trade = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000).trades[0]
    assert trade.entry_price == 100
    assert trade.exit_price == 105
    assert trade.gross_pnl == 5
    assert trade.costs == 0
    assert trade.net_pnl == 5
    assert trade.gross_pnl - trade.costs == trade.net_pnl


def test_spread_is_charged_on_both_sides_and_changes_execution_prices():
    trade = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000, CostModel(spread=2)).trades[0]
    assert trade.entry_price == 101
    assert trade.exit_price == 104
    assert trade.cost_breakdown.spread == 2
    assert trade.net_pnl == 3


@pytest.mark.parametrize("side,entry,exit,expected_net", [("long", 101, 104, 3), ("short", 99, 96, 3)])
def test_slippage_is_adverse_for_long_and_short(side, entry, exit, expected_net):
    closes = [100, 100, 100, 105] if side == "long" else [100, 100, 100, 95]
    trade = run_backtest(strategy(side), bars(closes), 1000, CostModel(slippage=1)).trades[0]
    assert trade.entry_price == entry
    assert trade.exit_price == exit
    assert trade.cost_breakdown.slippage == 2
    assert trade.net_pnl == expected_net


def test_commission_is_applied_once_per_entry_and_exit_fill():
    result = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000, CostModel(commission=2))
    trade = result.trades[0]
    assert trade.cost_breakdown.commission == 4
    assert trade.costs == 4
    assert trade.net_pnl == trade.gross_pnl - trade.costs
    assert result.equity_curve.iloc[-1].equity == 1001


@pytest.mark.parametrize("side,closes", [("long", [100, 100, 100, 105]), ("short", [100, 100, 100, 95])])
def test_open_equity_and_forced_close_charge_costs_once(side, closes):
    costs = CostModel(spread=2, commission=2, slippage=1, swap=3, conversion=4)
    result = run_backtest(strategy(side), bars(closes), 1000, costs)
    trade = result.trades[0]
    assert trade.gross_pnl == 5
    assert trade.costs == 15
    assert trade.net_pnl == -10
    assert result.equity_curve.iloc[2].equity == 985
    assert result.equity_curve.iloc[-1].equity == 990


@pytest.mark.parametrize("side", ["long", "short"])
def test_open_equity_marks_completed_candle_close(side):
    costs = CostModel(spread=2, commission=2, slippage=1, swap=3, conversion=4)
    result = run_backtest(strategy(side), open_close_move(side), 1000, costs)
    # The position is entered at 100; index 3 then closes at 105 (long) or
    # 95 (short), so the completed-candle close must drive valuation.
    assert result.equity_curve.iloc[3].equity == 990
    assert result.trades[0].net_pnl == -10
    assert result.equity_curve.iloc[-1].equity == 990


@pytest.mark.parametrize(
    "side,field,expected_price,expected_net",
    [("long", "stop_loss", 93, -9), ("short", "take_profit", 97, 1)],
)
def test_protective_exit_uses_directional_costs(side, field, expected_price, expected_net):
    configured = Strategy.from_dict({
        "id": "cost-exit", "version": "1.0.0", "pair": "EURUSD", "timeframe": "1h",
        "created_at": "2024-01-01T00:00:00Z", "entry_rules": [{"type": "always"}],
        "exit_rules": [], "position_size": 1, field: 5, "metadata": {"side": side},
    })
    result = run_backtest(configured, bars([100, 100, 100, 95]), 1000, CostModel(spread=2, slippage=1))
    assert result.trades[0].exit_price == expected_price
    assert result.trades[0].net_pnl == expected_net


def test_numeric_cost_strings_are_normalized():
    costs = CostModel(spread="0.1", commission="2")
    assert costs.spread == 0.1
    assert costs.commission == 2.0


@pytest.mark.parametrize("field,value", [("spread", -1), ("commission", -1), ("slippage", math.nan), ("swap", math.inf), ("conversion", "bad")])
def test_invalid_cost_inputs_are_rejected(field, value):
    with pytest.raises((TypeError, ValueError)):
        CostModel(**{field: value})
