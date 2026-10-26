import json
import math

import pandas as pd
import pytest

from engine.backtester import run_backtest
from engine.costs import CostModel
from engine.execution import ExecutionConfig
from engine.risk import MarginConfig
from engine.strategy import Strategy


def bars(closes):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC"),
        "open": closes,
        "high": [value + 1 for value in closes],
        "low": [value - 1 for value in closes],
        "close": closes,
    })


def strategy(quantity=1):
    return Strategy.from_dict({
        "id": "execution-regression",
        "version": "1.0.0",
        "pair": "EURUSD",
        "timeframe": "1h",
        "created_at": "2024-01-01T00:00:00Z",
        "entry_rules": [{"type": "always"}],
        "exit_rules": [],
        "position_size": quantity,
        "metadata": {"side": "long"},
    })


def test_default_execution_is_a_compatibility_full_fill():
    result = run_backtest(strategy(3), bars([100, 100, 100, 105]), 1000)

    assert result.trades[0].quantity == 3
    assert result.trades[0].gross_pnl == 15
    assert result.metadata["execution_config_hash"]
    assert result.events[-1]["event_type"] == "backtest_completed"


def test_partial_entry_cancels_remainder_and_uses_filled_quantity_everywhere():
    result = run_backtest(
        strategy(3),
        bars([100, 100, 100, 105]),
        200,
        CostModel(commission=2),
        margin=MarginConfig(leverage=1),
        execution=ExecutionConfig(entry_fill_quantity_cap=1),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.quantity == 1
    assert trade.gross_pnl == 5
    assert trade.costs == 4
    assert trade.net_pnl == 1
    state = result.equity_curve.iloc[2]
    assert state.exposure == 100
    assert state.used_margin == 100
    partial = next(event for event in result.events if event["event_type"] == "entry_partially_filled")
    assert partial["details"] == {
        "side": "long",
        "requested_quantity": 3.0,
        "filled_quantity": 1.0,
        "remaining_quantity": 2.0,
        "fill_status": "partially_filled",
        "mid_price": 100.0,
        "execution_price": 100.0,
    }
    cancelled = next(event for event in result.events if event["event_type"] == "remaining_quantity_cancelled")
    assert cancelled["details"]["policy"] == "cancel"
    json.dumps(result.events)


def test_margin_and_exposure_are_checked_after_simulated_fill_quantity():
    result = run_backtest(
        strategy(3), bars([100, 100, 100, 105]), 100,
        margin=MarginConfig(max_exposure=100),
        execution=ExecutionConfig(entry_fill_quantity_cap=1),
    )

    assert result.trades[0].quantity == 1
    assert result.equity_curve.iloc[2].exposure == 100
    assert not any(event["kind"] == "exposure_limit" for event in result.events)


def test_rejected_order_records_requested_quantity_and_no_cost_or_position():
    result = run_backtest(strategy(3), bars([100, 100, 100, 105]), 100, margin=MarginConfig())

    rejected = next(event for event in result.events if event["kind"] == "insufficient_margin")
    assert rejected["details"]["requested_quantity"] == 3.0
    assert rejected["details"]["filled_quantity"] == 0.0
    assert rejected["details"]["remaining_quantity"] == 3.0
    assert not result.trades
    assert not any(event["event_type"] == "transaction_costs_applied" for event in result.events)


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf, "invalid"])
def test_invalid_fill_caps_are_rejected(value):
    with pytest.raises(ValueError):
        ExecutionConfig(entry_fill_quantity_cap=value)


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf, -math.inf, "invalid"])
def test_invalid_requested_quantities_are_rejected(value):
    with pytest.raises(ValueError):
        ExecutionConfig().fill_quantity(value)


def test_only_cancelled_remainder_policy_is_supported_in_m1():
    with pytest.raises(ValueError, match="remaining_quantity_policy"):
        ExecutionConfig(remaining_quantity_policy="keep_open")


def test_backtester_rejects_non_execution_config_before_simulation():
    with pytest.raises(ValueError, match="execution must be an ExecutionConfig") as caught:
        run_backtest(strategy(), bars([100, 100, 100, 105]), 1000, execution=None)

    assert caught.value.events[-1]["event_type"] == "backtest_rejected"
