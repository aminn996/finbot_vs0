import json

import pandas as pd
import pytest

from engine.backtester import run_backtest
from engine.costs import CostModel
from engine.data import DataValidationError
from engine.risk import MarginConfig, RiskConfig
from engine.strategy import Strategy


def bars(closes):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC"),
        "open": closes,
        "high": [value + 1 for value in closes],
        "low": [value - 1 for value in closes],
        "close": closes,
    })


def strategy(**overrides):
    document = {
        "id": "event-regression",
        "version": "1.0.0",
        "pair": "EURUSD",
        "timeframe": "1h",
        "created_at": "2024-01-01T00:00:00Z",
        "entry_rules": [{"type": "always"}],
        "exit_rules": [],
        "position_size": 1,
        "metadata": {"side": "long"},
    }
    document.update(overrides)
    return Strategy.from_dict(document)


def event_types(result):
    return [event["event_type"] for event in result.events]


def test_completed_run_has_ordered_lifecycle_and_json_safe_events():
    result = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000, CostModel(commission=2))
    types = event_types(result)

    assert types[0:3] == ["backtest_started", "strategy_validated", "data_validated"]
    assert types[-1] == "backtest_completed"
    assert types.index("entry_signal") < types.index("entry_accepted")
    assert types.index("entry_accepted") < types.index("position_opened")
    assert types.index("position_opened") < types.index("forced_end_of_data_close")
    assert types.index("forced_end_of_data_close") < types.index("position_closed")
    assert types.index("position_closed") < types.index("backtest_completed")
    assert [event["sequence"] for event in result.events] == list(range(1, len(result.events) + 1))
    assert all(event["source"] == "simulated" for event in result.events)
    assert all(event["run_id"] == result.metadata["run_id"] for event in result.events)
    json.dumps(result.events)

    trade_events = [event for event in result.events if event["trade_id"] is not None]
    assert {event["trade_id"] for event in trade_events} == {"trade-1"}
    cost_events = [event for event in result.events if event["event_type"] == "transaction_costs_applied"]
    assert [event["kind"] for event in cost_events] == ["entry_costs", "exit_costs"]
    assert cost_events[0]["details"]["commission"] == 2


@pytest.mark.parametrize(
    ("field", "expected_event"),
    [("stop_loss", "stop_loss_executed"), ("take_profit", "take_profit_executed")],
)
def test_protective_exit_events_are_recorded_once(field, expected_event):
    frame = bars([100, 100, 100, 100])
    frame.loc[3, "low" if field == "stop_loss" else "high"] = 95 if field == "stop_loss" else 105
    result = run_backtest(strategy(**{field: 5}), frame, 1000)

    assert event_types(result).count(expected_event) == 1
    assert event_types(result).count("position_closed") == 1
    assert event_types(result)[-1] == "backtest_completed"
    assert not any(event["event_type"] == "forced_end_of_data_close" for event in result.events)


def test_insufficient_margin_is_a_rejected_entry_and_run_still_completes():
    result = run_backtest(strategy(), bars([100, 100, 100, 105]), 50, margin=MarginConfig())

    rejected = [event for event in result.events if event["kind"] == "insufficient_margin"]
    assert rejected
    assert all(event["event_type"] == "entry_rejected" for event in rejected)
    assert rejected[0]["details"]["required_margin"] == 100
    assert result.status == "COMPLETED"
    assert result.events[-1]["event_type"] == "backtest_completed"


def test_risk_preventive_and_hard_events_are_structured_and_terminal_status_is_rejected():
    result = run_backtest(
        strategy(), bars([100, 100, 100, 89]), 100,
        risk=RiskConfig(total_preventive_pct=0.01, total_hard_pct=0.02),
    )
    risk_events = [event for event in result.events if event["kind"] in {"daily_preventive", "total_preventive", "daily_hard", "total_hard"}]

    assert [event["event_type"] for event in risk_events] == [
        "risk.daily_preventive", "risk.total_preventive", "risk.daily_hard", "risk.total_hard",
    ]
    assert all(event["phase"] for event in risk_events)
    assert all(event["run_id"] == result.metadata["run_id"] for event in risk_events)
    assert result.status == "REJECTED"
    assert result.events[-1]["event_type"] == "backtest_rejected"
    assert not any(event["event_type"] == "forced_end_of_data_close" for event in result.events)


def test_validation_failure_keeps_existing_exception_behavior_and_exposes_event_ledger():
    invalid = bars([100, 100])
    invalid.loc[1, "high"] = 99
    with pytest.raises(DataValidationError) as raised:
        run_backtest(strategy(), invalid, 1000)

    events = raised.value.events
    assert [event["event_type"] for event in events] == [
        "backtest_started", "strategy_validated", "data_validation_failed", "backtest_rejected",
    ]
    assert events[-1]["status"] == "REJECTED"
    json.dumps(events)


def test_margin_and_cost_actions_do_not_duplicate_one_lifecycle_action():
    result = run_backtest(
        strategy(), bars([100, 100, 100, 105]), 1000,
        CostModel(spread=2, commission=1, slippage=1),
    )
    assert sum(event["event_type"] == "position_opened" for event in result.events) == 1
    assert sum(event["event_type"] == "position_closed" for event in result.events) == 1
    assert sum(event["event_type"] == "transaction_costs_applied" for event in result.events) == 2
    assert len({event["sequence"] for event in result.events}) == len(result.events)
