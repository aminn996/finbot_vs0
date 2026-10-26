import pandas as pd
import pytest

from engine.backtester import run_backtest
from engine.costs import CostModel
from engine.risk import MarginConfig, RiskConfig, RiskMonitor
from engine.strategy import Strategy


def bars(closes):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC"),
            "open": closes,
            "high": [value + 1 for value in closes],
            "low": [value - 1 for value in closes],
            "close": closes,
        }
    )


def strategy(**overrides):
    document = {
        "id": "risk-regression",
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


def risk_at(equity, *, daily=True):
    monitor = RiskMonitor(100)
    timestamp = pd.Timestamp("2024-01-01T00:00:00Z")
    monitor.begin_period(timestamp, 100 if daily else equity)
    monitor.check(timestamp + pd.Timedelta(hours=1), equity)
    return monitor


@pytest.mark.parametrize(
    ("equity", "expected_kind"),
    [(96.01, None), (96.0, "daily_preventive"), (95.99, "daily_preventive")],
)
def test_daily_preventive_boundary(equity, expected_kind):
    monitor = risk_at(equity)
    kinds = [event.kind for event in monitor.events]
    assert (expected_kind in kinds) if expected_kind else not kinds


@pytest.mark.parametrize(
    ("equity", "expected_kind"),
    [(95.01, None), (95.0, None), (94.99, "daily_hard")],
)
def test_daily_hard_boundary(equity, expected_kind):
    monitor = risk_at(equity)
    kinds = [event.kind for event in monitor.events]
    assert (expected_kind in kinds) if expected_kind else "daily_hard" not in kinds


@pytest.mark.parametrize(
    ("equity", "expected_kind"),
    [(92.01, None), (92.0, "total_preventive"), (91.99, "total_preventive")],
)
def test_total_preventive_boundary(equity, expected_kind):
    monitor = risk_at(equity, daily=False)
    kinds = [event.kind for event in monitor.events]
    assert (expected_kind in kinds) if expected_kind else not kinds


@pytest.mark.parametrize(
    ("equity", "expected_kind"),
    [(90.01, None), (90.0, None), (89.99, "total_hard")],
)
def test_total_hard_boundary(equity, expected_kind):
    monitor = risk_at(equity, daily=False)
    kinds = [event.kind for event in monitor.events]
    assert (expected_kind in kinds) if expected_kind else "total_hard" not in kinds


@pytest.mark.parametrize(
    "kwargs",
    [
        {"daily_preventive_pct": float("nan")},
        {"daily_hard_pct": float("inf")},
        {"total_preventive_pct": -0.01},
        {"daily_preventive_pct": 0.05, "daily_hard_pct": 0.05},
    ],
)
def test_invalid_risk_threshold_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RiskConfig(**kwargs)


def test_daily_preventive_stop_resets_at_new_day():
    monitor = RiskMonitor(100)
    day_one = pd.Timestamp("2024-01-01T00:00:00Z")
    day_two = pd.Timestamp("2024-01-02T00:00:00Z")
    monitor.begin_period(day_one, 100)
    monitor.check(day_one + pd.Timedelta(hours=1), 96)
    assert not monitor.entries_allowed(day_one + pd.Timedelta(hours=1))

    monitor.begin_period(day_two, 96)
    monitor.check(day_two, 96)
    assert monitor.entries_allowed(day_two)


def test_hard_loss_cap_rejects_backtest_after_preventive_protection():
    result = run_backtest(strategy(), bars([100, 100, 100, 89]), 100)

    assert result.status == "REJECTED"
    assert result.trades == ()
    assert result.equity_curve.iloc[-1]["equity"] == 89
    risk_kinds = [event["kind"] for event in result.events if event["kind"] in {
        "daily_preventive", "total_preventive", "daily_hard", "total_hard",
    }]
    assert risk_kinds == [
        "daily_preventive",
        "total_preventive",
        "daily_hard",
        "total_hard",
    ]
    hard = next(event for event in result.events if event["kind"] == "total_hard")
    assert hard["action"] == "terminate_and_reject_backtest"
    assert hard["phase"] == "candle_close"
    assert hard["equity"] == 89
    assert hard["threshold"] == 10
    assert hard["status"] == "REJECTED"


def test_risk_includes_entry_and_estimated_exit_costs():
    result = run_backtest(
        strategy(), bars([100, 100, 100, 96]), 100, CostModel(commission=1), margin=MarginConfig(leverage=2)
    )

    assert result.status == "REJECTED"
    assert result.equity_curve.iloc[-1]["equity"] == pytest.approx(94)
    hard = next(event for event in result.events if event["kind"] == "daily_hard")
    assert hard["loss"] == pytest.approx(6)
    assert hard["phase"] == "candle_close"


def test_stop_exit_is_checked_before_backtest_continues():
    frame = bars([100, 100, 100, 100])
    frame.loc[3, "low"] = 95
    result = run_backtest(strategy(stop_loss=5), frame, 100)

    assert result.status == "COMPLETED"
    assert result.trades[0].net_pnl == -5
    event = next(event for event in result.events if event["kind"] == "daily_preventive")
    assert event["phase"] == "exit"
    assert event["equity"] == 95


def test_take_profit_does_not_create_a_loss_risk_event():
    frame = bars([100, 100, 100, 100])
    frame.loc[3, "high"] = 105
    result = run_backtest(strategy(take_profit=5), frame, 100)

    assert result.status == "COMPLETED"
    assert result.trades[0].net_pnl == 5
    assert not any(event["kind"].endswith("hard") for event in result.events)


def test_risk_event_log_is_deterministic():
    first = run_backtest(strategy(), bars([100, 100, 100, 89]), 100)
    second = run_backtest(strategy(), bars([100, 100, 100, 89]), 100)

    assert first.status == second.status == "REJECTED"
    def without_run_id(events):
        return tuple({**event, "run_id": None} for event in events)

    assert without_run_id(first.events) == without_run_id(second.events)
