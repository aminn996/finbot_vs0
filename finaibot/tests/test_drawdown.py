import pandas as pd
import pytest

from engine.backtester import _add_drawdown_columns, run_backtest
from engine.costs import CostModel
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


def strategy(side="long"):
    return Strategy.from_dict(
        {
            "id": "drawdown",
            "version": "1.0.0",
            "pair": "EURUSD",
            "timeframe": "1h",
            "created_at": "2024-01-01T00:00:00Z",
            "entry_rules": [{"type": "always"}],
            "exit_rules": [],
            "position_size": 1,
            "metadata": {"side": side},
        }
    )


def test_drawdown_tracks_peak_decline_recovery_and_new_peak():
    curve = _add_drawdown_columns(pd.DataFrame({"equity": [100, 110, 105, 110, 120]}), 100)

    assert curve["running_peak_equity"].tolist() == [100, 110, 110, 110, 120]
    assert curve["drawdown_amount"].tolist() == [0, 0, 5, 0, 0]
    assert curve["drawdown_pct_initial"].tolist() == [0, 0, 0.05, 0, 0]
    assert curve["drawdown_pct_peak"].tolist() == [0, 0, pytest.approx(5 / 110), 0, 0]


@pytest.mark.parametrize("side,closes", [("long", [100, 100, 100, 95]), ("short", [100, 100, 100, 105])])
def test_long_and_short_drawdown_uses_recorded_equity(side, closes):
    result = run_backtest(strategy(side), bars(closes), 100)

    assert result.equity_curve["equity"].tolist() == [100, 100, 100, 95]
    assert result.equity_curve["running_peak_equity"].tolist() == [100, 100, 100, 100]
    assert result.equity_curve["drawdown_amount"].tolist() == [0, 0, 0, 5]
    assert result.equity_curve.iloc[-1].drawdown_pct_initial == pytest.approx(0.05)


def test_costs_and_unrealized_loss_are_included_in_drawdown():
    result = run_backtest(
        strategy(), bars([100, 100, 100, 105]), 1000,
        CostModel(spread=2, commission=2, slippage=1, swap=3, conversion=4),
    )

    assert result.equity_curve.iloc[2].equity == 985
    assert result.equity_curve.iloc[2].drawdown_amount == 15
    assert result.equity_curve.iloc[2].drawdown_pct_initial == pytest.approx(0.015)
    assert result.equity_curve.iloc[-1].equity == 990
    assert result.equity_curve.iloc[-1].drawdown_amount == 10


def test_forced_closure_updates_final_drawdown_value():
    result = run_backtest(strategy(), bars([100, 100, 100, 95]), 100)

    assert result.trades[0].exit_reason == "end_of_data"
    assert result.equity_curve.iloc[-1].equity == 95
    assert result.equity_curve.iloc[-1].drawdown_amount == 5
    assert result.equity_curve.iloc[-1].drawdown_pct_initial == pytest.approx(0.05)


def test_hard_cap_rejection_preserves_drawdown_at_rejection():
    result = run_backtest(strategy(), bars([100, 100, 100, 89]), 100)

    assert result.status == "REJECTED"
    assert not result.trades
    assert result.equity_curve.iloc[-1].equity == 89
    assert result.equity_curve.iloc[-1].running_peak_equity == 100
    assert result.equity_curve.iloc[-1].drawdown_amount == 11
    assert result.equity_curve.iloc[-1].drawdown_pct_initial == pytest.approx(0.11)


def test_non_declining_equity_has_zero_drawdown():
    result = run_backtest(strategy(), bars([100, 100, 100, 105]), 100)

    assert (result.equity_curve["drawdown_amount"] == 0).all()
    assert (result.equity_curve["drawdown_pct_initial"] == 0).all()


def test_empty_or_invalid_equity_curve_is_handled_safely():
    empty = _add_drawdown_columns(pd.DataFrame(), 100)
    assert empty.empty
    assert "drawdown_amount" in empty.columns

    with pytest.raises(ValueError, match="non-finite"):
        _add_drawdown_columns(pd.DataFrame({"equity": [100, float("nan")]}), 100)
