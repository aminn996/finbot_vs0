import pandas as pd
import pytest

from engine.backtester import run_backtest
from engine.costs import CostModel
from engine.risk import RiskConfig
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


def strategy(side="long", entry_rules=None):
    return Strategy.from_dict(
        {
            "id": "equity-accounting",
            "version": "1.0.0",
            "pair": "EURUSD",
            "timeframe": "1h",
            "created_at": "2024-01-01T00:00:00Z",
            "entry_rules": [{"type": "always"}] if entry_rules is None else entry_rules,
            "exit_rules": [],
            "position_size": 1,
            "metadata": {"side": side},
        }
    )


def assert_reconciles(result, initial_capital):
    curve = result.equity_curve
    for _, row in curve.iterrows():
        assert row["cash"] == pytest.approx(initial_capital + row["realized_pnl"])
        assert row["equity"] == pytest.approx(row["cash"] + row["unrealized_pnl"])
        assert row["equity"] == pytest.approx(initial_capital + row["gross_pnl"] - row["total_costs"])


def test_flat_account_exposes_zero_pnl_and_cost_components():
    result = run_backtest(strategy(entry_rules=[]), bars([100, 100]), 1000)
    curve = result.equity_curve

    assert set(["cash", "realized_pnl", "unrealized_pnl", "gross_pnl", "total_costs", "equity"]).issubset(curve.columns)
    assert (curve["cash"] == 1000).all()
    for field in ("realized_pnl", "unrealized_pnl", "gross_pnl", "total_costs"):
        assert (curve[field] == 0).all()
    assert (curve["equity"] == 1000).all()
    assert_reconciles(result, 1000)


@pytest.mark.parametrize(
    "side,closes,expected_unrealized,expected_realized",
    [("long", [100, 100, 100, 105, 105], 5, 5), ("short", [100, 100, 100, 95, 95], 5, 5)],
)
def test_open_and_closed_long_short_pnl_are_explicit(side, closes, expected_unrealized, expected_realized):
    result = run_backtest(strategy(side), bars(closes), 1000)
    open_row = result.equity_curve.iloc[3]
    final_row = result.equity_curve.iloc[-1]

    assert open_row["realized_pnl"] == 0
    assert open_row["unrealized_pnl"] == expected_unrealized
    assert open_row["gross_pnl"] == expected_unrealized
    assert open_row["total_costs"] == 0
    assert final_row["realized_pnl"] == expected_realized
    assert final_row["unrealized_pnl"] == 0
    assert final_row["equity"] == 1000 + expected_realized
    assert_reconciles(result, 1000)


def test_costs_reconcile_open_and_forced_closed_equity():
    costs = CostModel(spread=2, commission=2, slippage=1, swap=3, conversion=4)
    result = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000, costs)
    open_row = result.equity_curve.iloc[2]
    final_row = result.equity_curve.iloc[-1]

    assert open_row["cash"] == 1000
    assert open_row["realized_pnl"] == 0
    assert open_row["gross_pnl"] == 0
    assert open_row["total_costs"] == 15
    assert open_row["unrealized_pnl"] == -15
    assert open_row["equity"] == 985
    assert final_row["gross_pnl"] == 5
    assert final_row["total_costs"] == 15
    assert final_row["realized_pnl"] == -10
    assert final_row["unrealized_pnl"] == 0
    assert final_row["equity"] == 990
    assert_reconciles(result, 1000)


def test_margin_fields_reconcile_with_explicit_equity():
    result = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000)
    open_row = result.equity_curve.iloc[2]
    final_row = result.equity_curve.iloc[-1]

    assert open_row["used_margin"] == open_row["required_margin"] == 100
    assert open_row["free_margin"] == open_row["equity"] - open_row["used_margin"]
    assert final_row["used_margin"] == final_row["required_margin"] == 0
    assert final_row["free_margin"] == final_row["equity"]
    assert_reconciles(result, 1000)


def test_unrealized_loss_fields_are_used_for_hard_cap_rejection():
    result = run_backtest(
        strategy(),
        bars([100, 100, 100, 94]),
        100,
        risk=RiskConfig(total_preventive_pct=0.08, total_hard_pct=0.10),
    )

    assert result.status == "REJECTED"
    rejection = result.equity_curve.iloc[-1]
    assert rejection["realized_pnl"] == 0
    assert rejection["unrealized_pnl"] == -6
    assert rejection["equity"] == 94
    assert rejection["drawdown_amount"] == 6
    assert_reconciles(result, 100)
