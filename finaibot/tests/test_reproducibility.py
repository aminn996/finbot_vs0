import copy
import json

import pandas as pd

from engine.backtester import run_backtest
from engine.costs import CostModel
from engine.data import DataValidationConfig
from engine.risk import MarginConfig, RiskConfig
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
        "id": "reproducibility",
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


def test_completed_and_rejected_results_contain_serializable_metadata():
    completed = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000)
    rejected = run_backtest(strategy(), bars([100, 100, 100, 89]), 100)

    for result in (completed, rejected):
        assert result.metadata["run_id"]
        assert result.metadata["created_at"].endswith("Z")
        assert result.metadata["status"] == result.status
        assert json.loads(json.dumps(result.metadata, sort_keys=True)) == result.metadata


def test_run_ids_are_unique_but_identical_inputs_have_identical_input_hashes():
    first = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000)
    second = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000)

    assert first.metadata["run_id"] != second.metadata["run_id"]
    assert first.metadata["input_hash"] == second.metadata["input_hash"]
    assert first.metadata["strategy"]["hash"] == second.metadata["strategy"]["hash"]
    assert first.metadata["dataset_hash"] == second.metadata["dataset_hash"]


def test_strategy_and_dataset_changes_change_their_hashes():
    data = bars([100, 100, 100, 105])
    changed_data = data.copy()
    changed_data.loc[3, "close"] = 104
    baseline = run_backtest(strategy(), data, 1000)
    changed_strategy = run_backtest(strategy(position_size=2), data, 1000)
    changed_dataset = run_backtest(strategy(), changed_data, 1000)

    assert baseline.metadata["strategy"]["hash"] != changed_strategy.metadata["strategy"]["hash"]
    assert baseline.metadata["dataset_hash"] != changed_dataset.metadata["dataset_hash"]
    assert baseline.metadata["input_hash"] != changed_strategy.metadata["input_hash"]
    assert baseline.metadata["input_hash"] != changed_dataset.metadata["input_hash"]


def test_effective_cost_risk_and_margin_changes_change_configuration_fingerprints():
    data = bars([100, 100, 100, 105])
    baseline = run_backtest(strategy(), data, 1000)
    changed_cost = run_backtest(strategy(), data, 1000, CostModel(spread=1))
    changed_risk = run_backtest(strategy(), data, 1000, risk=RiskConfig(total_preventive_pct=0.07))
    changed_margin = run_backtest(strategy(), data, 1000, margin=MarginConfig(leverage=2))
    changed_validation = run_backtest(
        strategy(), data, 1000,
        data_validation=DataValidationConfig(expected_closure_dates=("2024-01-01",)),
    )

    assert baseline.metadata["cost_model_hash"] != changed_cost.metadata["cost_model_hash"]
    assert baseline.metadata["risk_config_hash"] != changed_risk.metadata["risk_config_hash"]
    assert baseline.metadata["margin_config_hash"] != changed_margin.metadata["margin_config_hash"]
    assert baseline.metadata["data_validation_config_hash"] != changed_validation.metadata["data_validation_config_hash"]
    assert baseline.metadata["input_hash"] != changed_cost.metadata["input_hash"]
    assert baseline.metadata["input_hash"] != changed_risk.metadata["input_hash"]
    assert baseline.metadata["input_hash"] != changed_margin.metadata["input_hash"]
    assert baseline.metadata["input_hash"] != changed_validation.metadata["input_hash"]


def test_hashing_does_not_mutate_supplied_strategy_or_market_data():
    data = bars([100, 100, 100, 105])
    original_data = data.copy(deep=True)
    configured = strategy(metadata={"side": "long", "nested": {"threshold": 1.0}})
    original_strategy = copy.deepcopy(configured.to_dict())

    run_backtest(configured, data, 1000)

    pd.testing.assert_frame_equal(data, original_data)
    assert configured.to_dict() == original_strategy


def test_equivalent_timestamp_and_numeric_forms_have_stable_strategy_hash():
    first = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000)
    equivalent = strategy(created_at="2024-01-01T01:00:00+01:00", position_size=1.0)
    second = run_backtest(equivalent, bars([100, 100, 100, 105]), 1000)

    assert first.metadata["strategy"]["hash"] == second.metadata["strategy"]["hash"]
