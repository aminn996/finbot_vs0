import copy
import json

import pandas as pd
import pytest

from engine.backtester import run_backtest
from engine.strategy import Strategy, StrategyValidationError, validate_strategy_schema


def legacy_document():
    return {
        "id": "schema-test",
        "version": "1.0.0",
        "created_at": "2024-01-01T00:00:00Z",
        "pair": "EURUSD",
        "timeframe": "1h",
        "entry_rules": [{"type": "always"}],
        "exit_rules": [{"type": "bars", "value": 1}],
        "position_size": 1,
        "metadata": {"side": "long", "labels": {"source": "synthetic"}},
    }


def complete_future_document():
    document = legacy_document()
    document.pop("position_size")
    document.update(
        {
            "schema_version": "1.0",
            "parent_version": "0.9.0",
            "indicators": [{"name": "sma", "parameters": {"period": 20}}],
            "stop_loss": 0.01,
            "take_profit": 0.02,
            "risk_reward": {"minimum_ratio": 2, "enforce": True},
            "position_sizing": {"mode": "risk_fraction", "value": 0.01},
            "trend_filters": [{"type": "adx", "parameters": {"minimum": 20}}],
            "volatility_filters": [{"type": "atr", "parameters": {"period": 14}}],
            "market_filters": [{"type": "spread", "parameters": {"maximum": 0.0002}}],
            "allowed_sessions": [{"name": "london", "start": "08:00", "end": "16:00", "timezone": "UTC"}],
            "invalidation_conditions": [{"type": "max_drawdown", "parameters": {"value": 0.08}}],
            "capital_protection": {
                "daily_preventive_stop_pct": 0.04,
                "daily_loss_cap_pct": 0.05,
                "total_preventive_stop_pct": 0.08,
                "total_loss_cap_pct": 0.10,
            },
        }
    )
    return document


def test_schema_declares_all_versioned_strategy_sections():
    with open("strategy.schema.json", encoding="utf-8") as file:
        schema = json.load(file)
    expected = {
        "parent_version", "indicators", "risk_reward", "position_sizing", "trend_filters",
        "volatility_filters", "market_filters", "allowed_sessions", "invalidation_conditions",
        "capital_protection",
    }
    assert expected <= set(schema["properties"])
    assert schema["additionalProperties"] is False


def test_complete_schema_document_validates_but_current_executor_rejects_unsupported_behavior():
    document = complete_future_document()
    original = copy.deepcopy(document)
    normalized = validate_strategy_schema(document)
    assert normalized == original
    assert document == original
    with pytest.raises(StrategyValidationError, match="schema-valid but unsupported"):
        Strategy.from_dict(document)


@pytest.mark.parametrize(
    "field,value",
    [
        ("position_sizing", {"mode": "risk_fraction", "value": 1.1}),
        ("risk_reward", {"minimum_ratio": 0}),
        ("allowed_sessions", [{"name": "london", "start": "8:00", "end": "16:00"}]),
        ("capital_protection", {
            "daily_preventive_stop_pct": 0.05, "daily_loss_cap_pct": 0.04,
            "total_preventive_stop_pct": 0.08, "total_loss_cap_pct": 0.10,
        }),
    ],
)
def test_nested_schema_bounds_and_conditions_are_rejected(field, value):
    document = complete_future_document()
    document[field] = value
    with pytest.raises(StrategyValidationError):
        validate_strategy_schema(document)


def test_unknown_top_level_fields_and_malformed_nested_configurations_are_rejected():
    document = legacy_document()
    document["unexpected"] = True
    with pytest.raises(StrategyValidationError, match="unknown strategy fields"):
        Strategy.from_dict(document)

    document = complete_future_document()
    document["indicators"] = [{"name": "not_supported", "parameters": {}}]
    with pytest.raises(StrategyValidationError, match="unsupported indicator"):
        validate_strategy_schema(document)

    document = complete_future_document()
    document["indicators"][0]["parameters"]["period"] = float("nan")
    with pytest.raises(StrategyValidationError, match="non-finite"):
        validate_strategy_schema(document)


def test_parent_version_and_json_round_trip_are_preserved():
    document = legacy_document()
    document["schema_version"] = "1.0"
    document["parent_version"] = "0.9.0"
    strategy = Strategy.from_json(json.dumps(document))
    assert strategy.parent_version == "0.9.0"
    assert strategy.schema_version == "1.0"
    assert Strategy.from_json(json.dumps(strategy.to_dict())).to_dict() == strategy.to_dict()


def test_schema_validation_does_not_mutate_legacy_numeric_inputs():
    document = legacy_document()
    document["position_size"] = "1"
    document["entry_rules"] = [{"type": "close_above", "value": "100"}]
    original = copy.deepcopy(document)
    strategy = Strategy.from_dict(document)
    assert document == original
    assert strategy.position_size == 1.0
    assert strategy.entry_rules[0]["value"] == 100.0


def test_backtester_revalidates_directly_constructed_strategies_before_data_access():
    strategy = Strategy(
        id="unsafe", version="1.0.0", pair="EURUSD", timeframe="1h",
        created_at="2024-01-01T00:00:00Z", entry_rules=({"type": "future_prediction"},),
        exit_rules=(), position_size=1,
    )
    with pytest.raises(StrategyValidationError, match="unsupported"):
        run_backtest(strategy, pd.DataFrame(), 1000)
