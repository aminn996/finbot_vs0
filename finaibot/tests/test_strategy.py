import copy
import json

import pytest

from engine.strategy import Strategy, StrategyValidationError


def valid_document():
    return {
        "id": "synthetic", "version": "1.0.0", "pair": "EURUSD", "timeframe": "1h",
        "created_at": "2024-01-01T00:00:00Z", "entry_rules": [{"type": "always"}],
        "exit_rules": [{"type": "bars", "value": 1}], "position_size": 1,
        "metadata": {"side": "long", "nested": {"enabled": True}},
    }


@pytest.mark.parametrize("field,value", [("position_size", float("nan")), ("position_size", float("inf")), ("position_size", float("-inf")), ("stop_loss", float("nan")), ("take_profit", float("-inf")), ("position_size", "not-a-number")])
def test_non_finite_and_invalid_numeric_values_are_rejected(field, value):
    document = valid_document()
    document[field] = value
    with pytest.raises(StrategyValidationError):
        Strategy.from_dict(document)


@pytest.mark.parametrize("field", ["id", "version", "pair", "timeframe"])
def test_identifiers_must_be_non_empty(field):
    document = valid_document()
    document[field] = "   "
    with pytest.raises(StrategyValidationError, match="non-empty"):
        Strategy.from_dict(document)


@pytest.mark.parametrize("timestamp", ["not-a-date", "2024-01-01T00:00:00", "2024-01-01"])
def test_created_at_requires_timezone_aware_iso_timestamp(timestamp):
    document = valid_document()
    document["created_at"] = timestamp
    with pytest.raises(StrategyValidationError):
        Strategy.from_dict(document)


def test_direction_is_explicitly_validated():
    document = valid_document()
    document["metadata"]["side"] = "sideways"
    with pytest.raises(StrategyValidationError, match="side"):
        Strategy.from_dict(document)


def test_unsupported_rules_and_missing_parameters_are_rejected():
    document = valid_document()
    document["entry_rules"] = [{"type": "future_prediction"}]
    with pytest.raises(StrategyValidationError, match="unsupported"):
        Strategy.from_dict(document)
    document["entry_rules"] = [{"type": "close_above"}]
    with pytest.raises(StrategyValidationError, match="missing"):
        Strategy.from_dict(document)


@pytest.mark.parametrize("bars", [0, -1, 1.5, "-2"])
def test_bar_exit_requires_positive_integer(bars):
    document = valid_document()
    document["exit_rules"] = [{"type": "bars", "value": bars}]
    with pytest.raises(StrategyValidationError):
        Strategy.from_dict(document)


def test_nested_input_mutation_does_not_mutate_strategy():
    document = valid_document()
    original = copy.deepcopy(document)
    original.update({"stop_loss": None, "take_profit": None})
    strategy = Strategy.from_dict(document)
    document["metadata"]["nested"]["enabled"] = False
    document["entry_rules"][0]["type"] = "close_below"
    assert strategy.to_dict() == original
    with pytest.raises(TypeError):
        strategy.metadata["nested"]["enabled"] = False


def test_json_conversion_errors_are_normalized():
    with pytest.raises(StrategyValidationError, match="invalid strategy JSON") as error:
        Strategy.from_json("{")
    assert isinstance(error.value.__cause__, json.JSONDecodeError)
