"""Validated, immutable strategy definitions used by the M1 backtester.

The JSON schema describes the versioned strategy interchange format.  The
current M1 engine executes only the legacy rule subset represented by
``Strategy``.  Schema-valid fields that would change trading behaviour are
therefore rejected before a backtest starts rather than silently ignored.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any
import json
import re


class StrategyValidationError(ValueError):
    """Raised when a strategy document is incomplete or unsafe."""


SUPPORTED_ENTRY_RULES = {
    "always": (),
    "close_above": ("value",),
    "close_below": ("value",),
    "crosses_above": ("value",),
    "crosses_below": ("value",),
}
SUPPORTED_EXIT_RULES = {"bars": ("value",)}
SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "m1", "m5", "m15", "m30", "h1", "h4", "d1"}
SUPPORTED_SCHEMA_VERSION = "1.0"
_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "strategy.schema.json"
_PAIR_PATTERN = re.compile(r"^[A-Z]{6}$")
_VERSION_PATTERN = re.compile(r"^[^\s]+$")
_TIME_PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")

# These sections are intentionally represented by the schema, but the M1
# execution engine does not yet implement their semantics.
UNSUPPORTED_EXECUTION_FIELDS = (
    "indicators", "risk_reward", "position_sizing", "trend_filters",
    "volatility_filters", "market_filters", "allowed_sessions",
    "invalidation_conditions", "capital_protection",
)


@dataclass(frozen=True)
class Strategy:
    id: str
    version: str
    pair: str
    timeframe: str
    created_at: str
    entry_rules: tuple[Mapping[str, Any], ...]
    exit_rules: tuple[Mapping[str, Any], ...]
    position_size: float
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parent_version: str | None = None
    schema_version: str | None = None

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "Strategy":
        normalized = validate_strategy_schema(document)
        unsupported = [name for name in UNSUPPORTED_EXECUTION_FIELDS if name in normalized]
        if unsupported:
            fields = ", ".join(unsupported)
            raise StrategyValidationError(
                f"strategy is schema-valid but unsupported by the current M1 engine: {fields}"
            )
        values = {key: _non_empty_string(normalized[key], key) for key in ("id", "version", "pair", "timeframe")}
        values["created_at"] = _timestamp(normalized["created_at"])
        metadata = _freeze(normalized.get("metadata", {}))
        if "side" in metadata and metadata["side"] not in {"long", "short"}:
            raise StrategyValidationError("metadata.side must be 'long' or 'short'")
        parent_version = normalized.get("parent_version")
        if parent_version is not None:
            parent_version = _non_empty_string(parent_version, "parent_version")
        return cls(
            **values,
            entry_rules=tuple(_validate_rule(rule, SUPPORTED_ENTRY_RULES, "entry") for rule in normalized["entry_rules"]),
            exit_rules=tuple(_validate_rule(rule, SUPPORTED_EXIT_RULES, "exit") for rule in normalized["exit_rules"]),
            position_size=_finite_number(normalized["position_size"], "position_size", positive=True),
            stop_loss=_optional_positive_number(normalized.get("stop_loss"), "stop_loss"),
            take_profit=_optional_positive_number(normalized.get("take_profit"), "take_profit"),
            metadata=metadata,
            parent_version=parent_version,
            schema_version=normalized.get("schema_version"),
        )

    @classmethod
    def from_json(cls, text: str) -> "Strategy":
        try:
            document = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            message = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            raise StrategyValidationError(f"invalid strategy JSON: {message}") from exc
        return cls.from_dict(document)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id, "version": self.version, "pair": self.pair, "timeframe": self.timeframe,
            "created_at": self.created_at, "entry_rules": [_thaw(rule) for rule in self.entry_rules],
            "exit_rules": [_thaw(rule) for rule in self.exit_rules], "position_size": self.position_size,
            "stop_loss": self.stop_loss, "take_profit": self.take_profit, "metadata": _thaw(self.metadata),
        }
        if self.parent_version is not None:
            result["parent_version"] = self.parent_version
        if self.schema_version is not None:
            result["schema_version"] = self.schema_version
        return result


def validate_strategy_schema(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy a versioned strategy document.

    This is separate from :meth:`Strategy.from_dict`: the complete schema can
    describe future strategy behaviour, while ``Strategy`` accepts only
    behaviour implemented by the current M1 engine.  Numeric strings in the
    original legacy fields are normalized for backward compatibility.
    """
    candidate = _copy_json_value(document, "strategy")
    if not isinstance(candidate, dict):
        raise StrategyValidationError("strategy must be an object")
    _normalize_legacy_numbers(candidate)
    _validate_schema_document(candidate)
    return candidate


def _validate_schema_document(document: dict[str, Any]) -> None:
    try:
        with _SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
            schema = json.load(schema_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyValidationError("strategy schema could not be loaded") from exc
    allowed = {
        "schema_version", "id", "version", "created_at", "parent_version", "pair", "timeframe",
        "entry_rules", "exit_rules", "indicators", "stop_loss", "take_profit", "risk_reward",
        "position_size", "position_sizing", "trend_filters", "volatility_filters", "market_filters",
        "allowed_sessions", "invalidation_conditions", "capital_protection", "metadata",
    }
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise StrategyValidationError(f"unknown strategy fields: {', '.join(unknown)}")
    required = ("id", "version", "created_at", "pair", "timeframe", "entry_rules", "exit_rules")
    missing = [name for name in required if name not in document]
    if missing:
        raise StrategyValidationError(f"missing required fields: {', '.join(missing)}")
    for name in ("id", "version", "pair", "timeframe"):
        _schema_non_empty_string(document[name], name)
    if document.get("schema_version", SUPPORTED_SCHEMA_VERSION) != SUPPORTED_SCHEMA_VERSION:
        raise StrategyValidationError(f"schema_version must be {SUPPORTED_SCHEMA_VERSION!r}")
    if "parent_version" in document and document["parent_version"] is not None:
        _schema_version_string(document["parent_version"], "parent_version")
    _validate_schema_timestamp(document["created_at"])
    if document["pair"] != document["pair"].upper() or not _PAIR_PATTERN.fullmatch(document["pair"]):
        raise StrategyValidationError("pair must be a six-letter uppercase currency pair")
    if document["timeframe"] not in SUPPORTED_TIMEFRAMES:
        raise StrategyValidationError(f"unsupported timeframe: {document['timeframe']!r}")
    _schema_array(document["entry_rules"], "entry_rules")
    for rule in document["entry_rules"]:
        _validate_schema_rule(rule, "entry")
    _schema_array(document["exit_rules"], "exit_rules")
    for rule in document["exit_rules"]:
        _validate_schema_rule(rule, "exit")
    if "position_size" not in document and "position_sizing" not in document:
        raise StrategyValidationError("one of position_size or position_sizing is required")
    if "position_size" in document:
        _schema_positive_number(document["position_size"], "position_size")
    for name in ("stop_loss", "take_profit"):
        if name in document and document[name] is not None:
            _schema_positive_number(document[name], name)
    if "metadata" in document:
        if not isinstance(document["metadata"], dict):
            raise StrategyValidationError("metadata must be an object")
        if "side" in document["metadata"] and document["metadata"]["side"] not in {"long", "short"}:
            raise StrategyValidationError("metadata.side must be 'long' or 'short'")
        _validate_json_values(document["metadata"], "metadata")
    _validate_indicators(document)
    _validate_risk_reward(document)
    _validate_position_sizing(document)
    _validate_filters(document)
    _validate_sessions(document)
    _validate_invalidation(document)
    _validate_capital_protection(document)
    _validate_json_schema(document, schema, schema, "strategy")


def _validate_json_schema(instance: Any, schema: dict[str, Any], root: dict[str, Any], path: str) -> None:
    """Validate the JSON-Schema subset used by strategy.schema.json.

    Keeping this small validator in the standard library avoids making the
    backtester depend on a third-party package merely to load strategy data.
    Semantic checks that JSON Schema cannot express here (finite values and
    the relationship between preventive and hard caps) remain below it.
    """
    if "$ref" in schema:
        reference = schema["$ref"]
        if not reference.startswith("#/$defs/"):
            raise StrategyValidationError(f"unsupported schema reference at {path}")
        definition = root["$defs"][reference.removeprefix("#/$defs/")]
        _validate_json_schema(instance, definition, root, path)
        return
    if "oneOf" in schema:
        matches = 0
        for option in schema["oneOf"]:
            try:
                _validate_json_schema(instance, option, root, path)
            except StrategyValidationError:
                continue
            matches += 1
        if matches != 1:
            raise StrategyValidationError(f"{path} does not match exactly one allowed schema variant")
    for subschema in schema.get("allOf", []):
        _validate_json_schema(instance, subschema, root, path)
    if "if" in schema:
        try:
            _validate_json_schema(instance, schema["if"], root, path)
        except StrategyValidationError:
            pass
        else:
            if "then" in schema:
                _validate_json_schema(instance, schema["then"], root, path)
    if "not" in schema:
        try:
            _validate_json_schema(instance, schema["not"], root, path)
        except StrategyValidationError:
            pass
        else:
            raise StrategyValidationError(f"{path} matches a forbidden schema")
    if "const" in schema and instance != schema["const"]:
        raise StrategyValidationError(f"{path} must equal {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise StrategyValidationError(f"{path} has an unsupported value")
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_json_type_matches(instance, type_name) for type_name in types):
            raise StrategyValidationError(f"{path} has the wrong type")
    if isinstance(instance, dict):
        for required in schema.get("required", []):
            if required not in instance:
                raise StrategyValidationError(f"{path} is missing required field {required!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = set(instance) - set(properties)
            if unknown:
                raise StrategyValidationError(f"{path} contains unknown field(s): {', '.join(sorted(unknown))}")
        for key, value in instance.items():
            if key in properties:
                _validate_json_schema(value, properties[key], root, f"{path}.{key}")
    if isinstance(instance, list) and "items" in schema:
        for index, value in enumerate(instance):
            _validate_json_schema(value, schema["items"], root, f"{path}[{index}]")
    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            raise StrategyValidationError(f"{path} is shorter than the minimum length")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            raise StrategyValidationError(f"{path} has an invalid format")
    if isinstance(instance, list) and len(instance) < schema.get("minItems", 0):
        raise StrategyValidationError(f"{path} has too few items")
    if _json_type_matches(instance, "number") or _json_type_matches(instance, "integer"):
        numeric = float(instance)
        if not isfinite(numeric):
            raise StrategyValidationError(f"{path} must be finite")
        if "minimum" in schema and numeric < schema["minimum"]:
            raise StrategyValidationError(f"{path} is below the minimum")
        if "exclusiveMinimum" in schema and numeric <= schema["exclusiveMinimum"]:
            raise StrategyValidationError(f"{path} is not above the exclusive minimum")
        if "maximum" in schema and numeric > schema["maximum"]:
            raise StrategyValidationError(f"{path} is above the maximum")


def _json_type_matches(value: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(value, dict)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def _validate_schema_rule(rule: Any, kind: str) -> None:
    if not isinstance(rule, dict):
        raise StrategyValidationError(f"{kind} rule must be an object")
    supported = SUPPORTED_ENTRY_RULES if kind == "entry" else SUPPORTED_EXIT_RULES
    rule_type = rule.get("type")
    if rule_type not in supported:
        raise StrategyValidationError(f"unsupported {kind} rule type: {rule_type!r}")
    expected = {"type", *supported[rule_type]}
    unknown = sorted(set(rule) - expected)
    if unknown:
        raise StrategyValidationError(f"unknown fields in {kind} rule: {', '.join(unknown)}")
    missing = [name for name in supported[rule_type] if name not in rule]
    if missing:
        raise StrategyValidationError(f"{kind} rule {rule_type!r} missing required fields: {', '.join(missing)}")
    if "value" in expected:
        _schema_number(rule["value"], f"{kind} rule {rule_type!r}.value")
        if rule_type == "bars" and (not isinstance(rule["value"], int) or isinstance(rule["value"], bool) or rule["value"] < 1):
            raise StrategyValidationError("exit rule 'bars.value' must be a positive integer")


def _validate_indicators(document: dict[str, Any]) -> None:
    if "indicators" not in document:
        return
    _schema_array(document["indicators"], "indicators")
    supported_names = {"sma", "ema", "rsi", "atr", "macd", "bollinger_bands"}
    for indicator in document["indicators"]:
        if not isinstance(indicator, dict) or set(indicator) != {"name", "parameters"}:
            raise StrategyValidationError("indicator requires only name and parameters")
        if indicator["name"] not in supported_names:
            raise StrategyValidationError(f"unsupported indicator: {indicator.get('name')!r}")
        _schema_parameters(indicator["parameters"], "indicator parameters")


def _validate_risk_reward(document: dict[str, Any]) -> None:
    value = document.get("risk_reward")
    if value is None:
        return
    if not isinstance(value, dict) or set(value) - {"minimum_ratio", "enforce"} or "minimum_ratio" not in value:
        raise StrategyValidationError("risk_reward requires minimum_ratio and optional enforce")
    _schema_positive_number(value["minimum_ratio"], "risk_reward.minimum_ratio")
    if "enforce" in value and not isinstance(value["enforce"], bool):
        raise StrategyValidationError("risk_reward.enforce must be boolean")
    if value.get("enforce") and (document.get("stop_loss") is None or document.get("take_profit") is None):
        raise StrategyValidationError("risk_reward.enforce requires stop_loss and take_profit")


def _validate_position_sizing(document: dict[str, Any]) -> None:
    value = document.get("position_sizing")
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"mode", "value"}:
        raise StrategyValidationError("position_sizing requires mode and value")
    if value["mode"] not in {"fixed_units", "risk_fraction", "equity_percent"}:
        raise StrategyValidationError(f"unsupported position_sizing mode: {value['mode']!r}")
    _schema_positive_number(value["value"], "position_sizing.value")
    maximum = 1 if value["mode"] == "risk_fraction" else 100 if value["mode"] == "equity_percent" else None
    if maximum is not None and value["value"] > maximum:
        raise StrategyValidationError(f"position_sizing.value must be <= {maximum} for {value['mode']}")
    if "position_size" in document:
        raise StrategyValidationError("position_size and position_sizing cannot both be supplied")


def _validate_filters(document: dict[str, Any]) -> None:
    allowed = {
        "trend_filters": {"sma", "ema", "adx"},
        "volatility_filters": {"atr", "bollinger_width", "atr_percentile"},
        "market_filters": {"spread", "news", "liquidity", "day_of_week"},
    }
    for field_name, types in allowed.items():
        if field_name not in document:
            continue
        _schema_array(document[field_name], field_name)
        for item in document[field_name]:
            if not isinstance(item, dict) or set(item) != {"type", "parameters"}:
                raise StrategyValidationError(f"{field_name} entries require type and parameters")
            if item["type"] not in types:
                raise StrategyValidationError(f"unsupported {field_name} type: {item.get('type')!r}")
            _schema_parameters(item["parameters"], f"{field_name} parameters")


def _validate_sessions(document: dict[str, Any]) -> None:
    if "allowed_sessions" not in document:
        return
    _schema_array(document["allowed_sessions"], "allowed_sessions")
    names = {"tokyo", "london", "new_york", "london_new_york_overlap"}
    for session in document["allowed_sessions"]:
        if not isinstance(session, dict) or set(session) - {"name", "start", "end", "timezone"} or not {"name", "start", "end"} <= set(session):
            raise StrategyValidationError("allowed session requires name, start, and end")
        if session["name"] not in names:
            raise StrategyValidationError(f"unsupported session: {session.get('name')!r}")
        if not isinstance(session["start"], str) or not _TIME_PATTERN.fullmatch(session["start"]):
            raise StrategyValidationError("session start must be HH:MM")
        if not isinstance(session["end"], str) or not _TIME_PATTERN.fullmatch(session["end"]):
            raise StrategyValidationError("session end must be HH:MM")
        if "timezone" in session and (not isinstance(session["timezone"], str) or not session["timezone"].strip()):
            raise StrategyValidationError("session timezone must be a non-empty string")


def _validate_invalidation(document: dict[str, Any]) -> None:
    if "invalidation_conditions" not in document:
        return
    _schema_array(document["invalidation_conditions"], "invalidation_conditions")
    types = {"opposite_signal", "time_expiry", "max_drawdown", "data_quality"}
    for condition in document["invalidation_conditions"]:
        if not isinstance(condition, dict) or set(condition) != {"type", "parameters"}:
            raise StrategyValidationError("invalidation condition requires type and parameters")
        if condition["type"] not in types:
            raise StrategyValidationError(f"unsupported invalidation condition: {condition.get('type')!r}")
        _schema_parameters(condition["parameters"], "invalidation parameters")


def _validate_capital_protection(document: dict[str, Any]) -> None:
    value = document.get("capital_protection")
    if value is None:
        return
    fields = {"daily_preventive_stop_pct", "daily_loss_cap_pct", "total_preventive_stop_pct", "total_loss_cap_pct"}
    if not isinstance(value, dict) or set(value) != fields:
        raise StrategyValidationError("capital_protection requires the four M1 threshold fields")
    for name in fields:
        _schema_bounded_number(value[name], f"capital_protection.{name}", maximum=1)
    if value["daily_preventive_stop_pct"] >= value["daily_loss_cap_pct"]:
        raise StrategyValidationError("daily preventive stop must be below daily loss cap")
    if value["total_preventive_stop_pct"] >= value["total_loss_cap_pct"]:
        raise StrategyValidationError("total preventive stop must be below total loss cap")


def _schema_parameters(value: Any, field_name: str) -> None:
    if not isinstance(value, dict):
        raise StrategyValidationError(f"{field_name} must be an object")
    _validate_json_values(value, field_name)


def _validate_json_values(value: Any, field_name: str) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise StrategyValidationError(f"{field_name} contains a non-finite number")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrategyValidationError(f"{field_name} contains a non-string key")
            _validate_json_values(item, f"{field_name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_values(item, f"{field_name}[{index}]")
    elif not isinstance(value, (str, int, float, bool)) and value is not None:
        raise StrategyValidationError(f"{field_name} contains a non-JSON value")


def _schema_array(value: Any, field_name: str) -> None:
    if not isinstance(value, list):
        raise StrategyValidationError(f"{field_name} must be an array")


def _schema_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip() or (field_name in {"version", "parent_version"} and not _VERSION_PATTERN.fullmatch(value)):
        raise StrategyValidationError(f"{field_name} must be a non-empty string")


def _schema_version_string(value: Any, field_name: str) -> None:
    _schema_non_empty_string(value, field_name)


def _schema_number(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise StrategyValidationError(f"{field_name} must be a finite number")


def _schema_positive_number(value: Any, field_name: str) -> None:
    _schema_number(value, field_name)
    if value <= 0:
        raise StrategyValidationError(f"{field_name} must be positive")


def _schema_bounded_number(value: Any, field_name: str, *, maximum: float) -> None:
    _schema_positive_number(value, field_name)
    if value > maximum:
        raise StrategyValidationError(f"{field_name} must be <= {maximum}")


def _validate_schema_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise StrategyValidationError("created_at must be a timezone-aware ISO-8601 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except (TypeError, ValueError) as exc:
        raise StrategyValidationError(f"created_at is not a valid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StrategyValidationError("created_at must include a timezone")


def _copy_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise StrategyValidationError(f"{field_name} contains a non-string key")
            result[key] = _copy_json_value(item, f"{field_name}.{key}")
        return result
    if isinstance(value, list):
        return [_copy_json_value(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise StrategyValidationError(f"{field_name} contains a non-JSON value")


def _normalize_legacy_numbers(document: dict[str, Any]) -> None:
    for field_name in ("position_size", "stop_loss", "take_profit"):
        value = document.get(field_name)
        if isinstance(value, str):
            try:
                document[field_name] = float(value)
            except (TypeError, ValueError, OverflowError):
                pass
    for rules in (document.get("entry_rules"), document.get("exit_rules")):
        if not isinstance(rules, list):
            continue
        for rule in rules:
            if isinstance(rule, dict) and isinstance(rule.get("value"), str):
                try:
                    parsed = float(rule["value"])
                    rule["value"] = int(parsed) if rule.get("type") == "bars" and parsed.is_integer() else parsed
                except (TypeError, ValueError, OverflowError):
                    pass


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyValidationError(f"{field_name} must be a non-empty string")
    return value


def _timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StrategyValidationError("created_at must be a timezone-aware ISO-8601 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except (TypeError, ValueError) as exc:
        raise StrategyValidationError(f"created_at is not a valid ISO-8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StrategyValidationError("created_at must include a timezone")
    return value


def _finite_number(value: Any, field_name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise StrategyValidationError(f"{field_name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StrategyValidationError(f"{field_name} must be a finite number; got {value!r}") from exc
    if not isfinite(result) or (positive and result <= 0):
        requirement = "positive and finite" if positive else "finite"
        raise StrategyValidationError(f"{field_name} must be {requirement}; got {value!r}")
    return result


def _optional_positive_number(value: Any, field_name: str) -> float | None:
    return None if value is None else _finite_number(value, field_name, positive=True)


def _validate_rule(rule: Any, supported: dict[str, tuple[str, ...]], kind: str) -> Mapping[str, Any]:
    if not isinstance(rule, Mapping):
        raise StrategyValidationError(f"{kind} rule must be an object")
    rule_type = rule.get("type")
    if rule_type not in supported:
        raise StrategyValidationError(f"unsupported {kind} rule type: {rule_type!r}")
    missing = [name for name in supported[rule_type] if name not in rule]
    if missing:
        raise StrategyValidationError(f"{kind} rule {rule_type!r} missing required fields: {', '.join(missing)}")
    result = dict(rule)
    if "value" in supported[rule_type]:
        value = _finite_number(rule["value"], f"{kind} rule {rule_type!r}.value")
        if rule_type == "bars" and (not value.is_integer() or value < 1):
            raise StrategyValidationError("exit rule 'bars.value' must be a positive integer")
        result["value"] = int(value) if rule_type == "bars" else value
    return _freeze(result)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset)):
        return [_thaw(item) for item in value]
    return value
