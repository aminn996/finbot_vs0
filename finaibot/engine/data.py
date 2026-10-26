"""Market-data loading and validation. Input files are never modified."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import pandas as pd
import numpy as np


class DataValidationError(ValueError):
    """Raised when market data cannot be safely simulated."""

    def __init__(self, message: str, report: "DataValidationReport | None" = None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class DataValidationIssue:
    code: str
    severity: str
    message: str
    affected_timestamps: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataValidationConfig:
    """Explicit closure policy for deterministic historical-data validation.

    Weekend dates are expected closures by default. Holiday dates must be
    supplied by the caller; no external calendar is inferred or downloaded.
    Any other gap is an error under the conservative default policy.
    """

    expected_closure_dates: tuple[str, ...] = ()
    weekend_days: tuple[int, ...] = (5, 6)
    unknown_closure_policy: str = "reject"

    def __post_init__(self) -> None:
        if self.unknown_closure_policy not in {"reject"}:
            raise ValueError("unknown_closure_policy must be 'reject'")
        if any(isinstance(day, bool) or not isinstance(day, int) or day < 0 or day > 6 for day in self.weekend_days):
            raise ValueError("weekend_days must contain integers from 0 through 6")
        normalized_dates = []
        for value in self.expected_closure_dates:
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid expected closure date: {value!r}") from exc
            if timestamp.tzinfo is not None:
                timestamp = timestamp.tz_convert("UTC")
            normalized_dates.append(timestamp.date().isoformat())
        object.__setattr__(self, "expected_closure_dates", tuple(sorted(set(normalized_dates))))
        object.__setattr__(self, "weekend_days", tuple(sorted(set(self.weekend_days))))


@dataclass(frozen=True)
class DataValidationReport:
    issues: tuple[DataValidationIssue, ...]
    safe_to_use: bool
    row_count: int
    requested_timeframe: str | None
    expected_interval: str | None
    normalized_frame: pd.DataFrame


@dataclass(frozen=True)
class MarketData:
    frame: pd.DataFrame
    sha256: str
    validation_report: DataValidationReport | None = None


@dataclass(frozen=True)
class _Timeframe:
    name: str
    interval: pd.Timedelta


REQUIRED = ("timestamp", "open", "high", "low", "close")
SPREAD_COLUMN = "spread"
BID_ASK_COLUMNS = tuple(
    f"{side}_{field}"
    for side in ("bid", "ask")
    for field in ("open", "high", "low", "close")
)
EXECUTION_DATA_COLUMNS = (SPREAD_COLUMN,) + BID_ASK_COLUMNS
_SUPPORTED_TIMEFRAMES = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}
_TIMEFRAME_ALIASES = {
    "m1": "1m",
    "m5": "5m",
    "m15": "15m",
    "m30": "30m",
    "h1": "1h",
    "h4": "4h",
    "d1": "1d",
}


def validate_market_data(
    frame: pd.DataFrame,
    timeframe: str | None = None,
    *,
    config: DataValidationConfig = DataValidationConfig(),
    minimum_rows: int | None = None,
) -> pd.DataFrame:
    """Validate and normalize market data, preserving the existing return type."""
    report = validate_market_data_report(frame, timeframe, config=config, minimum_rows=minimum_rows)
    if not report.safe_to_use:
        raise DataValidationError(_report_message(report), report)
    return report.normalized_frame


def validate_market_data_report(
    frame: pd.DataFrame,
    timeframe: str | None = None,
    *,
    config: DataValidationConfig = DataValidationConfig(),
    minimum_rows: int | None = None,
) -> DataValidationReport:
    """Return deterministic, inspectable validation results without raising."""
    issues: list[DataValidationIssue] = []
    requested_timeframe = _normalize_timeframe_name(timeframe, issues)
    expected_interval = _SUPPORTED_TIMEFRAMES.get(requested_timeframe) if requested_timeframe else None
    expected_interval_text = str(expected_interval) if expected_interval is not None else None

    if not isinstance(frame, pd.DataFrame):
        issues.append(DataValidationIssue("invalid_dataset_type", "error", "market data must be a pandas DataFrame"))
        return DataValidationReport(tuple(issues), False, 0, requested_timeframe, expected_interval_text, pd.DataFrame())

    result = frame.copy(deep=True)
    missing_columns = [column for column in REQUIRED if column not in result.columns]
    if missing_columns:
        issues.append(DataValidationIssue("missing_required_columns", "error", f"missing required columns: {', '.join(missing_columns)}"))
        return DataValidationReport(tuple(issues), False, len(result), requested_timeframe, expected_interval_text, result)

    if result.empty:
        issues.append(DataValidationIssue("empty_dataset", "error", "market data is empty"))
    if minimum_rows is not None:
        if isinstance(minimum_rows, bool) or not isinstance(minimum_rows, int) or minimum_rows < 1:
            raise ValueError("minimum_rows must be a positive integer")
        if len(result) < minimum_rows:
            issues.append(DataValidationIssue("insufficient_history", "error", f"market data has {len(result)} rows; at least {minimum_rows} are required"))

    raw_timestamps = result["timestamp"].copy()
    timestamps = pd.to_datetime(raw_timestamps, utc=True, errors="coerce")
    invalid_timestamp_mask = timestamps.isna()
    if invalid_timestamp_mask.any():
        issues.append(DataValidationIssue(
            "invalid_timestamp",
            "error",
            "timestamps must be valid and timezone-normalizable",
            _affected_values(raw_timestamps, invalid_timestamp_mask),
        ))
    result["timestamp"] = timestamps

    if not invalid_timestamp_mask.any():
        if timestamps.duplicated().any():
            issues.append(DataValidationIssue(
                "duplicate_timestamps",
                "error",
                "timestamps must be unique",
                _affected_values(timestamps, timestamps.duplicated(keep=False)),
            ))
        if not timestamps.is_monotonic_increasing:
            issues.append(DataValidationIssue(
                "non_monotonic_timestamps",
                "error",
                "timestamps must be strictly increasing",
                _affected_values(timestamps, pd.Series(True, index=timestamps.index)),
            ))

    prices = result[list(REQUIRED[1:])].apply(pd.to_numeric, errors="coerce")
    for column in REQUIRED[1:]:
        original = result[column]
        missing_mask = original.isna()
        invalid_numeric_mask = prices[column].isna() & ~missing_mask
        non_finite_mask = prices[column].notna() & ~np.isfinite(prices[column].to_numpy())
        non_positive_mask = prices[column].notna() & (prices[column] <= 0)
        if missing_mask.any():
            issues.append(DataValidationIssue("missing_values", "error", f"{column} contains missing values", _affected_timestamps(result["timestamp"], missing_mask)))
        if invalid_numeric_mask.any():
            issues.append(DataValidationIssue("invalid_numeric_price", "error", f"{column} contains non-numeric prices", _affected_timestamps(result["timestamp"], invalid_numeric_mask)))
        if non_finite_mask.any():
            issues.append(DataValidationIssue("non_finite_price", "error", f"{column} contains non-finite prices", _affected_timestamps(result["timestamp"], non_finite_mask)))
        if non_positive_mask.any():
            issues.append(DataValidationIssue("non_positive_price", "error", f"{column} must be positive", _affected_timestamps(result["timestamp"], non_positive_mask)))

    result.loc[:, list(REQUIRED[1:])] = prices
    valid_prices = prices.notna().all(axis=1) & np.isfinite(prices).all(axis=1)
    if valid_prices.any():
        relationship_invalid = (
            (prices["low"] > prices["open"])
            | (prices["low"] > prices["close"])
            | (prices["high"] < prices["open"])
            | (prices["high"] < prices["close"])
            | (prices["low"] > prices["high"])
        ) & valid_prices
        if relationship_invalid.any():
            issues.append(DataValidationIssue(
                "inconsistent_ohlc",
                "error",
                "each candle must satisfy low <= open/close <= high and low <= high",
                _affected_timestamps(result["timestamp"], relationship_invalid),
            ))

    _validate_execution_data(result, issues)

    if expected_interval is not None and not invalid_timestamp_mask.any() and len(result) > 1 and timestamps.is_monotonic_increasing and not timestamps.duplicated().any():
        _append_gap_issues(issues, timestamps, expected_interval, config)
    elif timeframe is None:
        issues.append(DataValidationIssue(
            "timeframe_unavailable",
            "warning",
            "gap validation was not performed because no requested timeframe was supplied",
        ))

    safe_to_use = not any(issue.severity == "error" for issue in issues)
    return DataValidationReport(tuple(issues), safe_to_use, len(result), requested_timeframe, expected_interval_text, result.reset_index(drop=True))


def load_parquet(
    path: str | Path,
    timeframe: str | None = None,
    *,
    config: DataValidationConfig = DataValidationConfig(),
    minimum_rows: int | None = None,
) -> MarketData:
    file_path = Path(path)
    raw = file_path.read_bytes()
    frame = pd.read_parquet(file_path)
    report = validate_market_data_report(frame, timeframe, config=config, minimum_rows=minimum_rows)
    if not report.safe_to_use:
        raise DataValidationError(_report_message(report), report)
    return MarketData(report.normalized_frame, hashlib.sha256(raw).hexdigest(), report)


def _normalize_timeframe_name(timeframe: str | None, issues: list[DataValidationIssue]) -> str | None:
    if timeframe is None:
        return None
    if not isinstance(timeframe, str) or not timeframe.strip():
        issues.append(DataValidationIssue("invalid_timeframe", "error", "timeframe must be a supported non-empty string"))
        return None
    normalized = timeframe.strip().lower()
    normalized = _TIMEFRAME_ALIASES.get(normalized, normalized)
    if normalized not in _SUPPORTED_TIMEFRAMES:
        issues.append(DataValidationIssue("unsupported_timeframe", "error", f"unsupported timeframe: {timeframe!r}"))
        return None
    return normalized


def _validate_execution_data(result: pd.DataFrame, issues: list[DataValidationIssue]) -> None:
    """Validate optional execution-price data without changing OHLC semantics.

    ``spread`` is a full bid/ask width in the dataset's explicit price units.
    A bid/ask representation must provide complete OHLC columns for both sides.
    Complete bid/ask data takes precedence over ``spread``, which takes
    precedence over the scalar cost-model fallback in the backtester.
    """
    if SPREAD_COLUMN in result.columns:
        values = pd.to_numeric(result[SPREAD_COLUMN], errors="coerce")
        original = result[SPREAD_COLUMN]
        missing_mask = original.isna()
        invalid_numeric_mask = values.isna() & ~missing_mask
        non_finite_mask = values.notna() & ~np.isfinite(values.to_numpy())
        negative_mask = values.notna() & (values < 0)
        if missing_mask.any():
            issues.append(DataValidationIssue(
                "missing_spread",
                "error",
                "spread contains missing values",
                _affected_timestamps(result["timestamp"], missing_mask),
            ))
        if invalid_numeric_mask.any():
            issues.append(DataValidationIssue(
                "invalid_numeric_spread",
                "error",
                "spread must contain numeric price widths",
                _affected_timestamps(result["timestamp"], invalid_numeric_mask),
            ))
        if non_finite_mask.any():
            issues.append(DataValidationIssue(
                "non_finite_spread",
                "error",
                "spread must be finite",
                _affected_timestamps(result["timestamp"], non_finite_mask),
            ))
        if negative_mask.any():
            issues.append(DataValidationIssue(
                "negative_spread",
                "error",
                "spread must be non-negative",
                _affected_timestamps(result["timestamp"], negative_mask),
            ))
        result[SPREAD_COLUMN] = values

    supplied_bid_ask = [column for column in BID_ASK_COLUMNS if column in result.columns]
    if not supplied_bid_ask:
        return
    missing_bid_ask = [column for column in BID_ASK_COLUMNS if column not in result.columns]
    if missing_bid_ask:
        issues.append(DataValidationIssue(
            "incomplete_bid_ask",
            "error",
            "bid/ask execution data must include all bid_open/high/low/close and ask_open/high/low/close columns",
        ))
        return

    values = result[list(BID_ASK_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    for column in BID_ASK_COLUMNS:
        original = result[column]
        missing_mask = original.isna()
        invalid_numeric_mask = values[column].isna() & ~missing_mask
        non_finite_mask = values[column].notna() & ~np.isfinite(values[column].to_numpy())
        non_positive_mask = values[column].notna() & (values[column] <= 0)
        if missing_mask.any():
            issues.append(DataValidationIssue(
                "missing_bid_ask_value",
                "error",
                f"{column} contains missing values",
                _affected_timestamps(result["timestamp"], missing_mask),
            ))
        if invalid_numeric_mask.any():
            issues.append(DataValidationIssue(
                "invalid_numeric_bid_ask",
                "error",
                f"{column} must contain numeric prices",
                _affected_timestamps(result["timestamp"], invalid_numeric_mask),
            ))
        if non_finite_mask.any():
            issues.append(DataValidationIssue(
                "non_finite_bid_ask",
                "error",
                f"{column} must be finite",
                _affected_timestamps(result["timestamp"], non_finite_mask),
            ))
        if non_positive_mask.any():
            issues.append(DataValidationIssue(
                "non_positive_bid_ask",
                "error",
                f"{column} must be positive",
                _affected_timestamps(result["timestamp"], non_positive_mask),
            ))
    result.loc[:, list(BID_ASK_COLUMNS)] = values

    valid_values = values.notna().all(axis=1) & np.isfinite(values).all(axis=1)
    if not valid_values.any():
        return
    for side in ("bid", "ask"):
        side_values = values[[f"{side}_{field}" for field in ("open", "high", "low", "close")]]
        invalid_ohlc = (
            (side_values[f"{side}_low"] > side_values[f"{side}_open"])
            | (side_values[f"{side}_low"] > side_values[f"{side}_close"])
            | (side_values[f"{side}_high"] < side_values[f"{side}_open"])
            | (side_values[f"{side}_high"] < side_values[f"{side}_close"])
            | (side_values[f"{side}_low"] > side_values[f"{side}_high"])
        ) & valid_values
        if invalid_ohlc.any():
            issues.append(DataValidationIssue(
                "inconsistent_bid_ask_ohlc",
                "error",
                f"{side} OHLC must satisfy low <= open/close <= high and low <= high",
                _affected_timestamps(result["timestamp"], invalid_ohlc),
            ))
    for field in ("open", "high", "low", "close"):
        crossed = (values[f"bid_{field}"] > values[f"ask_{field}"]) & valid_values
        if crossed.any():
            issues.append(DataValidationIssue(
                "inconsistent_bid_ask",
                "error",
                f"bid_{field} must be less than or equal to ask_{field}",
                _affected_timestamps(result["timestamp"], crossed),
            ))


def _append_gap_issues(
    issues: list[DataValidationIssue],
    timestamps: pd.Series,
    expected_interval: pd.Timedelta,
    config: DataValidationConfig,
) -> None:
    for index in range(1, len(timestamps)):
        previous = timestamps.iloc[index - 1]
        current = timestamps.iloc[index]
        delta = current - previous
        if delta <= expected_interval:
            continue
        missing_count = int(delta // expected_interval) - 1
        if delta % expected_interval != pd.Timedelta(0):
            issues.append(DataValidationIssue(
                "interval_alignment",
                "error",
                f"timestamp gap from {previous.isoformat()} to {current.isoformat()} is not aligned to the requested interval",
                (_canonical_timestamp(previous), _canonical_timestamp(current)),
            ))
            continue
        missing_times = pd.date_range(previous + expected_interval, current - expected_interval, freq=expected_interval, tz="UTC")
        missing_dates = {timestamp.date().isoformat() for timestamp in missing_times}
        expected_dates = set(config.expected_closure_dates)
        expected = all(timestamp.weekday() in config.weekend_days or timestamp.date().isoformat() in expected_dates for timestamp in missing_times)
        if expected:
            code = "expected_market_closure"
            message = f"{missing_count} expected market-closure intervals between {previous.isoformat()} and {current.isoformat()}"
            issues.append(DataValidationIssue("expected_market_closure", "info", message, (_canonical_timestamp(previous), _canonical_timestamp(current))))
        else:
            issues.append(DataValidationIssue(
                "unexpected_gap",
                "error",
                f"{missing_count} unexplained intervals between {previous.isoformat()} and {current.isoformat()}",
                (_canonical_timestamp(previous), _canonical_timestamp(current)),
            ))
            issues.append(DataValidationIssue(
                "calendar_unavailable",
                "error",
                "no holiday calendar/provider metadata was supplied; weekday gaps are conservatively unsafe",
                tuple(sorted(missing_dates)),
            ))


def _affected_values(values: pd.Series, mask: pd.Series) -> tuple[str, ...]:
    return tuple(str(value) for value in values[mask].head(20).tolist())


def _affected_timestamps(timestamps: pd.Series, mask: pd.Series) -> tuple[str, ...]:
    values = []
    for value in timestamps[mask].head(20).tolist():
        if pd.isna(value):
            values.append("<invalid timestamp>")
        else:
            values.append(_canonical_timestamp(value))
    return tuple(values)


def _canonical_timestamp(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _report_message(report: DataValidationReport) -> str:
    errors = [issue.message for issue in report.issues if issue.severity == "error"]
    return "; ".join(errors) if errors else "market data is not safe to use"

