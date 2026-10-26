import copy

import numpy as np
import pandas as pd
import pytest

from engine.backtester import run_backtest
from engine.data import (
    DataValidationConfig,
    DataValidationError,
    validate_market_data,
    validate_market_data_report,
)
from engine.strategy import Strategy


def frame_at(timestamps, values=None):
    values = values or list(range(100, 100 + len(timestamps)))
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": values,
            "high": [value + 1 for value in values],
            "low": [value - 1 for value in values],
            "close": values,
        }
    )


def strategy(exit_rules=None):
    return Strategy.from_dict(
        {
            "id": "data-validation",
            "version": "1.0.0",
            "pair": "EURUSD",
            "timeframe": "1h",
            "created_at": "2024-01-01T00:00:00Z",
            "entry_rules": [{"type": "always"}],
            "exit_rules": exit_rules or [],
            "position_size": 1,
            "metadata": {"side": "long"},
        }
    )


def issue_codes(report):
    return [issue.code for issue in report.issues]


def test_valid_chronological_ohlc_is_safe_and_normalized_to_utc():
    data = frame_at(pd.date_range("2024-01-02", periods=3, freq="h"))

    report = validate_market_data_report(data, "1h")
    normalized = validate_market_data(data, "1h")

    assert report.safe_to_use
    assert not report.issues
    assert report.expected_interval == "0 days 01:00:00"
    assert str(normalized.timestamp.dt.tz) == "UTC"
    assert normalized.timestamp.tolist() == list(pd.date_range("2024-01-02", periods=3, freq="h", tz="UTC"))


def test_duplicate_and_out_of_order_timestamps_are_rejected_without_sorting():
    duplicate = frame_at(pd.to_datetime(["2024-01-01 00:00Z", "2024-01-01 00:00Z", "2024-01-01 01:00Z"]))
    with pytest.raises(DataValidationError) as duplicate_error:
        validate_market_data(duplicate, "1h")
    assert "duplicate_timestamps" in issue_codes(duplicate_error.value.report)

    out_of_order = frame_at(pd.to_datetime(["2024-01-01 01:00Z", "2024-01-01 00:00Z", "2024-01-01 02:00Z"]))
    with pytest.raises(DataValidationError) as order_error:
        validate_market_data(out_of_order, "1h")
    assert "non_monotonic_timestamps" in issue_codes(order_error.value.report)
    assert out_of_order.timestamp.iloc[0] > out_of_order.timestamp.iloc[1]


def test_unexpected_gap_is_reported_as_unsafe_when_calendar_is_unavailable():
    data = frame_at(pd.to_datetime(["2024-01-02 00:00Z", "2024-01-02 01:00Z", "2024-01-02 03:00Z"]))

    report = validate_market_data_report(data, "1h")

    assert not report.safe_to_use
    assert "unexpected_gap" in issue_codes(report)
    assert "calendar_unavailable" in issue_codes(report)
    gap = next(issue for issue in report.issues if issue.code == "unexpected_gap")
    assert len(gap.affected_timestamps) == 2


def test_weekend_gap_is_expected_and_explicit_closure_dates_are_supported():
    weekend = frame_at(pd.to_datetime(["2024-01-05 23:00Z", "2024-01-08 00:00Z"]))
    weekend_report = validate_market_data_report(weekend, "1h")
    holiday = frame_at(pd.to_datetime(["2024-01-04 23:00Z", "2024-01-05 01:00Z"]))
    holiday_report = validate_market_data_report(
        holiday,
        "1h",
        config=DataValidationConfig(expected_closure_dates=("2024-01-05",)),
    )

    assert weekend_report.safe_to_use
    assert "expected_market_closure" in issue_codes(weekend_report)
    assert holiday_report.safe_to_use
    assert "expected_market_closure" in issue_codes(holiday_report)


@pytest.mark.parametrize(
    "mutator,expected_code",
    [
        (lambda data: data.__setitem__("open", [100, np.nan, 102]), "missing_values"),
        (lambda data: data.__setitem__("high", [101, np.inf, 103]), "non_finite_price"),
        (lambda data: data.__setitem__("low", [100, -1, 102]), "non_positive_price"),
        (lambda data: data.__setitem__("close", [100, "not-a-price", 102]), "invalid_numeric_price"),
    ],
)
def test_missing_invalid_and_non_finite_prices_are_rejected(mutator, expected_code):
    data = frame_at(pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    mutator(data)

    report = validate_market_data_report(data, "1h")

    assert not report.safe_to_use
    assert expected_code in issue_codes(report)


def test_inconsistent_ohlc_relationships_are_rejected():
    data = frame_at(pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    data.loc[1, "low"] = data.loc[1, "open"] + 1
    data.loc[2, "high"] = data.loc[2, "close"] - 1

    report = validate_market_data_report(data, "1h")

    assert not report.safe_to_use
    assert "inconsistent_ohlc" in issue_codes(report)


def test_missing_columns_empty_data_invalid_timeframe_and_invalid_timestamp_fail():
    missing = pd.DataFrame({"timestamp": [], "open": [], "high": [], "low": []})
    with pytest.raises(DataValidationError) as missing_error:
        validate_market_data(missing, "1h")
    assert "missing_required_columns" in issue_codes(missing_error.value.report)

    empty = frame_at(pd.DatetimeIndex([], tz="UTC"))
    empty_report = validate_market_data_report(empty, "1h")
    assert not empty_report.safe_to_use
    assert "empty_dataset" in issue_codes(empty_report)

    invalid_timeframe = validate_market_data_report(frame_at(pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")), "2h")
    assert not invalid_timeframe.safe_to_use
    assert "unsupported_timeframe" in issue_codes(invalid_timeframe)

    invalid_timestamp = frame_at(["2024-01-01T00:00:00Z", "not-a-timestamp", "2024-01-01T02:00:00Z"])
    invalid_timestamp_report = validate_market_data_report(invalid_timestamp, "1h")
    assert not invalid_timestamp_report.safe_to_use
    assert "invalid_timestamp" in issue_codes(invalid_timestamp_report)


def test_backtester_rejects_history_too_short_before_any_trade_executes():
    data = frame_at(pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"))

    with pytest.raises(DataValidationError) as error:
        run_backtest(strategy(), data, 1000)

    assert "insufficient_history" in issue_codes(error.value.report)


def test_bar_exit_requires_history_for_the_requested_exit_rule():
    data = frame_at(pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))

    with pytest.raises(DataValidationError) as error:
        run_backtest(strategy(exit_rules=[{"type": "bars", "value": 1}]), data, 1000)

    assert "insufficient_history" in issue_codes(error.value.report)


def test_validation_report_is_deterministic_and_input_is_unchanged():
    data = frame_at(pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    original = copy.deepcopy(data)

    first = validate_market_data_report(data, "1h")
    second = validate_market_data_report(data, "1h")

    assert first.issues == second.issues
    assert first.safe_to_use == second.safe_to_use
    pd.testing.assert_frame_equal(first.normalized_frame, second.normalized_frame)
    pd.testing.assert_frame_equal(data, original)
