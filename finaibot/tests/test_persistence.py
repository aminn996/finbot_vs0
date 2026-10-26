import json

import pandas as pd
import pytest

from engine.backtester import run_backtest
from engine.costs import CostModel
from engine.persistence import (
    BacktestInterrupted,
    CHECKPOINT_VERSION,
    RunPersistenceConfig,
    RunPersistenceError,
    canonical_record_digest,
    load_run_record,
)
from engine.risk import RiskConfig
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
        "id": "persistence-regression",
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


def _without_run_ids(events):
    return [{key: value for key, value in event.items() if key != "run_id"} for event in events]


def _interrupted_checkpoint(tmp_path):
    config = RunPersistenceConfig(tmp_path, interrupt_after_bars=2)
    with pytest.raises(BacktestInterrupted) as caught:
        run_backtest(strategy(), bars([100, 100, 100, 100, 105]), 1000, persistence=config)
    run_id = caught.value.run_id
    return RunPersistenceConfig(tmp_path), run_id, load_run_record(RunPersistenceConfig(tmp_path), run_id)


def _rewrite_checkpoint(config, run_id, record, *, update_digest):
    if update_digest:
        record["integrity_digest"] = canonical_record_digest(record)
    config.run_path(run_id).write_text(json.dumps(record, sort_keys=True), encoding="utf-8")


def test_completed_run_record_is_atomic_json_and_contains_final_state(tmp_path):
    config = RunPersistenceConfig(tmp_path)
    result = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000, persistence=config)

    record = load_run_record(config, result.metadata["run_id"])
    assert record["status"] == "COMPLETED"
    assert record["checkpoint_version"] == CHECKPOINT_VERSION
    assert record["final"]["status"] == "COMPLETED"
    assert record["state"]["trades"]
    assert json.loads(json.dumps(record, sort_keys=True)) == record


def test_interrupted_run_resumes_without_duplicate_state_or_events(tmp_path):
    data = bars([100, 100, 100, 100, 105, 105])
    interrupted_config = RunPersistenceConfig(tmp_path, interrupt_after_bars=2)

    with pytest.raises(BacktestInterrupted) as caught:
        run_backtest(strategy(), data, 1000, persistence=interrupted_config)

    run_id = caught.value.run_id
    checkpoint = load_run_record(RunPersistenceConfig(tmp_path), run_id)
    assert checkpoint["status"] == "INTERRUPTED"
    assert checkpoint["progress"]["next_bar_index"] == 2

    resumed = run_backtest(strategy(), data, 1000, persistence=RunPersistenceConfig(tmp_path), resume_run_id=run_id)
    uninterrupted = run_backtest(strategy(), data, 1000)

    assert resumed.metadata["run_id"] == run_id
    assert resumed.trades == uninterrupted.trades
    pd.testing.assert_frame_equal(resumed.equity_curve, uninterrupted.equity_curve)
    assert _without_run_ids(resumed.events) == _without_run_ids(uninterrupted.events)
    assert resumed.metadata["input_hash"] == uninterrupted.metadata["input_hash"]


@pytest.mark.parametrize("change", ["strategy", "data", "costs"])
def test_resume_rejects_changed_inputs(tmp_path, change):
    data = bars([100, 100, 100, 100, 105])
    config = RunPersistenceConfig(tmp_path, interrupt_after_bars=2)
    with pytest.raises(BacktestInterrupted) as caught:
        run_backtest(strategy(), data, 1000, persistence=config)

    changed_strategy = strategy(position_size=2) if change == "strategy" else strategy()
    changed_data = bars([100, 100, 100, 100, 106]) if change == "data" else data
    changed_costs = CostModel(commission=1) if change == "costs" else CostModel()
    with pytest.raises(RunPersistenceError, match="incompatible"):
        run_backtest(
            changed_strategy,
            changed_data,
            1000,
            costs=changed_costs,
            persistence=RunPersistenceConfig(tmp_path),
            resume_run_id=caught.value.run_id,
        )


def test_corrupted_checkpoint_is_rejected(tmp_path):
    config = RunPersistenceConfig(tmp_path, interrupt_after_bars=2)
    with pytest.raises(BacktestInterrupted) as caught:
        run_backtest(strategy(), bars([100, 100, 100, 100, 105]), 1000, persistence=config)

    config.run_path(caught.value.run_id).write_text("{not-json", encoding="utf-8")
    with pytest.raises(RunPersistenceError, match="could not load"):
        run_backtest(strategy(), bars([100, 100, 100, 100, 105]), 1000, persistence=RunPersistenceConfig(tmp_path), resume_run_id=caught.value.run_id)


def test_completed_and_hard_rejected_runs_cannot_resume(tmp_path):
    config = RunPersistenceConfig(tmp_path)
    completed = run_backtest(strategy(), bars([100, 100, 100, 105]), 1000, persistence=config)
    with pytest.raises(RunPersistenceError, match="cannot be resumed"):
        run_backtest(strategy(), bars([100, 100, 100, 105]), 1000, persistence=config, resume_run_id=completed.metadata["run_id"])

    rejected = run_backtest(
        strategy(),
        bars([100, 100, 100, 89]),
        100,
        risk=RiskConfig(total_preventive_pct=0.01, total_hard_pct=0.02),
        persistence=config,
    )
    assert rejected.status == "REJECTED"
    rejected_record = load_run_record(config, rejected.metadata["run_id"])
    assert rejected_record["status"] == "REJECTED"
    assert rejected_record["final"]["status"] == "REJECTED"
    with pytest.raises(RunPersistenceError, match="cannot be resumed"):
        run_backtest(
            strategy(),
            bars([100, 100, 100, 89]),
            100,
            risk=RiskConfig(total_preventive_pct=0.01, total_hard_pct=0.02),
            persistence=config,
            resume_run_id=rejected.metadata["run_id"],
        )


def test_atomic_write_failure_does_not_return_a_result(tmp_path, monkeypatch):
    config = RunPersistenceConfig(tmp_path)

    def fail_replace(*args, **kwargs):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr("engine.persistence.os.replace", fail_replace)
    with pytest.raises(RunPersistenceError, match="atomically write"):
        run_backtest(strategy(), bars([100, 100, 100, 105]), 1000, persistence=config)


@pytest.mark.parametrize("next_bar_index", [-1, 999])
def test_resume_rejects_impossible_progress_before_simulation(tmp_path, next_bar_index):
    config, run_id, record = _interrupted_checkpoint(tmp_path)
    record["progress"]["next_bar_index"] = next_bar_index
    _rewrite_checkpoint(config, run_id, record, update_digest=True)

    with pytest.raises(RunPersistenceError, match="next_bar_index"):
        run_backtest(strategy(), bars([100, 100, 100, 100, 105]), 1000, persistence=config, resume_run_id=run_id)


def test_resume_rejects_payload_change_without_matching_digest(tmp_path):
    config, run_id, record = _interrupted_checkpoint(tmp_path)
    record["progress"]["next_bar_index"] = 999
    _rewrite_checkpoint(config, run_id, record, update_digest=False)

    with pytest.raises(RunPersistenceError, match="integrity digest"):
        run_backtest(strategy(), bars([100, 100, 100, 100, 105]), 1000, persistence=config, resume_run_id=run_id)


def test_resume_rejects_inconsistent_saved_state(tmp_path):
    config, run_id, record = _interrupted_checkpoint(tmp_path)
    record["state"]["equity_rows"] = []
    _rewrite_checkpoint(config, run_id, record, update_digest=True)

    with pytest.raises(RunPersistenceError, match="equity history"):
        run_backtest(strategy(), bars([100, 100, 100, 100, 105]), 1000, persistence=config, resume_run_id=run_id)


def test_resume_rejects_checkpoint_without_explicit_equity_fields(tmp_path):
    config, run_id, record = _interrupted_checkpoint(tmp_path)
    del record["state"]["equity_rows"][0]["realized_pnl"]
    _rewrite_checkpoint(config, run_id, record, update_digest=True)

    with pytest.raises(RunPersistenceError, match="equity row"):
        run_backtest(strategy(), bars([100, 100, 100, 100, 105]), 1000, persistence=config, resume_run_id=run_id)


def test_resume_rejects_previous_checkpoint_format_version(tmp_path):
    config, run_id, record = _interrupted_checkpoint(tmp_path)
    record["checkpoint_version"] = 1
    _rewrite_checkpoint(config, run_id, record, update_digest=True)

    with pytest.raises(RunPersistenceError, match="unsupported"):
        run_backtest(strategy(), bars([100, 100, 100, 100, 105]), 1000, persistence=config, resume_run_id=run_id)


@pytest.mark.parametrize("field", ["created_at", "initial_capital", "checkpoint_version"])
def test_resume_rejects_digest_valid_malformed_top_level_fields(tmp_path, field):
    config, run_id, record = _interrupted_checkpoint(tmp_path)
    record[field] = "not-valid" if field != "initial_capital" else "not-a-number"
    _rewrite_checkpoint(config, run_id, record, update_digest=True)

    with pytest.raises(RunPersistenceError):
        run_backtest(strategy(), bars([100, 100, 100, 100, 105]), 1000, persistence=config, resume_run_id=run_id)


@pytest.mark.parametrize("mutation", [
    lambda record: record["state"]["events"][0].update(sequence=2),
    lambda record: record["state"]["events"][0].update(run_id="other-run"),
])
def test_resume_rejects_inconsistent_event_ledger(tmp_path, mutation):
    config, run_id, record = _interrupted_checkpoint(tmp_path)
    mutation(record)
    _rewrite_checkpoint(config, run_id, record, update_digest=True)

    with pytest.raises(RunPersistenceError, match="event ledger"):
        run_backtest(strategy(), bars([100, 100, 100, 100, 105]), 1000, persistence=config, resume_run_id=run_id)

