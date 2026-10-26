"""Small deterministic OHLC backtester for M1."""

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from math import isfinite, isclose
import hashlib
import json
import platform
from collections.abc import Mapping
from numbers import Real
from typing import Any
import uuid
import pandas as pd
from .costs import CostBreakdown, CostModel
from .data import BID_ASK_COLUMNS, DataValidationConfig, DataValidationError, SPREAD_COLUMN, validate_market_data
from .events import BacktestEvent, EventLedger
from .execution import ExecutionConfig
from .persistence import (
    BacktestInterrupted,
    CHECKPOINT_VERSION,
    RunPersistenceConfig,
    RunPersistenceError,
    load_run_record,
    write_run_record,
)
from .risk import MarginConfig, RiskConfig, RiskMonitor
from .strategy import Strategy, StrategyValidationError


ENGINE_VERSION = "m1-backtester-2"
_MARKET_COLUMNS = ("timestamp", "open", "high", "low", "close")
_EXECUTION_SETTINGS = {
    "entry": "next_bar_open",
    "valuation": "completed_candle_close",
    "protective_exit_ambiguity": "stop_loss_first",
    "gap_fill": "candle_open",
    "forced_close": "last_completed_close",
    "price_model": "bid_ask_if_supplied_else_per_bar_spread_else_midpoint_with_directional_spread_and_slippage",
    "spread_precedence": "complete_bid_ask_ohlc > per_bar_spread > scalar_cost_model_spread",
    "bid_ask_columns": list(BID_ASK_COLUMNS),
    "per_bar_spread_column": SPREAD_COLUMN,
    "bid_ask_protective_fill": "configured_trigger_is_executable_side_price_when_exact_quote_at_level_is_unavailable",
    "fill_model": "entry_quantity_cap_with_cancelled_remainder; exits_full_fill",
}


@dataclass(frozen=True)
class Trade:
    entry_time: object
    exit_time: object
    side: str
    quantity: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    exit_reason: str
    cost_breakdown: CostBreakdown = CostBreakdown()


@dataclass(frozen=True)
class BacktestResult:
    trades: tuple[Trade, ...]
    equity_curve: pd.DataFrame
    events: tuple[dict[str, Any], ...]
    data_hash: str | None
    status: str = "COMPLETED"
    metadata: dict[str, Any] = field(default_factory=dict)


def run_backtest(strategy: Strategy, data: pd.DataFrame, initial_capital: float,
                 costs: CostModel = CostModel(), risk: RiskConfig = RiskConfig(), data_hash: str | None = None,
                 margin: MarginConfig = MarginConfig(),
                 data_validation: DataValidationConfig = DataValidationConfig(),
                 execution: ExecutionConfig = ExecutionConfig(),
                 persistence: RunPersistenceConfig = RunPersistenceConfig(),
                 resume_run_id: str | None = None) -> BacktestResult:
    if not isinstance(persistence, RunPersistenceConfig):
        raise ValueError("persistence must be a RunPersistenceConfig")
    if resume_run_id is not None and not persistence.enabled:
        raise RunPersistenceError("resume_run_id requires a persistence directory")
    resuming = resume_run_id is not None
    run_id = resume_run_id if resuming else str(uuid.uuid4())
    created_at = None if resuming else datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    ledger = EventLedger(run_id)
    if not resuming:
        ledger.emit(
            "backtest_started",
            phase="initialization",
            message="backtest run started",
        )
    if not isinstance(strategy, Strategy):
        raise ValueError("strategy must be a validated Strategy")
    if not isinstance(execution, ExecutionConfig):
        error = ValueError("execution must be an ExecutionConfig")
        _attach_failure_events(ledger, error, "calculation_error", "execution configuration validation failed")
        raise error
    # Revalidate the serialized definition at the execution boundary.  This
    # protects the backtester from callers that instantiate the frozen
    # dataclass directly and bypass ``Strategy.from_dict``.
    try:
        strategy = Strategy.from_dict(strategy.to_dict())
    except StrategyValidationError as exc:
        _attach_failure_events(ledger, exc, "strategy_validation_failed", "strategy validation failed")
        raise
    if isinstance(initial_capital, bool):
        raise ValueError("initial_capital must be a finite, positive number")
    try:
        initial_capital = float(initial_capital)
    except (TypeError, ValueError, OverflowError) as exc:
        _attach_failure_events(ledger, exc, "calculation_error", "initial capital validation failed")
        raise ValueError("initial_capital must be a finite, positive number") from exc
    if not isfinite(initial_capital) or initial_capital <= 0:
        error = ValueError("initial_capital must be a finite, positive number")
        _attach_failure_events(ledger, error, "calculation_error", "initial capital validation failed")
        raise error
    ledger.emit("strategy_validated", phase="validation", message="strategy validated for M1 execution", details={"strategy_id": strategy.id, "strategy_version": strategy.version})
    try:
        frame = validate_market_data(
            data,
            timeframe=strategy.timeframe,
            config=data_validation,
            minimum_rows=_minimum_history_for_strategy(strategy),
        )
    except DataValidationError as exc:
        _attach_failure_events(ledger, exc, "data_validation_failed", "market data validation failed")
        raise
    ledger.emit("data_validated", phase="validation", message="market data validated for simulation", details={"rows": len(frame), "timeframe": strategy.timeframe})
    dataset_hash = _hash_market_data(frame)
    strategy_hash = _hash_strategy(strategy)
    configuration = _configuration_payload(costs, margin, risk, data_validation, execution)
    configuration_hash = _hash_payload(configuration)
    input_hash = _hash_payload({
        "strategy_hash": strategy_hash,
        "dataset_hash": dataset_hash,
        "initial_capital": initial_capital,
        "configuration_hash": configuration_hash,
        "execution": _EXECUTION_SETTINGS,
        "engine_version": ENGINE_VERSION,
        "software": _software_versions(),
    })
    if resuming:
        record = load_run_record(persistence, run_id)
        if record["status"] not in {"RUNNING", "INTERRUPTED"}:
            raise RunPersistenceError("completed or rejected runs cannot be resumed")
        _validate_resume_record(
            record,
            frame=frame,
            strategy=strategy,
            strategy_hash=strategy_hash,
            dataset_hash=dataset_hash,
            data_reference=data_hash,
            initial_capital=initial_capital,
            configuration=configuration,
            configuration_hash=configuration_hash,
            input_hash=input_hash,
        )
        created_at = record["created_at"]
        ledger = EventLedger.from_dicts(record["state"]["events"], run_id)
        monitor, trades, equity_rows, margin_event_keys, position, pending_entry, margin_blocked = _restore_checkpoint_state(record, initial_capital, risk)
        start_index = int(record["progress"]["next_bar_index"])
    else:
        monitor = RiskMonitor(initial_capital, risk)
        trades = []
        equity_rows = []
        margin_event_keys = set()
        position = None
        pending_entry = False
        margin_blocked = False
        start_index = 0
        if persistence.enabled:
            _write_checkpoint(
                persistence,
                run_id=run_id,
                created_at=created_at,
                status="RUNNING",
                next_bar_index=0,
                strategy=strategy,
                strategy_hash=strategy_hash,
                dataset_hash=dataset_hash,
                data_reference=data_hash,
                initial_capital=initial_capital,
                configuration=configuration,
                configuration_hash=configuration_hash,
                input_hash=input_hash,
                ledger=ledger,
                monitor=monitor,
                trades=trades,
                equity_rows=equity_rows,
                margin_event_keys=margin_event_keys,
                position=position,
                pending_entry=pending_entry,
                margin_blocked=margin_blocked,
                persistence_config=persistence,
            )
    for i in range(start_index, len(frame)):
        row = frame.iloc[i]
        ts = row.timestamp
        prior_equity = equity_rows[-1]["equity"] if equity_rows else initial_capital
        monitor.begin_period(ts, prior_equity)
        cash = initial_capital + sum(t.net_pnl for t in trades)
        entry_accepted = False
        if position is not None:
            if _should_exit(position, row, strategy, i):
                exit_mid, reason = _exit_price(position, row, strategy, i)
                exit_price, exit_costs = _fill(
                    position["side"], "exit", exit_mid, position["quantity"], costs,
                    include_trade_costs=True, row=row,
                    price_field="close" if reason == "rule_exit" else ("open" if _protective_exit_gapped(position, row, reason) else "trigger"),
                )
                gross = _pnl(position["side"], position["entry_mid"], exit_mid, position["quantity"])
                breakdown = position["entry_costs"] + exit_costs
                trade_id = position["trade_id"]
                trades.append(Trade(position["entry_time"], ts, position["side"], position["quantity"], position["entry_price"], exit_price, gross, breakdown.total, gross - breakdown.total, reason, breakdown))
                exit_event_type = f"{reason}_executed"
                ledger.emit(
                    exit_event_type,
                    timestamp=ts,
                    kind=reason,
                    phase="execution",
                    message=f"{reason.replace('_', ' ')} executed",
                    trade_id=trade_id,
                    details={"mid_price": exit_mid, "execution_price": exit_price},
                )
                ledger.emit(
                    "transaction_costs_applied",
                    timestamp=ts,
                    kind="exit_costs",
                    phase="costs",
                    message="exit transaction costs applied",
                    trade_id=trade_id,
                    details=_cost_details(exit_costs),
                )
                ledger.emit(
                    "position_closed",
                    timestamp=ts,
                    phase="position",
                    message="position closed",
                    trade_id=trade_id,
                    details={"reason": reason, "gross_pnl": gross, "net_pnl": gross - breakdown.total},
                )
                position = None
                cash = initial_capital + sum(t.net_pnl for t in trades)
                equity = cash
                _check_risk(monitor, ledger, ts, equity, "exit")
            else:
                equity = _open_equity(position, _mid_price(row, "close"), cash, costs, row=row)
                _check_risk(monitor, ledger, ts, equity, "candle_close")
        else:
            equity = cash
            _check_risk(monitor, ledger, ts, equity, "bar_close")
        if position is None and pending_entry and monitor.entries_allowed(ts) and not margin_blocked:
            side = str(strategy.metadata.get("side", "long"))
            requested_quantity = strategy.position_size
            filled_quantity, remaining_quantity, fill_status = execution.fill_quantity(requested_quantity)
            entry_mid = _mid_price(row, "open")
            entry_price, entry_costs = _fill(side, "entry", entry_mid, filled_quantity, costs, row=row, price_field="open")
            notional = abs(entry_mid * filled_quantity)
            required_margin = notional / margin.leverage
            projected_free_margin = equity - entry_costs.total - required_margin
            rejected_reason = None
            if margin.max_exposure is not None and notional > margin.max_exposure:
                rejected_reason = "exposure_limit"
            elif projected_free_margin < 0:
                rejected_reason = "insufficient_margin"
            if rejected_reason is None:
                trade_id = f"trade-{len(trades) + 1}"
                position = {"side": side, "entry_time": ts, "entry_index": i, "entry_mid": entry_mid, "entry_price": entry_price, "entry_costs": entry_costs, "quantity": filled_quantity, "notional": notional, "required_margin": required_margin, "trade_id": trade_id}
                if strategy.stop_loss is not None:
                    position["stop"] = entry_mid - strategy.stop_loss if side == "long" else entry_mid + strategy.stop_loss
                if strategy.take_profit is not None:
                    position["take"] = entry_mid + strategy.take_profit if side == "long" else entry_mid - strategy.take_profit
                pending_entry = False
                entry_accepted = True
                ledger.emit(
                    "entry_accepted" if fill_status == "filled" else "entry_partially_filled",
                    timestamp=ts,
                    phase="entry",
                    message="entry accepted and fully filled" if fill_status == "filled" else "entry partially filled; remaining quantity cancelled",
                    trade_id=trade_id,
                    details={"side": side, "requested_quantity": requested_quantity, "filled_quantity": filled_quantity, "remaining_quantity": remaining_quantity, "fill_status": fill_status, "mid_price": entry_mid, "execution_price": entry_price},
                )
                if fill_status == "partially_filled":
                    ledger.emit(
                        "remaining_quantity_cancelled",
                        timestamp=ts,
                        kind="cancelled_remainder",
                        phase="entry",
                        message="unfilled entry quantity cancelled by execution policy",
                        trade_id=trade_id,
                        details={"requested_quantity": requested_quantity, "filled_quantity": filled_quantity, "remaining_quantity": remaining_quantity, "policy": execution.remaining_quantity_policy},
                    )
                ledger.emit(
                    "transaction_costs_applied",
                    timestamp=ts,
                    kind="entry_costs",
                    phase="costs",
                    message="entry transaction costs applied",
                    trade_id=trade_id,
                    details=_cost_details(entry_costs),
                )
                ledger.emit(
                    "position_opened",
                    timestamp=ts,
                    phase="position",
                    message="position opened",
                    trade_id=trade_id,
                    details={"side": side, "quantity": filled_quantity, "requested_quantity": requested_quantity, "entry_price": entry_price, "fill_status": fill_status},
                )
                _check_risk(monitor, ledger, ts, _open_equity(position, entry_mid, equity, costs, row=row), "entry")
            else:
                _record_margin_event(ledger, margin_event_keys, ts, rejected_reason, f"entry rejected: {rejected_reason}", required_margin, equity - entry_costs.total, notional, requested_quantity)
        if position is not None:
            equity = _open_equity(position, _mid_price(row, "close"), cash, costs, row=row)
            if entry_accepted:
                _check_risk(monitor, ledger, ts, equity, "candle_close")
        else:
            equity = cash
        components = _equity_components(
            trades, position, _mid_price(row, "close") if position is not None else None,
            cash, costs, row=row,
        )
        equity = components["equity"]
        if position is None and _rules_match(strategy.entry_rules, frame, i):
            ledger.emit(
                "entry_signal",
                timestamp=ts,
                phase="signal",
                message="entry rules matched; entry scheduled for next bar open",
            details={"bar_index": int(i), "requested_quantity": strategy.position_size},
            )
            if monitor.entries_allowed(ts) and not margin_blocked:
                pending_entry = True
            else:
                reason = "risk_blocked" if not monitor.entries_allowed(ts) else "margin_blocked"
                ledger.emit(
                    "entry_rejected",
                    timestamp=ts,
                    kind=reason,
                    severity="warning",
                    phase="entry",
                    message=f"entry rejected: {reason}",
                    details={"bar_index": int(i)},
                )
        margin_state = _margin_state(position, equity)
        if position is not None and margin_state["free_margin"] < 0:
            _record_margin_event(ledger, margin_event_keys, ts, "margin_breach", "free margin became negative; new entries remain blocked", margin_state["required_margin"], margin_state["free_margin"], margin_state["exposure"])
            margin_blocked = True
        equity_rows.append({"timestamp": ts, **components, "daily_loss": max(0.0, monitor._day_start.get(ts.date(), equity) - equity), **margin_state})
        if persistence.enabled:
            interrupt_requested = (
                persistence.interrupt_after_bars is not None
                and start_index < persistence.interrupt_after_bars <= i + 1
                and i + 1 < len(frame)
                and not monitor.hard_breach
            )
            if (i + 1) % persistence.checkpoint_every_bars == 0 or i + 1 == len(frame) or interrupt_requested:
                checkpoint_path = _write_checkpoint(
                    persistence,
                    run_id=run_id,
                    created_at=created_at,
                    status="INTERRUPTED" if interrupt_requested else "RUNNING",
                    next_bar_index=i + 1,
                    strategy=strategy,
                    strategy_hash=strategy_hash,
                    dataset_hash=dataset_hash,
                    data_reference=data_hash,
                    initial_capital=initial_capital,
                    configuration=configuration,
                    configuration_hash=configuration_hash,
                    input_hash=input_hash,
                    ledger=ledger,
                    monitor=monitor,
                    trades=trades,
                    equity_rows=equity_rows,
                    margin_event_keys=margin_event_keys,
                    position=position,
                    pending_entry=pending_entry,
                    margin_blocked=margin_blocked,
                    persistence_config=persistence,
                )
                if interrupt_requested:
                    raise BacktestInterrupted(run_id, checkpoint_path)
        if monitor.hard_breach:
            break
    if position is not None and not monitor.hard_breach:
        row = frame.iloc[-1]
        exit_mid = _mid_price(row, "close")
        exit_price, exit_costs = _fill(
            position["side"], "exit", exit_mid, position["quantity"], costs,
            include_trade_costs=True, row=row, price_field="close",
        )
        gross = _pnl(position["side"], position["entry_mid"], exit_mid, position["quantity"])
        breakdown = position["entry_costs"] + exit_costs
        trade_id = position["trade_id"]
        trades.append(Trade(position["entry_time"], row.timestamp, position["side"], position["quantity"], position["entry_price"], exit_price, gross, breakdown.total, gross - breakdown.total, "end_of_data", breakdown))
        ledger.emit(
            "forced_end_of_data_close",
            timestamp=row.timestamp,
            kind="end_of_data",
            phase="execution",
            message="open position forcibly closed at end of data",
            trade_id=trade_id,
            details={"mid_price": exit_mid, "execution_price": exit_price},
        )
        ledger.emit(
            "transaction_costs_applied",
            timestamp=row.timestamp,
            kind="exit_costs",
            phase="costs",
            message="forced-closure transaction costs applied",
            trade_id=trade_id,
            details=_cost_details(exit_costs),
        )
        ledger.emit(
            "position_closed",
            timestamp=row.timestamp,
            phase="position",
            message="position closed by end-of-data handling",
            trade_id=trade_id,
            details={"reason": "end_of_data", "gross_pnl": gross, "net_pnl": gross - breakdown.total},
        )
        final_cash = initial_capital + sum(t.net_pnl for t in trades)
        if equity_rows:
            final_components = _equity_components(trades, None, None, final_cash, costs)
            equity_rows[-1].update(final_components)
            day_start = monitor._day_start.get(row.timestamp.date(), final_cash)
            equity_rows[-1]["daily_loss"] = max(0.0, day_start - final_cash)
            equity_rows[-1].update(_margin_state(None, final_cash))
        position = None
    equity_curve = _add_drawdown_columns(_equity_frame(equity_rows), initial_capital)
    final_equity = float(equity_curve.iloc[-1]["equity"]) if not equity_curve.empty else initial_capital
    status = "REJECTED" if monitor.hard_breach else "COMPLETED"
    ledger.emit(
        "backtest_rejected" if monitor.hard_breach else "backtest_completed",
        phase="terminal",
        severity="error" if monitor.hard_breach else "info",
        message="backtest rejected" if monitor.hard_breach else "backtest completed",
        status=status,
        equity=final_equity,
        details={"trade_count": len(trades), "equity": final_equity},
    )
    metadata = _run_metadata(
        run_id=run_id,
        created_at=created_at,
        status=status,
        strategy=strategy,
        strategy_hash=strategy_hash,
        dataset_hash=dataset_hash,
        initial_capital=initial_capital,
        configuration=configuration,
        configuration_hash=configuration_hash,
        input_hash=input_hash,
        data_reference=data_hash,
        persistence=persistence,
    )
    result_data_hash = data_hash if data_hash is not None else dataset_hash
    result = BacktestResult(tuple(trades), equity_curve, ledger.to_dicts(), result_data_hash, status, metadata)
    if persistence.enabled:
        _write_terminal_record(
            persistence,
            run_id=run_id,
            created_at=created_at,
            status=status,
            strategy=strategy,
            strategy_hash=strategy_hash,
            dataset_hash=dataset_hash,
            data_reference=data_hash,
            initial_capital=initial_capital,
            configuration=configuration,
            configuration_hash=configuration_hash,
            input_hash=input_hash,
            ledger=ledger,
            monitor=monitor,
            trades=trades,
            equity_rows=equity_rows,
            equity_curve=equity_curve,
            margin_event_keys=margin_event_keys,
            position=position,
            pending_entry=pending_entry,
            margin_blocked=margin_blocked,
            persistence_config=persistence,
            result=result,
        )
    return result


def _write_checkpoint(
    persistence: RunPersistenceConfig,
    *,
    run_id: str,
    created_at: str,
    status: str,
    next_bar_index: int,
    strategy: Strategy,
    strategy_hash: str,
    dataset_hash: str,
    data_reference: str | None,
    initial_capital: float,
    configuration: dict[str, Any],
    configuration_hash: str,
    input_hash: str,
    ledger: EventLedger,
    monitor: RiskMonitor,
    trades: list[Trade],
    equity_rows: list[dict[str, Any]],
    margin_event_keys: set[tuple[object, str]],
    position: dict[str, Any] | None,
    pending_entry: bool,
    margin_blocked: bool,
    persistence_config: RunPersistenceConfig,
    final: dict[str, Any] | None = None,
) -> Any:
    record = _checkpoint_record(
        run_id=run_id,
        created_at=created_at,
        status=status,
        next_bar_index=next_bar_index,
        strategy=strategy,
        strategy_hash=strategy_hash,
        dataset_hash=dataset_hash,
        data_reference=data_reference,
        initial_capital=initial_capital,
        configuration=configuration,
        configuration_hash=configuration_hash,
        input_hash=input_hash,
        ledger=ledger,
        monitor=monitor,
        trades=trades,
        equity_rows=equity_rows,
        margin_event_keys=margin_event_keys,
        position=position,
        pending_entry=pending_entry,
        margin_blocked=margin_blocked,
        persistence_config=persistence_config,
        final=final,
    )
    return write_run_record(persistence, run_id, record)


def _write_terminal_record(
    persistence: RunPersistenceConfig,
    *,
    run_id: str,
    created_at: str,
    status: str,
    strategy: Strategy,
    strategy_hash: str,
    dataset_hash: str,
    data_reference: str | None,
    initial_capital: float,
    configuration: dict[str, Any],
    configuration_hash: str,
    input_hash: str,
    ledger: EventLedger,
    monitor: RiskMonitor,
    trades: list[Trade],
    equity_rows: list[dict[str, Any]],
    equity_curve: pd.DataFrame,
    margin_event_keys: set[tuple[object, str]],
    position: dict[str, Any] | None,
    pending_entry: bool,
    margin_blocked: bool,
    persistence_config: RunPersistenceConfig,
    result: BacktestResult,
) -> Any:
    final = {
        "status": result.status,
        "data_hash": result.data_hash,
        "trades": [_trade_record(trade) for trade in result.trades],
        "equity_curve": _serializable_equity_curve(equity_curve),
        "events": result.events,
        "metadata": result.metadata,
    }
    return _write_checkpoint(
        persistence,
        run_id=run_id,
        created_at=created_at,
        status=status,
        next_bar_index=len(equity_rows),
        strategy=strategy,
        strategy_hash=strategy_hash,
        dataset_hash=dataset_hash,
        data_reference=data_reference,
        initial_capital=initial_capital,
        configuration=configuration,
        configuration_hash=configuration_hash,
        input_hash=input_hash,
        ledger=ledger,
        monitor=monitor,
        trades=trades,
        equity_rows=equity_rows,
        margin_event_keys=margin_event_keys,
        position=position,
        pending_entry=pending_entry,
        margin_blocked=margin_blocked,
        persistence_config=persistence_config,
        final=final,
    )


def _checkpoint_record(
    *,
    run_id: str,
    created_at: str,
    status: str,
    next_bar_index: int,
    strategy: Strategy,
    strategy_hash: str,
    dataset_hash: str,
    data_reference: str | None,
    initial_capital: float,
    configuration: dict[str, Any],
    configuration_hash: str,
    input_hash: str,
    ledger: EventLedger,
    monitor: RiskMonitor,
    trades: list[Trade],
    equity_rows: list[dict[str, Any]],
    margin_event_keys: set[tuple[object, str]],
    position: dict[str, Any] | None,
    pending_entry: bool,
    margin_blocked: bool,
    persistence_config: RunPersistenceConfig,
    final: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "run_id": run_id,
        "status": status,
        "created_at": created_at,
        "strategy": _canonical_strategy_definition(strategy),
        "strategy_hash": strategy_hash,
        "dataset_hash": dataset_hash,
        "data_reference": data_reference,
        "initial_capital": initial_capital,
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "engine_version": ENGINE_VERSION,
        "software": _software_versions(),
        "input_hash": input_hash,
        "persistence": {
            "checkpoint_every_bars": persistence_config.checkpoint_every_bars,
            "checkpoint_boundary": "after_completed_bar",
        },
        "progress": {
            "next_bar_index": next_bar_index,
            "processed_bars": len(equity_rows),
            "last_timestamp": _canonical_timestamp(equity_rows[-1]["timestamp"]) if equity_rows else None,
        },
        "state": {
            "trades": [_trade_record(trade) for trade in trades],
            "equity_rows": _canonicalize(equity_rows),
            "open_position": _position_record(position),
            "pending_entry": pending_entry,
            "margin_blocked": margin_blocked,
            "margin_event_keys": [[_canonical_timestamp(timestamp), kind] for timestamp, kind in sorted(margin_event_keys, key=lambda item: (_canonical_timestamp(item[0]), item[1]))],
            "risk": _risk_record(monitor),
            "events": ledger.to_dicts(),
        },
        "final": final,
    }


def _trade_record(trade: Trade) -> dict[str, Any]:
    return {
        "entry_time": _canonical_timestamp(trade.entry_time),
        "exit_time": _canonical_timestamp(trade.exit_time),
        "side": trade.side,
        "quantity": trade.quantity,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "gross_pnl": trade.gross_pnl,
        "costs": trade.costs,
        "net_pnl": trade.net_pnl,
        "exit_reason": trade.exit_reason,
        "cost_breakdown": asdict(trade.cost_breakdown),
    }


def _serializable_equity_curve(equity_curve: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in equity_curve.to_dict(orient="records"):
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or (isinstance(value, Real) and not isinstance(value, bool) and not isfinite(float(value))):
                normalized[str(key)] = None
            else:
                normalized[str(key)] = _canonicalize(value)
        records.append(normalized)
    return records


def _equity_frame(equity_rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(equity_rows)
    preferred = [
        "timestamp", "cash", "realized_pnl", "unrealized_pnl", "gross_pnl", "total_costs", "equity",
        "daily_loss", "exposure", "used_margin", "required_margin", "free_margin", "margin_level",
    ]
    ordered = [column for column in preferred if column in frame.columns]
    ordered.extend(column for column in frame.columns if column not in ordered)
    return frame.loc[:, ordered]


def _position_record(position: dict[str, Any] | None) -> dict[str, Any] | None:
    if position is None:
        return None
    record = dict(position)
    record["entry_time"] = _canonical_timestamp(record["entry_time"])
    record["entry_costs"] = asdict(record["entry_costs"])
    return _canonicalize(record)


def _risk_record(monitor: RiskMonitor) -> dict[str, Any]:
    return {
        "day_start": {str(day): value for day, value in monitor._day_start.items()},
        "events": _event_records(monitor.events),
        "hard_breach": monitor.hard_breach,
        "total_block_entries": monitor.total_block_entries,
        "daily_blocked_days": [str(day) for day in sorted(monitor.daily_blocked_days)],
    }


def _event_records(events: list[BacktestEvent]) -> tuple[dict[str, Any], ...]:
    ledger = EventLedger()
    for event in events:
        ledger.add(event)
    return ledger.to_dicts()


def _validate_resume_record(
    record: dict[str, Any],
    *,
    frame: pd.DataFrame,
    strategy: Strategy,
    strategy_hash: str,
    dataset_hash: str,
    data_reference: str | None,
    initial_capital: float,
    configuration: dict[str, Any],
    configuration_hash: str,
    input_hash: str,
) -> None:
    checks = (
        (record.get("strategy_hash"), strategy_hash, "strategy fingerprint"),
        (record.get("dataset_hash"), dataset_hash, "dataset fingerprint"),
        (record.get("data_reference"), data_reference, "data reference"),
        (record.get("initial_capital"), initial_capital, "initial capital"),
        (record.get("configuration_hash"), configuration_hash, "configuration fingerprint"),
        (record.get("input_hash"), input_hash, "input fingerprint"),
        (record.get("engine_version"), ENGINE_VERSION, "engine version"),
        (record.get("software"), _software_versions(), "software versions"),
    )
    for stored, current, label in checks:
        if label == "initial capital":
            try:
                matches = isfinite(float(stored)) and float(stored) == current
            except (TypeError, ValueError, OverflowError):
                matches = False
        else:
            matches = stored == current
        if not matches:
            raise RunPersistenceError(f"checkpoint is incompatible: {label} changed")
    if record.get("strategy") != _canonical_strategy_definition(strategy):
        raise RunPersistenceError("checkpoint is incompatible: strategy definition changed")
    if record.get("configuration") != configuration:
        raise RunPersistenceError("checkpoint is incompatible: effective configuration changed")
    progress = record.get("progress")
    state = record.get("state")
    if not isinstance(progress, Mapping) or not isinstance(state, Mapping):
        raise RunPersistenceError("checkpoint is incomplete: progress/state are required objects")
    next_bar_index = progress.get("next_bar_index")
    if isinstance(next_bar_index, bool) or not isinstance(next_bar_index, int) or next_bar_index < 0:
        raise RunPersistenceError("checkpoint has an invalid next_bar_index")
    if next_bar_index > len(frame):
        raise RunPersistenceError("checkpoint next_bar_index is outside the validated dataset")
    processed_bars = progress.get("processed_bars")
    if isinstance(processed_bars, bool) or not isinstance(processed_bars, int) or processed_bars != next_bar_index:
        raise RunPersistenceError("checkpoint progress is inconsistent with next_bar_index")
    expected_last_timestamp = None if next_bar_index == 0 else _canonical_timestamp(frame.iloc[next_bar_index - 1].timestamp)
    if progress.get("last_timestamp") != expected_last_timestamp:
        raise RunPersistenceError("checkpoint last_timestamp is inconsistent with progress")
    required_state = {"trades", "equity_rows", "open_position", "pending_entry", "margin_blocked", "margin_event_keys", "risk", "events"}
    missing = sorted(required_state - set(state))
    if missing:
        raise RunPersistenceError(f"checkpoint is incomplete; missing state: {', '.join(missing)}")
    _validate_checkpoint_state(record, frame, next_bar_index)


def _validate_checkpoint_state(record: Mapping[str, Any], frame: pd.DataFrame, next_bar_index: int) -> None:
    """Validate persisted state before any checkpoint object is restored."""
    state = record["state"]
    for key in ("trades", "equity_rows", "margin_event_keys", "events"):
        if not isinstance(state[key], (list, tuple)):
            raise RunPersistenceError(f"checkpoint state field {key} must be an array")
    if len(state["equity_rows"]) != next_bar_index:
        raise RunPersistenceError("checkpoint equity history is inconsistent with progress")
    if not isinstance(state["pending_entry"], bool) or not isinstance(state["margin_blocked"], bool):
        raise RunPersistenceError("checkpoint pending or margin state is malformed")
    risk_state = state["risk"]
    if not isinstance(risk_state, Mapping):
        raise RunPersistenceError("checkpoint risk state must be an object")
    _validate_checkpoint_risk_state(risk_state)

    for index, item in enumerate(state["equity_rows"]):
        if not isinstance(item, Mapping):
            raise RunPersistenceError("checkpoint equity rows must be objects")
        required_equity_fields = {
            "timestamp", "cash", "realized_pnl", "unrealized_pnl", "gross_pnl", "total_costs", "equity",
            "daily_loss", "exposure", "used_margin", "required_margin", "free_margin", "margin_level",
        }
        missing_equity_fields = sorted(required_equity_fields - set(item))
        if missing_equity_fields:
            raise RunPersistenceError(f"checkpoint equity row is incomplete; missing: {', '.join(missing_equity_fields)}")
        timestamp = _checkpoint_timestamp(item.get("timestamp"), "equity timestamp")
        if timestamp != _canonical_timestamp(frame.iloc[index].timestamp):
            raise RunPersistenceError("checkpoint equity history does not match the processed dataset prefix")
        for key, value in item.items():
            if key != "timestamp" and value is not None:
                _validate_checkpoint_number(value, f"equity field {key}")
    for item in state["trades"]:
        _validate_checkpoint_trade(item)

    position = state["open_position"]
    if position is not None:
        _validate_checkpoint_position(position, frame, next_bar_index)
    if next_bar_index:
        _validate_checkpoint_margin_state(state["equity_rows"][-1], position)
    for item in state["margin_event_keys"]:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise RunPersistenceError("checkpoint margin event key is malformed")
        _checkpoint_timestamp(item[0], "margin event timestamp")
        if not isinstance(item[1], str) or not item[1]:
            raise RunPersistenceError("checkpoint margin event kind is malformed")
    _validate_checkpoint_events(state["events"], str(record["run_id"]), "event ledger")
    _validate_checkpoint_events(risk_state.get("events", ()), str(record["run_id"]), "risk event ledger", allow_unassigned_run_id=True)
    if next_bar_index and state["equity_rows"][-1].get("timestamp") != record["progress"]["last_timestamp"]:
        raise RunPersistenceError("checkpoint final equity timestamp is inconsistent with progress")
    if position is not None and position.get("entry_index", -1) >= next_bar_index:
        raise RunPersistenceError("checkpoint open position starts after the processed boundary")


def _validate_checkpoint_margin_state(equity_row: Mapping[str, Any], position: Mapping[str, Any] | None) -> None:
    if position is None:
        for key in ("exposure", "used_margin", "required_margin"):
            if not isclose(float(equity_row[key]), 0.0, abs_tol=1e-12):
                raise RunPersistenceError("checkpoint margin state is inconsistent with the closed position state")
        if equity_row["margin_level"] is not None:
            raise RunPersistenceError("checkpoint margin level must be null without an open position")
        return
    expected = {
        "exposure": position["notional"],
        "used_margin": position["required_margin"],
        "required_margin": position["required_margin"],
        "free_margin": float(equity_row["equity"]) - position["required_margin"],
    }
    for key, value in expected.items():
        if not isclose(float(equity_row[key]), float(value), rel_tol=1e-12, abs_tol=1e-12):
            raise RunPersistenceError(f"checkpoint {key} is inconsistent with the open position state")
    margin_level = equity_row["margin_level"]
    if margin_level is None or not isclose(float(margin_level), float(equity_row["equity"]) / float(position["required_margin"]), rel_tol=1e-12, abs_tol=1e-12):
        raise RunPersistenceError("checkpoint margin level is inconsistent with the open position state")


def _validate_checkpoint_risk_state(risk_state: Mapping[str, Any]) -> None:
    required = {"day_start", "events", "hard_breach", "total_block_entries", "daily_blocked_days"}
    missing = sorted(required - set(risk_state))
    if missing:
        raise RunPersistenceError(f"checkpoint risk state is incomplete; missing: {', '.join(missing)}")
    if not isinstance(risk_state["day_start"], Mapping) or not isinstance(risk_state["events"], (list, tuple)) or not isinstance(risk_state["daily_blocked_days"], (list, tuple)):
        raise RunPersistenceError("checkpoint risk state has invalid collection types")
    if not isinstance(risk_state["hard_breach"], bool) or not isinstance(risk_state["total_block_entries"], bool):
        raise RunPersistenceError("checkpoint risk state flags must be booleans")
    for day, value in risk_state["day_start"].items():
        try:
            date.fromisoformat(str(day))
        except ValueError as exc:
            raise RunPersistenceError("checkpoint risk day-start date is malformed") from exc
        _validate_checkpoint_number(value, "risk day-start equity")
    for day in risk_state["daily_blocked_days"]:
        try:
            date.fromisoformat(str(day))
        except ValueError as exc:
            raise RunPersistenceError("checkpoint blocked-day date is malformed") from exc


def _validate_checkpoint_trade(record: Any) -> None:
    if not isinstance(record, Mapping):
        raise RunPersistenceError("checkpoint trade must be an object")
    required = {"entry_time", "exit_time", "side", "quantity", "entry_price", "exit_price", "gross_pnl", "costs", "net_pnl", "exit_reason", "cost_breakdown"}
    missing = sorted(required - set(record))
    if missing:
        raise RunPersistenceError(f"checkpoint trade is incomplete; missing: {', '.join(missing)}")
    _checkpoint_timestamp(record["entry_time"], "trade entry timestamp")
    _checkpoint_timestamp(record["exit_time"], "trade exit timestamp")
    if record["side"] not in {"long", "short"} or not isinstance(record["exit_reason"], str):
        raise RunPersistenceError("checkpoint trade contains invalid identifiers")
    for key in ("quantity", "entry_price", "exit_price", "gross_pnl", "costs", "net_pnl"):
        _validate_checkpoint_number(record[key], f"trade {key}")
    _validate_checkpoint_costs(record["cost_breakdown"], "trade cost breakdown")


def _validate_checkpoint_position(record: Any, frame: pd.DataFrame, next_bar_index: int) -> None:
    if not isinstance(record, Mapping):
        raise RunPersistenceError("checkpoint open position must be an object or null")
    required = {"side", "entry_time", "entry_index", "entry_mid", "entry_price", "entry_costs", "quantity", "notional", "required_margin", "trade_id"}
    missing = sorted(required - set(record))
    if missing:
        raise RunPersistenceError(f"checkpoint position is incomplete; missing: {', '.join(missing)}")
    if record["side"] not in {"long", "short"} or not isinstance(record["trade_id"], str) or not record["trade_id"]:
        raise RunPersistenceError("checkpoint open position contains invalid identifiers")
    entry_index = record["entry_index"]
    if isinstance(entry_index, bool) or not isinstance(entry_index, int) or entry_index < 0 or entry_index >= next_bar_index:
        raise RunPersistenceError("checkpoint open position has an invalid entry_index")
    if _checkpoint_timestamp(record["entry_time"], "position entry timestamp") != _canonical_timestamp(frame.iloc[entry_index].timestamp):
        raise RunPersistenceError("checkpoint open position timestamp is inconsistent with entry_index")
    for key in ("entry_mid", "entry_price", "quantity", "notional", "required_margin", "stop", "take"):
        if key in record and record[key] is not None:
            _validate_checkpoint_number(record[key], f"position {key}")
    if record["quantity"] <= 0 or record["notional"] < 0 or record["required_margin"] < 0:
        raise RunPersistenceError("checkpoint open position has invalid quantities or margin")
    _validate_checkpoint_costs(record["entry_costs"], "position entry costs")


def _validate_checkpoint_costs(record: Any, label: str) -> None:
    if not isinstance(record, Mapping):
        raise RunPersistenceError(f"{label} must be an object")
    required = {"spread", "commission", "slippage", "swap", "conversion"}
    if set(record) != required:
        raise RunPersistenceError(f"{label} has unexpected or missing fields")
    for key, value in record.items():
        _validate_checkpoint_number(value, f"{label} {key}")


def _validate_checkpoint_events(events: Any, run_id: str, label: str, *, allow_unassigned_run_id: bool = False) -> None:
    previous = 0
    for item in events:
        if not isinstance(item, Mapping):
            raise RunPersistenceError(f"{label} contains a non-object event")
        sequence = item.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence != previous + 1:
            raise RunPersistenceError(f"{label} sequence numbers are not contiguous")
        previous = sequence
        event_run_id = item.get("run_id")
        if event_run_id != run_id and not (allow_unassigned_run_id and event_run_id is None):
            raise RunPersistenceError(f"{label} contains an event with an inconsistent run_id")
        if not isinstance(item.get("event_type"), str) or not item["event_type"]:
            raise RunPersistenceError(f"{label} contains an invalid event type")
        if item.get("timestamp") is not None:
            _checkpoint_timestamp(item["timestamp"], f"{label} timestamp")
        for key in ("loss", "equity", "threshold"):
            if key in item:
                _validate_checkpoint_number(item[key], f"{label} {key}")
        if not isinstance(item.get("details", {}), Mapping):
            raise RunPersistenceError(f"{label} event details must be an object")


def _checkpoint_timestamp(value: Any, label: str) -> str:
    try:
        timestamp = _canonical_timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunPersistenceError(f"checkpoint {label} is malformed") from exc
    return timestamp


def _validate_checkpoint_number(value: Any, label: str) -> None:
    if isinstance(value, bool):
        raise RunPersistenceError(f"checkpoint {label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunPersistenceError(f"checkpoint {label} must be numeric") from exc
    if not isfinite(number):
        raise RunPersistenceError(f"checkpoint {label} must be finite")


def _restore_checkpoint_state(
    record: dict[str, Any],
    initial_capital: float,
    risk: RiskConfig,
) -> tuple[RiskMonitor, list[Trade], list[dict[str, Any]], set[tuple[object, str]], dict[str, Any] | None, bool, bool]:
    try:
        state = record["state"]
        risk_state = state["risk"]
        if not isinstance(risk_state, Mapping):
            raise ValueError("risk state must be an object")
        monitor = RiskMonitor(initial_capital, risk)
        monitor._day_start = {
            date.fromisoformat(str(day)): _finite_checkpoint_number(value, "risk day-start equity")
            for day, value in risk_state.get("day_start", {}).items()
        }
        monitor.events = [_restore_event(item) for item in risk_state.get("events", ())]
        monitor.hard_breach = bool(risk_state.get("hard_breach", False))
        monitor.total_block_entries = bool(risk_state.get("total_block_entries", False))
        monitor.daily_blocked_days = {date.fromisoformat(str(day)) for day in risk_state.get("daily_blocked_days", ())}
        trades = [_restore_trade(item) for item in state["trades"]]
        equity_rows = []
        for item in state["equity_rows"]:
            if not isinstance(item, Mapping):
                raise ValueError("equity rows must be objects")
            restored = dict(item)
            restored["timestamp"] = pd.Timestamp(restored["timestamp"])
            for key, value in restored.items():
                if key != "timestamp" and value is not None:
                    restored[key] = _finite_checkpoint_number(value, f"equity field {key}")
            equity_rows.append(restored)
        margin_event_keys = {
            (pd.Timestamp(item[0]), str(item[1]))
            for item in state["margin_event_keys"]
        }
        position = _restore_position(state["open_position"])
        return monitor, trades, equity_rows, margin_event_keys, position, bool(state["pending_entry"]), bool(state["margin_blocked"])
    except (KeyError, TypeError, ValueError, OverflowError, IndexError) as exc:
        if isinstance(exc, RunPersistenceError):
            raise
        raise RunPersistenceError("checkpoint state is corrupted or malformed") from exc


def _restore_trade(record: Mapping[str, Any]) -> Trade:
    return Trade(
        pd.Timestamp(record["entry_time"]),
        pd.Timestamp(record["exit_time"]),
        str(record["side"]),
        _finite_checkpoint_number(record["quantity"], "trade quantity"),
        _finite_checkpoint_number(record["entry_price"], "trade entry price"),
        _finite_checkpoint_number(record["exit_price"], "trade exit price"),
        _finite_checkpoint_number(record["gross_pnl"], "trade gross P&L"),
        _finite_checkpoint_number(record["costs"], "trade costs"),
        _finite_checkpoint_number(record["net_pnl"], "trade net P&L"),
        str(record["exit_reason"]),
        CostBreakdown(**{key: _finite_checkpoint_number(value, f"trade cost {key}") for key, value in record["cost_breakdown"].items()}),
    )


def _restore_position(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    position = dict(record)
    position["entry_time"] = pd.Timestamp(position["entry_time"])
    position["entry_costs"] = CostBreakdown(**{
        key: _finite_checkpoint_number(value, f"position cost {key}")
        for key, value in position["entry_costs"].items()
    })
    for key in ("entry_mid", "entry_price", "quantity", "notional", "required_margin", "stop", "take"):
        if key in position and position[key] is not None:
            position[key] = _finite_checkpoint_number(position[key], f"position field {key}")
    position["entry_index"] = int(position["entry_index"])
    return position


def _restore_event(record: Mapping[str, Any]) -> BacktestEvent:
    timestamp = record.get("timestamp")
    return BacktestEvent(
        timestamp=pd.Timestamp(timestamp) if timestamp is not None else None,
        kind=str(record.get("kind", "")),
        loss=_finite_checkpoint_number(record.get("loss", 0.0), "event loss"),
        message=str(record.get("message", record.get("explanation", ""))),
        equity=_finite_checkpoint_number(record.get("equity", 0.0), "event equity"),
        threshold=_finite_checkpoint_number(record.get("threshold", 0.0), "event threshold"),
        action=str(record.get("action", "")),
        phase=str(record.get("phase", "")),
        status=str(record.get("status", "")),
        event_type=str(record.get("event_type", record.get("kind", ""))),
        severity=str(record.get("severity", "info")),
        run_id=record.get("run_id"),
        trade_id=record.get("trade_id"),
        order_id=record.get("order_id"),
        source=str(record.get("source", "simulated")),
        details=record.get("details", {}),
    )


def _finite_checkpoint_number(value: Any, label: str) -> float:
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _run_metadata(*, run_id: str, created_at: str, status: str, strategy: Strategy,
                  strategy_hash: str, dataset_hash: str, initial_capital: float,
                  configuration: dict[str, Any], configuration_hash: str,
                  input_hash: str, data_reference: str | None,
                  persistence: RunPersistenceConfig) -> dict[str, Any]:
    """Build JSON-serializable run metadata without machine-specific paths."""
    return {
        "run_id": run_id,
        "created_at": created_at,
        "status": status,
        "strategy": {
            "id": strategy.id,
            "version": strategy.version,
            "hash": strategy_hash,
            "definition": _canonical_strategy_definition(strategy),
        },
        "dataset_hash": dataset_hash,
        "data_reference": data_reference,
        "initial_capital": initial_capital,
        "configuration": configuration,
        "configuration_hash": configuration_hash,
        "cost_model_hash": _hash_payload(configuration["cost_model"]),
        "margin_config_hash": _hash_payload(configuration["margin"]),
        "risk_config_hash": _hash_payload(configuration["risk"]),
        "data_validation_config_hash": _hash_payload(configuration["data_validation"]),
        "execution_config_hash": _hash_payload(configuration["execution"]),
        "execution": dict(_EXECUTION_SETTINGS),
        "engine_version": ENGINE_VERSION,
        "software": _software_versions(),
        "input_hash": input_hash,
        "persistence": {
            "enabled": persistence.enabled,
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_boundary": "after_completed_bar",
        },
    }


def _minimum_history_for_strategy(strategy: Strategy) -> int:
    """Require enough rows for a next-bar entry and configured bar exits."""
    if not strategy.entry_rules:
        return 1
    minimum_rows = 3
    bar_exits = [int(rule["value"]) for rule in strategy.exit_rules if rule["type"] == "bars"]
    return minimum_rows + max(bar_exits, default=0)


def _configuration_payload(
    costs: CostModel,
    margin: MarginConfig,
    risk: RiskConfig,
    data_validation: DataValidationConfig,
    execution: ExecutionConfig,
) -> dict[str, Any]:
    return _canonicalize({
        "cost_model": asdict(costs),
        "margin": asdict(margin),
        "risk": asdict(risk),
        "data_validation": asdict(data_validation),
        "execution": asdict(execution),
    })


def _software_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "pandas": pd.__version__}


def _hash_strategy(strategy: Strategy) -> str:
    return _hash_payload(_canonical_strategy_definition(strategy))


def _canonical_strategy_definition(strategy: Strategy) -> dict[str, Any]:
    definition = strategy.to_dict()
    definition["created_at"] = _canonical_timestamp(definition["created_at"])
    return _canonicalize(definition)


def _hash_market_data(frame: pd.DataFrame) -> str:
    columns = list(_MARKET_COLUMNS)
    for column in (SPREAD_COLUMN, *BID_ASK_COLUMNS):
        if column in frame.columns:
            columns.append(column)
    rows = []
    for values in frame[columns].itertuples(index=False, name=None):
        row = {}
        for column, value in zip(columns, values):
            row[column] = _canonical_timestamp(value) if column == "timestamp" else _canonical_number(value)
        rows.append(row)
    return _hash_payload({"columns": columns, "rows": rows})


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(_canonicalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return _canonical_timestamp(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return _canonical_number(value)
    return value


def _canonical_number(value: Any) -> int | float:
    number = float(value)
    if not isfinite(number):
        raise ValueError("cannot hash non-finite numeric metadata")
    return int(number) if number.is_integer() else number


def _canonical_timestamp(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("cannot hash a timezone-naive timestamp")
    return timestamp.tz_convert("UTC").isoformat(timespec="microseconds").replace("+00:00", "Z")


def _add_drawdown_columns(equity_curve: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Add deterministic drawdown series to an already-recorded equity curve.

    ``drawdown_pct_initial`` is a decimal ratio using initial capital as the
    denominator, which is the denominator required by M1 rule R-07. The
    peak-relative ratio is exposed separately for conventional analysis.
    """
    result = equity_curve.copy()
    drawdown_columns = ("running_peak_equity", "drawdown_amount", "drawdown_pct_initial", "drawdown_pct_peak")
    if result.empty:
        for column in drawdown_columns:
            result[column] = pd.Series(dtype="float64", index=result.index)
        return result
    if not isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("initial_capital must be finite and positive for drawdown calculation")
    try:
        equity = pd.to_numeric(result["equity"], errors="raise").astype(float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("equity curve must contain numeric equity values") from exc
    if not equity.map(isfinite).all():
        raise ValueError("equity curve contains non-finite equity values")
    running_peak = equity.cummax()
    drawdown_amount = running_peak - equity
    result["running_peak_equity"] = running_peak
    result["drawdown_amount"] = drawdown_amount
    result["drawdown_pct_initial"] = drawdown_amount / initial_capital
    result["drawdown_pct_peak"] = drawdown_amount / running_peak.where(running_peak != 0)
    result["drawdown_pct_peak"] = result["drawdown_pct_peak"].fillna(0.0)
    return result


def _rules_match(rules: tuple[dict[str, Any], ...], frame: pd.DataFrame, i: int) -> bool:
    if not rules or i == 0:
        return False
    row, previous = frame.iloc[i], frame.iloc[i - 1]
    for rule in rules:
        kind, value = rule["type"], float(rule.get("value", 0))
        if kind == "always":
            continue
        if kind == "close_above" and not row.close > value:
            return False
        if kind == "close_below" and not row.close < value:
            return False
        if kind == "crosses_above" and not (previous.close <= value < row.close):
            return False
        if kind == "crosses_below" and not (previous.close >= value > row.close):
            return False
    return True


def _should_exit(position: dict[str, Any], row: Any, strategy: Strategy, index: int) -> bool:
    return _price_exit(position, row) is not None or any(
        rule["type"] == "bars" and index - position["entry_index"] >= int(rule["value"])
        for rule in strategy.exit_rules
    )


def _exit_price(position: dict[str, Any], row: Any, strategy: Strategy, index: int) -> tuple[float, str]:
    price_exit = _price_exit(position, row)
    return price_exit if price_exit is not None else (float(row.close), "rule_exit")


def _price_exit(position: dict[str, Any], row: Any) -> tuple[float, str] | None:
    """Return a price exit using only OHLC data.

    If a candle touches both protective levels, stop-loss wins conservatively
    because OHLC data cannot establish which level was reached first. If the
    candle opens beyond an active level, the fill is the open price (a gap),
    otherwise it is the configured level.
    """
    side = position["side"]
    stop = position.get("stop")
    take = position.get("take")
    open_price, high_price, low_price = _protective_prices(row, side)
    open_mid = _mid_price(row, "open")
    if side == "long":
        if stop is not None and low_price <= stop:
            return (open_mid if open_price <= stop else float(stop), "stop_loss")
        if take is not None and high_price >= take:
            return (open_mid if open_price >= take else float(take), "take_profit")
    else:
        if stop is not None and high_price >= stop:
            return (open_mid if open_price >= stop else float(stop), "stop_loss")
        if take is not None and low_price <= take:
            return (open_mid if open_price <= take else float(take), "take_profit")
    return None


def _market_data_mode(row: Any | None) -> str:
    if row is not None:
        columns = set(getattr(row, "index", ()))
        if set(BID_ASK_COLUMNS).issubset(columns):
            return "bid_ask"
        if SPREAD_COLUMN in columns:
            return "per_bar_spread"
    return "scalar"


def _row_value(row: Any, column: str) -> float:
    return float(row[column])


def _mid_price(row: Any, field: str) -> float:
    if _market_data_mode(row) == "bid_ask":
        return (_row_value(row, f"bid_{field}") + _row_value(row, f"ask_{field}")) / 2
    return _row_value(row, field)


def _protective_prices(row: Any, side: str) -> tuple[float, float, float]:
    """Return the executable-side open/high/low used for protective triggers."""
    if _market_data_mode(row) == "bid_ask":
        quote_side = "bid" if side == "long" else "ask"
        return (
            _row_value(row, f"{quote_side}_open"),
            _row_value(row, f"{quote_side}_high"),
            _row_value(row, f"{quote_side}_low"),
        )
    return _row_value(row, "open"), _row_value(row, "high"), _row_value(row, "low")


def _protective_exit_gapped(position: dict[str, Any], row: Any, reason: str) -> bool:
    if reason not in {"stop_loss", "take_profit"}:
        return False
    level = position.get("stop" if reason == "stop_loss" else "take")
    if level is None:
        return False
    open_price, _, _ = _protective_prices(row, position["side"])
    if position["side"] == "long":
        return open_price <= level if reason == "stop_loss" else open_price >= level
    return open_price >= level if reason == "stop_loss" else open_price <= level


def _fill(
    side: str,
    action: str,
    mid_price: float,
    quantity: float,
    costs: CostModel,
    *,
    include_trade_costs: bool = False,
    row: Any | None = None,
    price_field: str = "open",
) -> tuple[float, CostBreakdown]:
    mode = _market_data_mode(row)
    if mode == "bid_ask":
        if action == "entry":
            quote_side = "ask" if side == "long" else "bid"
            base_price = _row_value(row, f"{quote_side}_open")
            reference_mid = _mid_price(row, "open")
        elif price_field == "close":
            quote_side = "bid" if side == "long" else "ask"
            base_price = _row_value(row, f"{quote_side}_close")
            reference_mid = _mid_price(row, "close")
        elif price_field == "open":
            quote_side = "bid" if side == "long" else "ask"
            base_price = _row_value(row, f"{quote_side}_open")
            reference_mid = _mid_price(row, "open")
        else:
            # A full bid/ask OHLC candle does not identify the exact quote at
            # an intrabar stop/take level.  The configured trigger is treated
            # as an already executable side price, so no second spread is
            # charged for that fill.
            base_price = mid_price
            reference_mid = mid_price
        slippage = costs.slippage if (side == "long") == (action == "entry") else -costs.slippage
        price = base_price + slippage
        spread_cost = abs(base_price - reference_mid) * quantity
    else:
        spread = costs.spread if mode == "scalar" else _row_value(row, SPREAD_COLUMN)
        half_spread = spread / 2
        if side == "long":
            adverse = half_spread if action == "entry" else -half_spread
            slippage = costs.slippage if action == "entry" else -costs.slippage
        else:
            adverse = -half_spread if action == "entry" else half_spread
            slippage = -costs.slippage if action == "entry" else costs.slippage
        price = mid_price + adverse + slippage
        spread_cost = half_spread * quantity
    breakdown = CostBreakdown(
        spread=spread_cost,
        commission=costs.commission,
        slippage=costs.slippage * quantity,
        swap=costs.swap if include_trade_costs else 0.0,
        conversion=costs.conversion if include_trade_costs else 0.0,
    )
    return price, breakdown


def _open_equity(position: dict[str, Any], mark_price: float, cash: float, costs: CostModel, *, row: Any | None = None) -> float:
    """Mark an open position at a completed candle's close.

    The unrealized result uses the entry execution price and the estimated
    liquidation-side execution price. Entry and exit price costs are therefore
    embedded once in the execution P&L; only non-price charges are deducted
    separately.
    """
    return _equity_components([], position, mark_price, cash, costs, row=row)["equity"]


def _equity_components(
    trades: list[Trade],
    position: dict[str, Any] | None,
    mark_price: float | None,
    cash: float,
    costs: CostModel,
    *,
    row: Any | None = None,
) -> dict[str, float]:
    """Return explicit P&L and cost components for one recorded equity point.

    ``realized_pnl`` is the sum of closed trades' net P&L, including their
    actual costs.  ``unrealized_pnl`` is the open position's net result at the
    completed-candle liquidation estimate, including entry costs and estimated
    liquidation costs.  ``gross_pnl`` uses midpoint prices and excludes all
    costs.  ``total_costs`` includes actual closed-trade costs plus estimated
    costs for the open position.  Therefore both
    ``equity == cash + unrealized_pnl`` and
    ``equity == cash == initial_capital + realized_pnl`` when flat, while the
    full gross/cost reconciliation is ``equity == initial_capital + gross_pnl -
    total_costs``.
    """
    realized_pnl = sum(trade.net_pnl for trade in trades)
    gross_pnl = sum(trade.gross_pnl for trade in trades)
    total_costs = sum(trade.costs for trade in trades)
    unrealized_pnl = 0.0
    if position is not None:
        if mark_price is None:
            raise ValueError("mark_price is required for an open position")
        mark_execution_price, estimated_exit_costs = _fill(
            position["side"], "exit", mark_price, position["quantity"], costs,
            include_trade_costs=True, row=row, price_field="close",
        )
        open_gross = _pnl(position["side"], position["entry_mid"], mark_price, position["quantity"])
        open_costs = position["entry_costs"] + estimated_exit_costs
        gross_pnl += open_gross
        total_costs += open_costs.total
        unrealized_pnl = _pnl(position["side"], position["entry_price"], mark_execution_price, position["quantity"])
        entry_non_price_costs = position["entry_costs"].commission + position["entry_costs"].swap + position["entry_costs"].conversion
        exit_non_price_costs = estimated_exit_costs.commission + estimated_exit_costs.swap + estimated_exit_costs.conversion
        unrealized_pnl -= entry_non_price_costs + exit_non_price_costs
    equity = cash + unrealized_pnl
    return {
        "cash": float(cash),
        "realized_pnl": float(realized_pnl),
        "unrealized_pnl": float(unrealized_pnl),
        "gross_pnl": float(gross_pnl),
        "total_costs": float(total_costs),
        "equity": float(equity),
    }


def _margin_state(position: dict[str, Any] | None, equity: float) -> dict[str, Any]:
    if position is None:
        return {"exposure": 0.0, "used_margin": 0.0, "required_margin": 0.0, "free_margin": equity, "margin_level": None}
    used_margin = float(position["required_margin"])
    free_margin = equity - used_margin
    return {
        "exposure": float(position["notional"]), "used_margin": used_margin, "required_margin": used_margin,
        "free_margin": free_margin, "margin_level": equity / used_margin if used_margin else None,
    }


def _check_risk(monitor: RiskMonitor, ledger: EventLedger, timestamp: object, equity: float, phase: str) -> None:
    before = len(monitor.events)
    monitor.check(timestamp, equity, phase=phase)
    for event in monitor.events[before:]:
        ledger.add(event)


def _record_margin_event(ledger: EventLedger, keys: set[tuple[object, str]], timestamp: object, kind: str, message: str, required_margin: float, free_margin: float, exposure: float, requested_quantity: float | None = None) -> None:
    key = (timestamp, kind)
    if key in keys:
        return
    keys.add(key)
    ledger.emit(
        "margin_breach" if kind == "margin_breach" else "entry_rejected",
        timestamp=timestamp,
        kind=kind,
        severity="error" if kind == "margin_breach" else "warning",
        phase="margin" if kind == "margin_breach" else "entry",
        message=message,
        action="block_new_entries" if kind == "margin_breach" else "reject_entry",
        details={"required_margin": required_margin, "free_margin": free_margin, "exposure": exposure, "requested_quantity": requested_quantity, "filled_quantity": 0.0, "remaining_quantity": requested_quantity},
    )


def _cost_details(costs: CostBreakdown) -> dict[str, float]:
    return {
        "spread": costs.spread,
        "commission": costs.commission,
        "slippage": costs.slippage,
        "swap": costs.swap,
        "conversion": costs.conversion,
        "total": costs.total,
    }


def _attach_failure_events(ledger: EventLedger, error: BaseException, event_type: str, message: str) -> None:
    ledger.emit(event_type, phase="validation", severity="error", message=message, status="REJECTED", details={"error": str(error)})
    ledger.emit("backtest_rejected", phase="terminal", severity="error", message="backtest rejected", status="REJECTED", details={"error": str(error)})
    setattr(error, "events", ledger.to_dicts())


def _pnl(side: str, entry: float, exit: float, quantity: float) -> float:
    return (exit - entry) * quantity if side == "long" else (entry - exit) * quantity

