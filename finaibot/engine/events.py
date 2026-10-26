"""Structured, deterministic events for a simulated backtest run."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from numbers import Real
from collections.abc import Mapping
from typing import Any


@dataclass(frozen=True)
class BacktestEvent:
    """One simulated lifecycle event.

    The legacy fields (``kind``, ``loss``, ``equity``, ``threshold``,
    ``action``, and ``status``) remain first-class fields so existing risk and
    margin consumers continue to work.  ``event_type`` and the identifiers
    provide the consistent ledger contract used by new events.
    """

    timestamp: Any = None
    kind: str = ""
    loss: float = 0.0
    message: str = ""
    equity: float = 0.0
    threshold: float = 0.0
    action: str = ""
    phase: str = ""
    status: str = ""
    sequence: int = 0
    event_type: str = ""
    severity: str = "info"
    run_id: str | None = None
    trade_id: str | None = None
    order_id: str | None = None
    source: str = "simulated"
    details: Mapping[str, Any] | None = None


class EventLedger:
    """Append-only event ledger with deterministic sequence numbers."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id
        self._events: list[BacktestEvent] = []

    @property
    def events(self) -> tuple[BacktestEvent, ...]:
        return tuple(self._events)

    @classmethod
    def from_dicts(cls, events: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...], run_id: str | None = None) -> "EventLedger":
        """Restore JSON event records without deserializing executable objects."""
        ledger = cls(run_id)
        for item in events:
            if not isinstance(item, Mapping):
                raise TypeError("event records must be mappings")
            ledger.add(BacktestEvent(
                timestamp=item.get("timestamp"),
                kind=str(item.get("kind", "")),
                loss=item.get("loss", 0.0),
                message=str(item.get("message", item.get("explanation", ""))),
                equity=item.get("equity", 0.0),
                threshold=item.get("threshold", 0.0),
                action=str(item.get("action", "")),
                phase=str(item.get("phase", "")),
                status=str(item.get("status", "")),
                event_type=str(item.get("event_type", item.get("kind", ""))),
                severity=str(item.get("severity", "info")),
                run_id=item.get("run_id"),
                trade_id=item.get("trade_id"),
                order_id=item.get("order_id"),
                source=str(item.get("source", "simulated")),
                details=item.get("details", {}),
            ))
        return ledger

    def add(self, event: BacktestEvent) -> BacktestEvent:
        if not isinstance(event, BacktestEvent):
            raise TypeError("event must be a BacktestEvent")
        normalized = replace(
            event,
            sequence=len(self._events) + 1,
            event_type=event.event_type or event.kind,
            run_id=event.run_id or self.run_id,
        )
        self._events.append(normalized)
        return normalized

    def emit(
        self,
        event_type: str,
        *,
        timestamp: Any = None,
        kind: str | None = None,
        severity: str = "info",
        phase: str = "",
        message: str = "",
        trade_id: str | None = None,
        order_id: str | None = None,
        status: str = "",
        action: str = "",
        loss: float = 0.0,
        equity: float = 0.0,
        threshold: float = 0.0,
        details: Mapping[str, Any] | None = None,
    ) -> BacktestEvent:
        return self.add(BacktestEvent(
            timestamp=timestamp,
            kind=kind or event_type,
            loss=loss,
            message=message,
            equity=equity,
            threshold=threshold,
            action=action,
            phase=phase,
            status=status,
            event_type=event_type,
            severity=severity,
            trade_id=trade_id,
            order_id=order_id,
            details=details,
        ))

    def to_dicts(self) -> tuple[dict[str, Any], ...]:
        return tuple(_event_to_dict(event) for event in self._events)


def _event_to_dict(event: BacktestEvent) -> dict[str, Any]:
    explanation = event.message
    return {
        "sequence": event.sequence,
        "timestamp": _timestamp_value(event.timestamp),
        "event_type": event.event_type or event.kind,
        "kind": event.kind,
        "severity": event.severity,
        "phase": event.phase,
        "message": explanation,
        "explanation": explanation,
        "run_id": event.run_id,
        "trade_id": event.trade_id,
        "order_id": event.order_id,
        "source": event.source,
        "loss": _json_value(event.loss),
        "equity": _json_value(event.equity),
        "threshold": _json_value(event.threshold),
        "action": event.action,
        "status": event.status,
        "details": _json_value(event.details or {}),
    }


def _timestamp_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        timestamp = value
        if timestamp.tzinfo is not None and timestamp.utcoffset() is not None:
            timestamp = timestamp.astimezone(timezone.utc)
        return timestamp.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if hasattr(value, "tz_convert") and hasattr(value, "isoformat"):
        try:
            return value.tz_convert("UTC").isoformat(timespec="microseconds").replace("+00:00", "Z")
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return _json_value(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if isinstance(value, (datetime,)):
        return _timestamp_value(value)
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value) if not float(value).is_integer() else int(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)
