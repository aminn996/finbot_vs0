"""Independent, non-strategy-modifiable risk protections."""

from dataclasses import dataclass
from math import isfinite
from .events import BacktestEvent


@dataclass(frozen=True)
class MarginConfig:
    """Simplified initial-margin and exposure controls for the M1 simulator.

    Notional is calculated as ``abs(mid_price * quantity)`` in quote/account
    currency. Required margin is notional divided by leverage. This is not a
    broker-equivalent model and does not support maintenance-margin liquidation.
    """

    leverage: float = 1.0
    max_exposure: float | None = None

    def __post_init__(self) -> None:
        try:
            leverage = float(self.leverage)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("leverage must be a finite, positive number") from exc
        if isinstance(self.leverage, bool) or not isfinite(leverage) or leverage <= 0:
            raise ValueError("leverage must be a finite, positive number")
        object.__setattr__(self, "leverage", leverage)
        if self.max_exposure is not None:
            try:
                max_exposure = float(self.max_exposure)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("max_exposure must be finite and non-negative") from exc
            if isinstance(self.max_exposure, bool) or not isfinite(max_exposure) or max_exposure < 0:
                raise ValueError("max_exposure must be finite and non-negative")
            object.__setattr__(self, "max_exposure", max_exposure)


@dataclass(frozen=True)
class RiskConfig:
    daily_preventive_pct: float = 0.04
    daily_hard_pct: float = 0.05
    total_preventive_pct: float = 0.08
    total_hard_pct: float = 0.10

    def __post_init__(self) -> None:
        values = (
            ("daily_preventive_pct", self.daily_preventive_pct),
            ("daily_hard_pct", self.daily_hard_pct),
            ("total_preventive_pct", self.total_preventive_pct),
            ("total_hard_pct", self.total_hard_pct),
        )
        normalized: dict[str, float] = {}
        for name, value in values:
            try:
                converted = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite, positive number") from exc
            if isinstance(value, bool) or not isfinite(converted) or converted <= 0:
                raise ValueError(f"{name} must be a finite, positive number")
            normalized[name] = converted
        if normalized["daily_preventive_pct"] >= normalized["daily_hard_pct"]:
            raise ValueError("daily_preventive_pct must be below daily_hard_pct")
        if normalized["total_preventive_pct"] >= normalized["total_hard_pct"]:
            raise ValueError("total_preventive_pct must be below total_hard_pct")
        for name, value in normalized.items():
            object.__setattr__(self, name, value)


RiskEvent = BacktestEvent


class RiskMonitor:
    def __init__(self, initial_capital: float, config: RiskConfig = RiskConfig()):
        self.initial_capital = initial_capital
        self.config = config
        self._day_start: dict[object, float] = {}
        self.events: list[RiskEvent] = []
        self.hard_breach = False
        self.total_block_entries = False
        self.daily_blocked_days: set[object] = set()

    def begin_period(self, timestamp: object, equity: float) -> None:
        """Set the first equity observed for a calendar day as its baseline."""
        day = getattr(timestamp, "date", lambda: timestamp)()
        self._day_start.setdefault(day, equity)

    def entries_allowed(self, timestamp: object) -> bool:
        day = getattr(timestamp, "date", lambda: timestamp)()
        return not self.total_block_entries and day not in self.daily_blocked_days and not self.hard_breach

    def check(self, timestamp: object, equity: float, phase: str = "bar_close") -> None:
        self.begin_period(timestamp, equity)
        day = getattr(timestamp, "date", lambda: timestamp)()
        daily_loss = max(0.0, self._day_start[day] - equity)
        total_loss = max(0.0, self.initial_capital - equity)
        checks = ((daily_loss, self.config.daily_preventive_pct, "daily"), (total_loss, self.config.total_preventive_pct, "total"))
        for loss, threshold, label in checks:
            kind = f"{label}_preventive"
            already_triggered = any(e.kind == kind and (e.timestamp == timestamp or (label == "daily" and getattr(e.timestamp, "date", lambda: e.timestamp)() == day)) for e in self.events)
            if loss >= threshold * self.initial_capital and not already_triggered:
                action = "block_entries_for_day" if label == "daily" else "block_new_entries"
                self.events.append(RiskEvent(
                    timestamp=timestamp,
                    kind=kind,
                    loss=loss,
                    message=f"{label} preventive stop reached",
                    equity=equity,
                    threshold=threshold * self.initial_capital,
                    action=action,
                    phase=phase,
                    status="PROTECTION_ACTIVE",
                    event_type=f"risk.{kind}",
                    severity="warning",
                ))
                if label == "daily":
                    self.daily_blocked_days.add(day)
                else:
                    self.total_block_entries = True
        hard = ((daily_loss, self.config.daily_hard_pct, "daily"), (total_loss, self.config.total_hard_pct, "total"))
        for loss, threshold, label in hard:
            if loss > threshold * self.initial_capital:
                self.hard_breach = True
                self.total_block_entries = True
                kind = f"{label}_hard"
                if not any(e.kind == kind and e.timestamp == timestamp for e in self.events):
                    self.events.append(RiskEvent(
                        timestamp=timestamp,
                        kind=kind,
                        loss=loss,
                        message=f"{label} hard limit exceeded",
                        equity=equity,
                        threshold=threshold * self.initial_capital,
                        action="terminate_and_reject_backtest",
                        phase=phase,
                        status="REJECTED",
                        event_type=f"risk.{kind}",
                        severity="error",
                    ))

