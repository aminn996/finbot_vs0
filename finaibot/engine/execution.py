"""Deterministic simulated order-fill assumptions for the M1 backtester."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ExecutionConfig:
    """Configure the deliberately simplified simulated entry-fill policy.

    With the default ``None`` cap, accepted entries fill completely, which
    preserves the original M1 behavior.  When a positive cap is configured,
    an entry fills ``min(requested_quantity, cap)`` and the remaining quantity
    is cancelled.  This is a deterministic simulation assumption, not a
    liquidity observation or broker-equivalent execution model. Exit orders
    remain full-fill in M1.
    """

    entry_fill_quantity_cap: float | None = None
    remaining_quantity_policy: str = "cancel"

    def __post_init__(self) -> None:
        if self.entry_fill_quantity_cap is not None:
            try:
                cap = float(self.entry_fill_quantity_cap)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("entry_fill_quantity_cap must be finite and positive") from exc
            if isinstance(self.entry_fill_quantity_cap, bool) or not isfinite(cap) or cap <= 0:
                raise ValueError("entry_fill_quantity_cap must be finite and positive")
            object.__setattr__(self, "entry_fill_quantity_cap", cap)
        if self.remaining_quantity_policy != "cancel":
            raise ValueError("remaining_quantity_policy must be 'cancel'")

    def fill_quantity(self, requested_quantity: float) -> tuple[float, float, str]:
        """Return filled quantity, cancelled remainder, and fill status."""
        try:
            requested = float(requested_quantity)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("requested quantity must be finite and positive") from exc
        if not isfinite(requested) or requested <= 0:
            raise ValueError("requested quantity must be finite and positive")
        filled = requested if self.entry_fill_quantity_cap is None else min(requested, self.entry_fill_quantity_cap)
        remaining = requested - filled
        status = "filled" if remaining == 0 else "partially_filled"
        return filled, remaining, status
