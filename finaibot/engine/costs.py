"""Validated execution-cost assumptions for the deterministic M1 engine.

``spread`` and ``slippage`` are price units per fill. ``spread`` is the full
bid/ask width; the engine charges half on each side of a round trip.
``commission`` is account currency per fill. ``swap`` and ``conversion`` are
account-currency charges applied once when a trade is completed.
"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class CostModel:
    spread: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    swap: float = 0.0
    conversion: float = 0.0

    def __post_init__(self) -> None:
        for name in ("spread", "commission", "slippage", "swap", "conversion"):
            value = getattr(self, name)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be a finite, non-negative number") from exc
            if isinstance(value, bool) or not isfinite(numeric_value) or numeric_value < 0:
                raise ValueError(f"{name} must be a finite, non-negative number")
            object.__setattr__(self, name, numeric_value)

    def total(self) -> float:
        """Return the round-trip cost for one unit of quantity.

        This compatibility helper is not used for trade accounting because
        spread and slippage depend on quantity and fill direction.
        """
        return self.spread + (2 * self.commission) + (2 * self.slippage) + self.swap + self.conversion


@dataclass(frozen=True)
class CostBreakdown:
    spread: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    swap: float = 0.0
    conversion: float = 0.0

    @property
    def total(self) -> float:
        return self.spread + self.commission + self.slippage + self.swap + self.conversion

    def __add__(self, other: "CostBreakdown") -> "CostBreakdown":
        return CostBreakdown(
            self.spread + other.spread, self.commission + other.commission,
            self.slippage + other.slippage, self.swap + other.swap, self.conversion + other.conversion,
        )

