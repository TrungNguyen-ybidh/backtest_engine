"""Pluggable cost models: commission and slippage, plus order types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional


class CommissionModel(ABC):
    @abstractmethod
    def calculate(self, shares: int, price: float) -> float:
        ...


class PerShareCommission(CommissionModel):
    def __init__(self, per_share: float = 0.005, minimum: float = 1.0):
        self.per_share = per_share
        self.minimum = minimum

    def calculate(self, shares: int, price: float) -> float:
        return max(self.minimum, abs(shares) * self.per_share)


class SlippageModel(ABC):
    @abstractmethod
    def adjust(self, price: float, shares: int) -> float:
        ...


class FixedBpsSlippage(SlippageModel):
    def __init__(self, bps: float = 5.0):
        self.bps = bps

    def adjust(self, price: float, shares: int) -> float:
        direction = 1 if shares > 0 else -1
        return price * (1 + direction * self.bps / 10_000)


# v1.1 (M9): limit orders.

_LIMIT_LIFETIMES = {"day", "gtc_expire", "gtc"}
_LIMIT_SIDES = {"buy", "sell"}


@dataclass
class LimitOrder:
    """A pending limit order in Broker.open_orders.

    Convention: `shares` is always positive; direction is encoded by `side`.
    Fill rule (checked by Broker.process_open_orders each day):
      - buy:  fills if today's low  <= limit_price.
      - sell: fills if today's high >= limit_price.
    Fill price is `limit_price` (no slippage); commission still applies.

    A limit order placed on day D is eligible to fill starting day D+1.
    This avoids the appearance of look-ahead — the strategy saw day D's
    bar before choosing the limit price, so letting it also fill on day D
    would be reusing that data both to size the order and to fill it.
    """

    ticker: str
    side: str
    shares: int
    limit_price: float
    lifetime: str = "day"
    expiration_date: Optional[date] = None

    # Set by Broker.place_limit_order at placement; do not pass at construction.
    placed_on: Optional[date] = None

    def __post_init__(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker:
            raise ValueError("ticker must be a non-empty string")
        self.ticker = self.ticker.upper()

        if self.side not in _LIMIT_SIDES:
            raise ValueError(
                f"side must be one of {sorted(_LIMIT_SIDES)}, got {self.side!r}"
            )
        if not isinstance(self.shares, int) or isinstance(self.shares, bool):
            raise TypeError(f"shares must be int, got {type(self.shares).__name__}")
        if self.shares <= 0:
            raise ValueError(f"shares must be positive, got {self.shares}")
        if not isinstance(self.limit_price, (int, float)) or self.limit_price <= 0:
            raise ValueError(f"limit_price must be > 0, got {self.limit_price}")

        if self.lifetime not in _LIMIT_LIFETIMES:
            raise ValueError(
                f"lifetime must be one of {sorted(_LIMIT_LIFETIMES)}, got {self.lifetime!r}"
            )
        # `gtc_expire` requires an expiration_date; the other two must NOT have one.
        # This keeps the lifetime/expiration_date contract unambiguous at the type level.
        if self.lifetime == "gtc_expire":
            if self.expiration_date is None:
                raise ValueError("lifetime='gtc_expire' requires expiration_date")
            if not isinstance(self.expiration_date, date):
                raise TypeError("expiration_date must be a datetime.date")
        else:
            if self.expiration_date is not None:
                raise ValueError(
                    f"expiration_date is only valid for lifetime='gtc_expire', "
                    f"not {self.lifetime!r}"
                )
