"""Pluggable cost models: commission and slippage."""

from abc import ABC, abstractmethod


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
