"""Broker — simulates trade execution.

Invariant: cash + sum(shares * price) == total_value at all times.
"""

from .models import CommissionModel, SlippageModel


class Broker:
    def __init__(
        self,
        initial_cash: float,
        commission: CommissionModel,
        slippage: SlippageModel,
    ):
        self.cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.holdings: dict[str, int] = {}

    def execute_order(self, ticker: str, shares: int, price: float) -> None:
        raise NotImplementedError

    def total_value(self, prices: dict[str, float]) -> float:
        raise NotImplementedError
