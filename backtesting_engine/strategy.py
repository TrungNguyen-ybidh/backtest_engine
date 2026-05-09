"""Strategy base class + example strategies.

Rules:
- generate_signals() returns {ticker: weight}, weights are 0.0-1.0, sum <= 1.0.
- Strategy NEVER touches Broker.
- Strategy NEVER reads data beyond current_date.
"""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_signals(
        self, current_date: date, price_data: pd.DataFrame
    ) -> dict[str, float]:
        ...


class AllCash(Strategy):
    def generate_signals(self, current_date, price_data):
        return {}


class BuyAndHold(Strategy):
    def __init__(self, tickers: list[str]):
        self.tickers = tickers

    def generate_signals(self, current_date, price_data):
        if not self.tickers:
            return {}
        w = 1.0 / len(self.tickers)
        return {t: w for t in self.tickers}
