"""Strategy base class + example strategies.

Rules:
- generate_signals() returns {ticker: weight}, weights are 0.0-1.0, sum <= 1.0.
- Strategy NEVER touches Broker.
- Strategy NEVER reads data beyond current_date.

The base class enforces both invariants via the template method pattern:
subclasses override `_compute`, and `generate_signals` wraps it with input
(no look-ahead) and output (weight contract) validation.
"""

import math
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


# Tolerance for floating-point sum check (e.g., 1/3 * 3 = 0.9999...).
_WEIGHT_SUM_EPS = 1e-9


class Strategy(ABC):
    def generate_signals(
        self, current_date: date, price_data: pd.DataFrame
    ) -> dict[str, float]:
        # Rule #1: no look-ahead. Data must end at or before current_date.
        if not price_data.empty and "date" in price_data.columns:
            max_date = price_data["date"].max()
            # Coerce pandas Timestamp -> date for comparison
            if hasattr(max_date, "date"):
                max_date = max_date.date()
            if max_date > current_date:
                raise ValueError(
                    f"look-ahead violation: price_data through {max_date}, "
                    f"current_date is {current_date}"
                )

        weights = self._compute(current_date, price_data)
        self._validate_weights(weights)
        return weights

    @abstractmethod
    def _compute(
        self, current_date: date, price_data: pd.DataFrame
    ) -> dict[str, float]:
        ...

    @staticmethod
    def _validate_weights(weights: dict[str, float]) -> None:
        if not isinstance(weights, dict):
            raise TypeError(f"weights must be dict, got {type(weights).__name__}")
        total = 0.0
        for ticker, w in weights.items():
            if not isinstance(ticker, str) or not ticker:
                raise ValueError(f"ticker keys must be non-empty strings, got {ticker!r}")
            if not isinstance(w, (int, float)) or isinstance(w, bool):
                raise TypeError(f"weight for {ticker!r} must be numeric, got {type(w).__name__}")
            if math.isnan(w) or math.isinf(w):
                raise ValueError(f"weight for {ticker!r} is not finite: {w}")
            if w < 0.0 or w > 1.0:
                raise ValueError(f"weight for {ticker!r} must be in [0, 1], got {w}")
            total += w
        if total > 1.0 + _WEIGHT_SUM_EPS:
            raise ValueError(f"weights sum to {total}, must be <= 1.0")


class AllCash(Strategy):
    """Hold 100% cash. Reference for validation test #1 (0% return, 0 trades)."""

    def _compute(self, current_date, price_data):
        return {}


class BuyAndHold(Strategy):
    """Equal-weight across `tickers`, rebalanced to target weights each period.

    Reference for validation test #2 (return matches manual calc on a single
    ticker) and test #3 (broad universe ≈ S&P 500 ballpark).
    """

    def __init__(self, tickers: list[str]):
        if not isinstance(tickers, list):
            raise TypeError(f"tickers must be a list, got {type(tickers).__name__}")
        if any(not isinstance(t, str) or not t for t in tickers):
            raise ValueError("every ticker must be a non-empty string")
        self.tickers = [t.upper() for t in tickers]

    def _compute(self, current_date, price_data):
        if not self.tickers:
            return {}
        w = 1.0 / len(self.tickers)
        return {t: w for t in self.tickers}
