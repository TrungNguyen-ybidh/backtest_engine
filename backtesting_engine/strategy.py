"""Strategy base class + concrete strategies.

Rules:
- `generate_signals()` returns `{ticker: weight}`, weights are 0.0-1.0, sum <= 1.0.
- Strategy NEVER touches Broker.
- Strategy NEVER touches DataInterface directly — engine fetches data and passes
  it in. If a strategy needs fundamentals, declare them via the class attribute
  `requires_fundamentals` and the engine will pre-fetch + slice them per day.
- Strategy NEVER reads data beyond current_date. The base class enforces this
  on price_data via the template-method wrapper.
"""

import math
from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import pandas as pd


# Tolerance for floating-point sum check (e.g., 1/3 * 3 = 0.9999...).
_WEIGHT_SUM_EPS = 1e-9


class Strategy(ABC):
    # Subclasses declare which fundamental tables the engine should fetch.
    # Empty tuple = price-only strategy. Engine does NOT pass `fundamentals`
    # to `_compute` when this is empty — the kwarg is None.
    requires_fundamentals: tuple[str, ...] = ()

    def generate_signals(
        self,
        current_date: date,
        price_data: pd.DataFrame,
        fundamentals: Optional[dict[str, pd.DataFrame]] = None,
    ) -> dict[str, float]:
        # Rule #1: no look-ahead. price_data must end at or before current_date.
        if not price_data.empty and "date" in price_data.columns:
            max_date = price_data["date"].max()
            if hasattr(max_date, "date"):
                max_date = max_date.date()
            if max_date > current_date:
                raise ValueError(
                    f"look-ahead violation: price_data through {max_date}, "
                    f"current_date is {current_date}"
                )

        # Rule #2 belt-and-braces: also verify fundamentals are filing-date-safe.
        # Engine already slices on filing_date <= current_date, but a sanity check
        # is cheap and would catch a buggy engine change immediately.
        if fundamentals:
            for name, df in fundamentals.items():
                if df is None or df.empty or "filing_date" not in df.columns:
                    continue
                max_filing = df["filing_date"].max()
                if hasattr(max_filing, "date"):
                    max_filing = max_filing.date()
                if max_filing > current_date:
                    raise ValueError(
                        f"look-ahead in fundamentals[{name!r}]: filing_date "
                        f"{max_filing} > current_date {current_date}"
                    )

        weights = self._compute(current_date, price_data, fundamentals)
        self._validate_weights(weights)
        return weights

    @abstractmethod
    def _compute(
        self,
        current_date: date,
        price_data: pd.DataFrame,
        fundamentals: Optional[dict[str, pd.DataFrame]],
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

    def _compute(self, current_date, price_data, fundamentals):
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

    def _compute(self, current_date, price_data, fundamentals):
        if not self.tickers:
            return {}
        w = 1.0 / len(self.tickers)
        return {t: w for t in self.tickers}


class MovingAverageCrossover(Strategy):
    """Fast/slow MA crossover, per ticker.

    Each ticker gets a fixed 1/N slot — only "bullish" tickers (fast_ma > slow_ma)
    hold their slot, the rest sit in cash. Three of five bullish → 60% invested.
    Ticker is skipped until it has at least `slow_window` bars of history.
    """

    def __init__(
        self,
        tickers: list[str],
        fast_window: int = 50,
        slow_window: int = 200,
    ):
        if not isinstance(tickers, list):
            raise TypeError(f"tickers must be a list, got {type(tickers).__name__}")
        if any(not isinstance(t, str) or not t for t in tickers):
            raise ValueError("every ticker must be a non-empty string")
        if not isinstance(fast_window, int) or not isinstance(slow_window, int):
            raise TypeError("fast_window and slow_window must be ints")
        if fast_window <= 0 or slow_window <= 0:
            raise ValueError("windows must be positive")
        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window ({fast_window}) must be < slow_window ({slow_window})"
            )

        self.tickers = [t.upper() for t in tickers]
        self.fast_window = fast_window
        self.slow_window = slow_window

    def _compute(self, current_date, price_data, fundamentals):
        if not self.tickers or price_data.empty:
            return {}

        slot = 1.0 / len(self.tickers)
        weights: dict[str, float] = {}

        for ticker in self.tickers:
            ticker_closes = (
                price_data.loc[price_data["ticker"] == ticker, "close"]
                .tail(self.slow_window)
            )
            if len(ticker_closes) < self.slow_window:
                continue  # not enough history for slow MA — sit out
            slow_ma = ticker_closes.mean()
            fast_ma = ticker_closes.tail(self.fast_window).mean()
            if fast_ma > slow_ma:
                weights[ticker] = slot

        return weights
