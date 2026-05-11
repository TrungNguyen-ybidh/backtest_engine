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

from .models import LimitOrder


# Tolerance for floating-point sum check (e.g., 1/3 * 3 = 0.9999...).
_WEIGHT_SUM_EPS = 1e-9


class Strategy(ABC):
    # Subclasses declare which fundamental tables the engine should fetch.
    # Empty tuple = price-only strategy. Engine does NOT pass `fundamentals`
    # to `_compute` when this is empty — the kwarg is None.
    requires_fundamentals: tuple[str, ...] = ()

    # M9: when True, the engine does NOT auto-exit holdings missing from the
    # returned weight dict. The strategy is asserting "I handle entries and
    # exits myself" — typically via limit orders. Default False preserves v1
    # semantics where {ticker: weight} is the FULL target portfolio.
    manages_positions: bool = False

    # M10: when True, _validate_weights accepts negative weights in [-1, 1]
    # with sum(|w|) <= 1.0. Engine writes this from config.allow_short at the
    # start of run() so the strategy and broker stay in sync. Default False
    # preserves the v1 long-only contract exactly.
    allow_short: bool = False

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

    def generate_orders(
        self,
        current_date: date,
        price_data: pd.DataFrame,
        fundamentals: Optional[dict[str, pd.DataFrame]] = None,
        holdings: Optional[dict[str, int]] = None,
    ) -> list[LimitOrder]:
        """v1.1 (M9): emit new limit orders for the broker's open book.

        Default implementation returns []. Subclasses override to use limits.
        Coexists with `generate_signals` — engine routes weights to market
        orders (only on rebalance days) and limit orders to the limit book
        (every day, regardless of rebalance frequency).

        `holdings` is a read-only snapshot of `{ticker: shares}` passed by the
        engine. Strategy can use it to make share-count decisions (e.g., skip
        emitting a buy limit when already in the position). The strategy still
        never touches Broker directly — this is the same hand-in-data pattern
        as `price_data`.
        """
        return []

    def _validate_weights(self, weights: dict[str, float]) -> None:
        """Validate the weight dict returned by `_compute`.

        Long-only contract (allow_short=False, v1 default):
          - Each weight in [0, 1].
          - sum(weights) <= 1.0 (leftover sits in cash).

        Long/short contract (allow_short=True, v1.1 M10):
          - Each weight in [-1, 1]. Negative = short position.
          - sum(|weights|) <= 1.0 (gross exposure cap).

        This is the single chokepoint for rule #4 in critical-rules.md. Any
        change here must update that file in lockstep.
        """
        if not isinstance(weights, dict):
            raise TypeError(f"weights must be dict, got {type(weights).__name__}")

        lower, upper = (-1.0, 1.0) if self.allow_short else (0.0, 1.0)
        gross = 0.0
        for ticker, w in weights.items():
            if not isinstance(ticker, str) or not ticker:
                raise ValueError(f"ticker keys must be non-empty strings, got {ticker!r}")
            if not isinstance(w, (int, float)) or isinstance(w, bool):
                raise TypeError(f"weight for {ticker!r} must be numeric, got {type(w).__name__}")
            if math.isnan(w) or math.isinf(w):
                raise ValueError(f"weight for {ticker!r} is not finite: {w}")
            if w < lower or w > upper:
                raise ValueError(
                    f"weight for {ticker!r} must be in [{lower}, {upper}], got {w}"
                )
            gross += abs(w) if self.allow_short else w
        if gross > 1.0 + _WEIGHT_SUM_EPS:
            cap_name = "sum(|weights|)" if self.allow_short else "sum(weights)"
            raise ValueError(f"{cap_name} = {gross}, must be <= 1.0")


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


class ValueScreen(Strategy):
    """Pick the top-N tickers by earnings yield (EPS / price), equal weight.

    Earnings yield uses the most recent quarterly EPS as a proxy — NOT TTM.
    For ranking purposes within a fixed quarterly cadence this is consistent
    across tickers; if you want TTM, sum the last 4 quarters before dividing.
    Limitation flagged in build notes.

    Look-ahead safety: engine pre-filters fundamentals by `filing_date <=
    current_date` and the base class double-checks. We always use the most
    recently *filed* report, never the most recent fiscal period.
    """

    requires_fundamentals = ("income_stmt",)

    def __init__(self, tickers: list[str], top_n: int = 3):
        if not isinstance(tickers, list):
            raise TypeError(f"tickers must be a list, got {type(tickers).__name__}")
        if any(not isinstance(t, str) or not t for t in tickers):
            raise ValueError("every ticker must be a non-empty string")
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n must be a positive int")
        if top_n > len(tickers):
            raise ValueError(
                f"top_n ({top_n}) cannot exceed universe size ({len(tickers)})"
            )

        self.tickers = [t.upper() for t in tickers]
        self.top_n = top_n

    def _compute(self, current_date, price_data, fundamentals):
        if not fundamentals or "income_stmt" not in fundamentals:
            return {}

        income = fundamentals["income_stmt"]
        if income.empty or price_data.empty:
            return {}

        # Most recent close per ticker — use the last row up through today.
        latest_close = (
            price_data.sort_values(["ticker", "date"])
            .groupby("ticker")["close"]
            .last()
        )

        # Most recent filing per ticker that satisfies filing_date <= current_date.
        # Engine already slices by filing_date upstream, so all rows here are eligible.
        latest_filing = (
            income.sort_values(["ticker", "filing_date"])
            .groupby("ticker")
            .tail(1)
            .set_index("ticker")
        )

        yields: dict[str, float] = {}
        for ticker in self.tickers:
            if ticker not in latest_filing.index or ticker not in latest_close.index:
                continue
            eps = latest_filing.loc[ticker, "eps"]
            price = latest_close.loc[ticker]
            if pd.isna(eps) or pd.isna(price) or price <= 0 or eps <= 0:
                continue  # skip negative-earnings or missing-data names
            yields[ticker] = float(eps) / float(price)

        if not yields:
            return {}

        # Pick top N by yield, equal weight.
        top = sorted(yields, key=yields.get, reverse=True)[: self.top_n]
        w = 1.0 / self.top_n
        return {t: w for t in top}


class LimitBuyTheDip(Strategy):
    """Reference M9 strategy. Equal-weight buy-and-hold, but entries happen
    via day-only buy limit orders placed `drop_pct` below the previous close.

    Once a ticker has any position (the limit filled at least once), no more
    limits are emitted for it — the strategy just rides. `generate_signals`
    returns an empty weight dict so the engine performs no market trades.

    Requires `manages_positions=True` so the engine does NOT auto-exit the
    limit-acquired holdings on rebalance days (the default rule would treat
    an empty weight dict as "100% cash" and unwind everything).
    """

    manages_positions = True

    def __init__(
        self,
        tickers: list[str],
        drop_pct: float = 0.02,
        shares_per_order: int = 100,
    ):
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("tickers must be a non-empty list")
        if any(not isinstance(t, str) or not t for t in tickers):
            raise ValueError("every ticker must be a non-empty string")
        if not isinstance(drop_pct, (int, float)) or drop_pct <= 0 or drop_pct >= 1:
            raise ValueError("drop_pct must be in (0, 1)")
        if not isinstance(shares_per_order, int) or shares_per_order <= 0:
            raise ValueError("shares_per_order must be a positive int")

        self.tickers = [t.upper() for t in tickers]
        self.drop_pct = float(drop_pct)
        self.shares_per_order = shares_per_order

    def _compute(self, current_date, price_data, fundamentals):
        # No market-order rebalancing; entries come purely from limits.
        return {}

    def generate_orders(self, current_date, price_data, fundamentals=None, holdings=None):
        if price_data.empty:
            return []
        held = holdings or {}
        orders: list[LimitOrder] = []
        # Use the most recent close per ticker through current_date as the anchor.
        latest_close = (
            price_data.sort_values(["ticker", "date"])
            .groupby("ticker")["close"]
            .last()
        )
        for ticker in self.tickers:
            if held.get(ticker, 0) > 0:
                continue  # already entered — ride
            if ticker not in latest_close.index:
                continue
            anchor = float(latest_close.loc[ticker])
            if anchor <= 0:
                continue
            limit_price = round(anchor * (1.0 - self.drop_pct), 4)
            orders.append(LimitOrder(
                ticker=ticker,
                side="buy",
                shares=self.shares_per_order,
                limit_price=limit_price,
                lifetime="day",
            ))
        return orders


class LongShortBarbell(Strategy):
    """M10 reference strategy. Equal-weight `longs` at +1/N and `shorts` at -1/N.

    N = len(longs) + len(shorts). sum(|w|) = 1.0 exactly — sits at the gross
    exposure cap. Requires `BacktestConfig(allow_short=True)`; the engine
    writes the flag onto the strategy at run() start.
    """

    def __init__(self, longs: list[str], shorts: list[str]):
        if not isinstance(longs, list) or not isinstance(shorts, list):
            raise TypeError("longs and shorts must be lists")
        if any(not isinstance(t, str) or not t for t in (*longs, *shorts)):
            raise ValueError("every ticker must be a non-empty string")
        if not longs and not shorts:
            raise ValueError("at least one of longs / shorts must be non-empty")

        upper_longs = [t.upper() for t in longs]
        upper_shorts = [t.upper() for t in shorts]
        overlap = set(upper_longs) & set(upper_shorts)
        if overlap:
            raise ValueError(f"ticker cannot be both long and short: {sorted(overlap)}")

        self.longs = upper_longs
        self.shorts = upper_shorts

    def _compute(self, current_date, price_data, fundamentals):
        n = len(self.longs) + len(self.shorts)
        w = 1.0 / n
        weights = {t: w for t in self.longs}
        for t in self.shorts:
            weights[t] = -w
        return weights
