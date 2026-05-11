"""Engine — orchestrator that runs the daily loop.

Wires DataInterface, Strategy, and Broker. Engine does NOT execute trades
itself (that's Broker) and does NOT compute signals (that's Strategy).

Daily loop contract:
- Strategy sees price history through current_date (inclusive). No look-ahead.
- Targets are share counts computed from {ticker: weight} × portfolio_value / price,
  floor-rounded to whole shares (v1 has no fractionals).
- Sells execute before buys so freed cash funds buys.
- Trades are date-tagged on broker.trades[-1] post-execution (broker is date-agnostic).
"""

import math

import pandas as pd

from .broker import Broker
from .config import BacktestConfig
from .data_interface import DataInterface
from .strategy import Strategy


class Engine:
    def __init__(
        self,
        config: BacktestConfig,
        data: DataInterface,
        strategy: Strategy,
        broker: Broker,
    ):
        self.config = config
        self.data = data
        self.strategy = strategy
        self.broker = broker
        self.history: list[dict] = []

    def run(self) -> None:
        # One-shot fetch: all prices for the whole window, sliced per-day in memory.
        universe = list(self.config.tickers)
        if self.config.benchmark and self.config.benchmark not in universe:
            universe.append(self.config.benchmark)

        all_prices = self.data.get_prices(
            universe, self.config.start_date, self.config.end_date
        )
        if all_prices.empty:
            raise ValueError(
                "no price data returned — check tickers and date range against the DB"
            )

        # Fail fast on missing benchmark. Silently returning None per-row masks
        # config errors (e.g., benchmark='SPY' but DB only has individual stocks)
        # and breaks Analytics downstream.
        if self.config.benchmark:
            available = set(all_prices["ticker"].unique())
            if self.config.benchmark not in available:
                raise ValueError(
                    f"benchmark {self.config.benchmark!r} not found in price data "
                    f"for {self.config.start_date}..{self.config.end_date}. "
                    f"Set benchmark=None or pick one of: {sorted(available)}"
                )

        # Likewise for configured tickers: warn if any are missing entirely.
        missing_tickers = set(self.config.tickers) - set(all_prices["ticker"].unique())
        if missing_tickers:
            raise ValueError(
                f"tickers with no price data in window: {sorted(missing_tickers)}"
            )

        # Normalize the date column to plain Python date for comparisons with
        # current_date (which Strategy.generate_signals expects as datetime.date).
        all_prices = all_prices.copy()
        all_prices["date_only"] = all_prices["date"].dt.date

        trading_dates = sorted(all_prices["date_only"].unique())

        for current_date in trading_dates:
            today_rows = all_prices[all_prices["date_only"] == current_date]
            current_prices: dict[str, float] = dict(
                zip(today_rows["ticker"], today_rows["close"])
            )

            # History through today inclusive — Strategy sees today's close.
            price_hist = all_prices[all_prices["date_only"] <= current_date].drop(
                columns=["date_only"]
            )

            weights = self.strategy.generate_signals(current_date, price_hist)

            # Mark-to-market using today's close. If a held ticker has no price
            # today (delisting/halt mid-window), skip rebalance and record the gap.
            try:
                pv = self.broker.total_value(
                    {t: current_prices[t] for t in self.broker.holdings}
                )
            except KeyError as e:
                self._record_day(current_date, current_prices, note=f"missing_price:{e}")
                continue

            # Target shares per ticker. Tickers absent from current_prices today
            # can't be traded — skip them rather than guess a price.
            targets: dict[str, int] = {}
            for ticker, weight in weights.items():
                if ticker not in current_prices:
                    continue
                price = current_prices[ticker]
                if price <= 0:
                    continue
                targets[ticker] = math.floor(weight * pv / price)

            # Deltas across the union of (held, targeted). Untargeted holdings
            # implicitly target 0 — fully exit positions the strategy dropped.
            deltas: dict[str, int] = {}
            for ticker in set(targets) | set(self.broker.holdings):
                current_qty = self.broker.holdings.get(ticker, 0)
                target_qty = targets.get(ticker, 0)
                delta = target_qty - current_qty
                if delta != 0:
                    deltas[ticker] = delta

            # Sells first (frees cash), then buys.
            sells = sorted((t, d) for t, d in deltas.items() if d < 0)
            buys = sorted((t, d) for t, d in deltas.items() if d > 0)

            for ticker, delta in sells:
                self._execute_with_tag(ticker, delta, current_prices[ticker], current_date)
            for ticker, delta in buys:
                self._buy_with_retry(ticker, delta, current_prices[ticker], current_date)

            self._record_day(current_date, current_prices)

    def _execute_with_tag(
        self, ticker: str, shares: int, price: float, current_date
    ) -> None:
        """Execute one order and stamp it with current_date on broker.trades."""
        self.broker.execute_order(ticker, shares, price)
        self.broker.trades[-1]["date"] = current_date

    def _buy_with_retry(
        self, ticker: str, shares: int, price: float, current_date
    ) -> None:
        """Buy with shrink-and-retry if slippage+commission tip cost just past cash.

        Floor-rounding target shares can occasionally produce a buy whose
        post-slippage cost slightly exceeds available cash. Rather than embed
        slippage/commission math here (would duplicate Broker logic), we let
        Broker reject and retry with one fewer share.
        """
        while shares > 0:
            try:
                self._execute_with_tag(ticker, shares, price, current_date)
                return
            except ValueError:
                shares -= 1
        # shares == 0 → nothing to buy; silently skip.

    def _record_day(
        self,
        current_date,
        current_prices: dict[str, float],
        note: str | None = None,
    ) -> None:
        """Append one history row. Tolerates missing prices for held tickers."""
        try:
            total_value = self.broker.total_value(
                {t: current_prices[t] for t in self.broker.holdings}
            )
        except KeyError:
            # Reuse previous total_value if we can; otherwise just record cash.
            total_value = float("nan")

        bench = self.config.benchmark
        benchmark_price = current_prices.get(bench) if bench else None

        # Count today's trades by scanning the tail of broker.trades.
        n_trades_today = sum(
            1 for tr in reversed(self.broker.trades)
            if tr.get("date") == current_date
        )

        row = {
            "date": current_date,
            "total_value": total_value,
            "cash": self.broker.cash,
            "holdings_value": (
                total_value - self.broker.cash
                if not pd.isna(total_value) else float("nan")
            ),
            "holdings": dict(self.broker.holdings),
            "benchmark_price": benchmark_price,
            "n_trades": n_trades_today,
        }
        if note is not None:
            row["note"] = note
        self.history.append(row)
