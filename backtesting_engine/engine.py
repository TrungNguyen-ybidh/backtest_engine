"""Engine — orchestrator that runs the daily loop.

Wires DataInterface, Strategy, and Broker. Engine does NOT execute trades
itself (that's Broker) and does NOT compute signals (that's Strategy).

Daily loop contract:
- Strategy sees price history through current_date (inclusive). No look-ahead.
- Targets are share counts computed from {ticker: weight} × portfolio_value / price,
  floor-rounded to whole shares (v1 has no fractionals).
- Sells execute before buys so freed cash funds buys.
- Trades are date-tagged on broker.trades[-1] post-execution (broker is date-agnostic).
- Mark-to-market runs every trading day. Rebalance trades only on days where
  `_is_rebalance_day()` is True (M8). Default frequency 'daily' = trade every day,
  matching v1 behavior exactly.
"""

import math
from datetime import date

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
        # Last date on which we ran the strategy and traded. None until the
        # first rebalance fires — the very first trading day always rebalances.
        self._last_rebalance_date: date | None = None

    def run(self) -> None:
        # M10: propagate allow_short to broker + strategy so all three agree on
        # the same contract. Single source of truth = config.
        self.broker.allow_short = self.config.allow_short
        self.strategy.allow_short = self.config.allow_short

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

        # Pre-fetch any fundamental statements the strategy needs. One DB hit per
        # statement for the whole window; we slice by filing_date per day in memory.
        # Pass `as_of=end_date` so we get every filing up to the backtest end.
        all_fundamentals: dict[str, pd.DataFrame] = {}
        for statement in getattr(self.strategy, "requires_fundamentals", ()):
            all_fundamentals[statement] = self.data.get_fundamentals(
                self.config.tickers, statement, as_of=self.config.end_date
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
            # OHLC dict for limit-order fill checks. open/high/low come straight
            # from DataInterface and are split/dividend-adjusted alongside close.
            today_ohlc: dict[str, dict] = {
                row.ticker: {
                    "open": row.open, "high": row.high,
                    "low": row.low, "close": row.close,
                }
                for row in today_rows.itertuples(index=False)
            }

            # M9: process pending limit orders first. Fills modify cash/holdings
            # BEFORE the MTM step so pv reflects the post-fill state used by
            # rebalance sizing. Date-tag any new trades the broker just produced.
            n_trades_before = len(self.broker.trades)
            self.broker.process_open_orders(current_date, today_ohlc)
            for tr in self.broker.trades[n_trades_before:]:
                tr["date"] = current_date

            # Mark-to-market every trading day, regardless of rebalance frequency.
            # broker.total_value walks both long and short books and raises
            # KeyError if any held ticker is missing — we pass current_prices
            # entirely; unused entries are harmless.
            try:
                pv = self.broker.total_value(current_prices)
            except KeyError as e:
                # Still expire stale orders so they don't accumulate during gaps.
                self.broker.expire_orders(current_date)
                self._record_day(current_date, current_prices, note=f"missing_price:{e}")
                continue

            # M10: Reg T maintenance margin check on the post-MTM state. If
            # equity / total exposure < 25%, liquidate ALL shorts at today's
            # close (broker forces the cover even if cash goes negative — the
            # account is being unwound). Re-MTM after liquidation.
            if (
                self.config.allow_short
                and self.broker.short_holdings
                and not self.broker.maintenance_margin_ok(current_prices)
            ):
                self.broker.liquidate_all_shorts(current_prices, current_date)
                # Re-MTM with shorts gone (broker.total_value handles empty book).
                pv = self.broker.total_value(current_prices)

            # Build price/fundamentals slices for the strategy hooks.
            # History through today inclusive — Strategy sees today's close.
            price_hist = all_prices[all_prices["date_only"] <= current_date].drop(
                columns=["date_only"]
            )
            # Slice fundamentals by filing_date <= current_date. Critical rule #2:
            # `filing_date` (when public) gates look-ahead, not period-end `date`.
            fundamentals_today: dict[str, pd.DataFrame] | None = None
            if all_fundamentals:
                fundamentals_today = {}
                for name, df in all_fundamentals.items():
                    if df.empty or "filing_date" not in df.columns:
                        fundamentals_today[name] = df
                        continue
                    mask = df["filing_date"].dt.date <= current_date
                    fundamentals_today[name] = df[mask]

            # Rebalance only on days where the configured frequency fires.
            # The very first trading day always rebalances so the strategy
            # gets a chance to enter its initial positions.
            if self._is_rebalance_day(current_date):
                weights = self.strategy.generate_signals(
                    current_date, price_hist, fundamentals=fundamentals_today
                )

                # Signed target shares per ticker. Negative = short target.
                # Tickers absent from current_prices today can't be traded.
                targets: dict[str, int] = {}
                for ticker, weight in weights.items():
                    if ticker not in current_prices:
                        continue
                    price = current_prices[ticker]
                    if price <= 0:
                        continue
                    magnitude = math.floor(abs(weight) * pv / price)
                    targets[ticker] = -magnitude if weight < 0 else magnitude

                # Universe for delta computation. `manages_positions=True`
                # tells the engine "don't touch holdings I didn't target".
                manages = getattr(self.strategy, "manages_positions", False)
                if manages:
                    delta_universe: set[str] = set(targets)
                else:
                    delta_universe = (
                        set(targets)
                        | set(self.broker.long_holdings)
                        | set(self.broker.short_holdings)
                    )

                # Route each ticker into atomic close/open actions. A flip
                # (long→short or short→long) emits TWO actions in sequence.
                closes: list[tuple[str, str, int]] = []  # (ticker, action, qty)
                opens: list[tuple[str, str, int]] = []
                for ticker in delta_universe:
                    target_signed = targets.get(ticker, 0)
                    for action, qty in self._route_actions(ticker, target_signed):
                        if action in ("SELL", "COVER"):
                            closes.append((ticker, action, qty))
                        else:
                            opens.append((ticker, action, qty))

                # Closes/covers first (frees cash), then opens. Alphabetical
                # within each bucket for determinism.
                closes.sort()
                opens.sort()
                for ticker, action, qty in closes:
                    price_now = current_prices[ticker]
                    if action == "SELL":
                        self._execute_with_tag(ticker, -qty, price_now, current_date)
                    else:  # COVER
                        self._cover_with_retry(ticker, qty, price_now, current_date)
                for ticker, action, qty in opens:
                    price_now = current_prices[ticker]
                    if action == "BUY":
                        self._buy_with_retry(ticker, qty, price_now, current_date)
                    else:  # SHORT_OPEN
                        self._short_open_with_retry(
                            ticker, qty, price_now, current_prices, current_date
                        )

                self._last_rebalance_date = current_date

            # M9: every day, let the strategy emit new limit orders. Coexists
            # with rebalance — a strategy may use both channels. `holdings`
            # snapshot is long-side only (limits are long-only in v1.1).
            new_orders = self.strategy.generate_orders(
                current_date,
                price_hist,
                fundamentals=fundamentals_today,
                holdings=dict(self.broker.long_holdings),
            )
            for order in new_orders:
                self.broker.place_limit_order(order, placed_on=current_date)

            # End of day: expire orders past their lifetime.
            self.broker.expire_orders(current_date)

            self._record_day(current_date, current_prices)

    def _is_rebalance_day(self, current_date: date) -> bool:
        """Return True if today is a rebalance day under the configured frequency.

        First trading day always rebalances so the strategy can enter positions.
        After that, the boundary rules are:
          - daily      → every day
          - weekly     → ISO (year, week) changed since last rebalance
          - monthly    → (year, month) changed
          - quarterly  → (year, quarter) changed (quarter = (month-1)//3)
          - yearly     → year changed
        Using "boundary changed" rather than "first calendar day of period"
        is robust to holidays — e.g., Jan 1 closed → first trading day is Jan 2
        and that's still treated as the new-year rebalance.
        """
        if self._last_rebalance_date is None:
            return True

        freq = self.config.rebalance_frequency
        last = self._last_rebalance_date

        if freq == "daily":
            return True
        if freq == "weekly":
            return current_date.isocalendar()[:2] != last.isocalendar()[:2]
        if freq == "monthly":
            return (current_date.year, current_date.month) != (last.year, last.month)
        if freq == "quarterly":
            cur_q = (current_date.month - 1) // 3
            last_q = (last.month - 1) // 3
            return (current_date.year, cur_q) != (last.year, last_q)
        if freq == "yearly":
            return current_date.year != last.year
        # Config validation should have rejected anything else; defensive raise.
        raise ValueError(f"unknown rebalance_frequency: {freq!r}")

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

    def _cover_with_retry(
        self, ticker: str, shares: int, price: float, current_date
    ) -> None:
        """Cover with shrink-and-retry on insufficient-cash rejection.

        Mirrors `_buy_with_retry` — covers are cash-consuming like buys, so the
        same edge case (floor-rounding tipping cost over cash) applies. Note
        this is NOT the margin-call path; that goes through
        `broker.liquidate_all_shorts` with `force=True`.
        """
        while shares > 0:
            try:
                self.broker.execute_short_cover(ticker, shares, price)
                self.broker.trades[-1]["date"] = current_date
                return
            except ValueError:
                shares -= 1

    def _short_open_with_retry(
        self,
        ticker: str,
        shares: int,
        price: float,
        current_prices: dict[str, float],
        current_date,
    ) -> None:
        """Open a short with shrink-and-retry on initial-margin rejection.

        If the prospective gross short exposure would tip past 50% of PV,
        Broker raises ValueError. Shrinking shares by 1 and retrying yields
        the largest short we can afford to open under the cap.
        """
        while shares > 0:
            try:
                self.broker.execute_short_open(ticker, shares, price, current_prices)
                self.broker.trades[-1]["date"] = current_date
                return
            except ValueError:
                shares -= 1

    def _route_actions(
        self, ticker: str, target_signed: int
    ) -> list[tuple[str, int]]:
        """Map (current book state + signed target) → list of atomic actions.

        Action names: 'BUY' / 'SELL' (long book), 'SHORT_OPEN' / 'COVER' (short book).
        A flip (long → short or short → long) emits two actions in order.
        Quantities returned are always positive integers.

        9 cases (3 current × 3 target):
          - current_long  >0, target  >0: adjust long by delta sign
          - current_long  >0, target  =0: SELL all
          - current_long  >0, target  <0: SELL all + SHORT_OPEN |target|
          - current_short >0, target  <0: adjust short by delta sign
          - current_short >0, target  =0: COVER all
          - current_short >0, target  >0: COVER all + BUY target
          - flat,            target  >0: BUY
          - flat,            target  <0: SHORT_OPEN
          - flat,            target  =0: noop
        """
        current_long = self.broker.long_holdings.get(ticker, 0)
        current_short = self.broker.short_holdings.get(ticker, 0)
        actions: list[tuple[str, int]] = []

        if current_long > 0:
            if target_signed >= 0:
                delta = target_signed - current_long
                if delta > 0:
                    actions.append(("BUY", delta))
                elif delta < 0:
                    actions.append(("SELL", -delta))
            else:  # flip long → short
                actions.append(("SELL", current_long))
                actions.append(("SHORT_OPEN", -target_signed))
        elif current_short > 0:
            if target_signed <= 0:
                new_short = -target_signed  # positive
                delta = new_short - current_short
                if delta > 0:
                    actions.append(("SHORT_OPEN", delta))
                elif delta < 0:
                    actions.append(("COVER", -delta))
            else:  # flip short → long
                actions.append(("COVER", current_short))
                actions.append(("BUY", target_signed))
        else:  # flat
            if target_signed > 0:
                actions.append(("BUY", target_signed))
            elif target_signed < 0:
                actions.append(("SHORT_OPEN", -target_signed))

        return actions

    def _record_day(
        self,
        current_date,
        current_prices: dict[str, float],
        note: str | None = None,
    ) -> None:
        """Append one history row. Tolerates missing prices for held tickers.

        `holdings_value` is NET = long MTM − short MTM. For pure-long runs
        (no shorts ever opened) this equals the v1 behavior exactly.
        """
        try:
            total_value = self.broker.total_value(current_prices)
        except KeyError:
            total_value = float("nan")

        bench = self.config.benchmark
        benchmark_price = current_prices.get(bench) if bench else None

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
            "holdings": dict(self.broker.long_holdings),
            "short_holdings": dict(self.broker.short_holdings),
            "benchmark_price": benchmark_price,
            "n_trades": n_trades_today,
        }
        if note is not None:
            row["note"] = note
        self.history.append(row)
