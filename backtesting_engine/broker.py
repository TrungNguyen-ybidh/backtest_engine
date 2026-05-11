"""Broker — simulates trade execution.

Market orders execute immediately at the requested price (after slippage).
Limit orders (v1.1, M9) are placed onto `open_orders` and filled on subsequent
days when the day's [low, high] range touches the limit price. Short selling
(v1.1, M10) is gated on `allow_short=True` and enforces Reg T margin (initial
50%, maintenance 25%).

Invariants:
  - allow_short=False:  cash + sum(long_shares * price)                               == total_value
  - allow_short=True:   cash + sum(long_shares * price) - sum(short_shares * price)   == total_value

A ticker may appear in `long_holdings` OR `short_holdings`, never both. The
engine sequences orders so a long→short flip executes as (sell long, then
short open) — two atomic broker calls.
"""

from datetime import date

from .models import CommissionModel, LimitOrder, SlippageModel


# Reg T defaults (M10). Initial margin: gross short exposure capped at 50%
# of portfolio value at order entry. Maintenance: equity / total exposure
# must stay >= 25% on every mark-to-market.
_INITIAL_MARGIN_RATIO = 0.50
_MAINTENANCE_MARGIN_RATIO = 0.25


class Broker:
    def __init__(
        self,
        initial_cash: float,
        commission: CommissionModel,
        slippage: SlippageModel,
        allow_short: bool = False,
    ):
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")

        self.cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.allow_short = allow_short

        # Long-side book. v1 callers used `broker.holdings`; that name is now
        # a read-only property alias so existing code keeps working.
        self.long_holdings: dict[str, int] = {}
        # M10 short-side book. Always positive integers (covered ⇒ entry removed).
        # A ticker is never simultaneously in both books.
        self.short_holdings: dict[str, int] = {}

        # Trade log — engine tags with date if needed. Broker is date-agnostic.
        self.trades: list[dict] = []
        self.total_commission: float = 0.0

        # M9: book of pending limit orders.
        self.open_orders: list[LimitOrder] = []

    # ---- v1 compat alias -------------------------------------------------
    @property
    def holdings(self) -> dict[str, int]:
        """Alias for `long_holdings` so v1 code (`broker.holdings`) keeps working.

        Read-only by convention — mutating this dict mutates `long_holdings`,
        which is fine because they're the same object, but external callers
        should not mutate broker state directly anyway.
        """
        return self.long_holdings

    # ---------------------------------------------------------------------
    # Long-side execution (unchanged from v1, with renamed dict access)

    def execute_order(self, ticker: str, shares: int, price: float) -> None:
        """Execute a market order on the long-side book.

        Positive shares = buy, negative = sell. Raises ValueError on
        insufficient cash (buy) or insufficient holdings (sell). Short positions
        go through `execute_short_open` / `execute_short_cover` — execute_order
        never touches `short_holdings`.
        """
        if not isinstance(shares, int):
            raise TypeError(f"shares must be int, got {type(shares).__name__}")
        if shares == 0:
            return
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")

        fill_price = self.slippage.adjust(price, shares)
        commission = self.commission.calculate(shares, price)

        if shares > 0:
            cost = shares * fill_price + commission
            if cost > self.cash:
                raise ValueError(
                    f"insufficient cash for buy: need {cost:.2f}, have {self.cash:.2f}"
                )
            self.cash -= cost
            self.long_holdings[ticker] = self.long_holdings.get(ticker, 0) + shares
            side = "buy"
        else:
            sell_qty = -shares  # positive
            held = self.long_holdings.get(ticker, 0)
            if sell_qty > held:
                raise ValueError(
                    f"insufficient holdings to sell {sell_qty} {ticker}: hold {held}"
                )
            proceeds = sell_qty * fill_price - commission
            self.cash += proceeds
            new_qty = held - sell_qty
            if new_qty == 0:
                del self.long_holdings[ticker]
            else:
                self.long_holdings[ticker] = new_qty
            side = "sell"

        self.total_commission += commission
        self.trades.append({
            "ticker": ticker,
            "shares": shares,
            "fill_price": fill_price,
            "commission": commission,
            "side": side,
        })

    # ---------------------------------------------------------------------
    # Short-side execution (M10)

    def execute_short_open(
        self,
        ticker: str,
        shares: int,
        price: float,
        current_prices: dict[str, float],
    ) -> None:
        """Open a short position. Cash increases by `shares × fill - commission`.

        `current_prices` is needed to compute the prospective gross short
        exposure for the Reg T initial margin check. Engine has it readily.

        Raises:
          ValueError if `allow_short` is False.
          ValueError if the ticker is currently held long (must close long first).
          ValueError if the resulting gross short exposure would exceed 50%
            of the prospective portfolio value (Reg T initial margin).
        """
        if not self.allow_short:
            raise ValueError("short selling disabled (allow_short=False)")
        if not isinstance(shares, int) or isinstance(shares, bool):
            raise TypeError(f"shares must be int, got {type(shares).__name__}")
        if shares <= 0:
            raise ValueError(f"shares must be positive, got {shares}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if self.long_holdings.get(ticker, 0) > 0:
            raise ValueError(
                f"{ticker} is currently held long ({self.long_holdings[ticker]}); "
                "close the long position before shorting"
            )

        # Selling-direction slippage (negative shares argument to the model).
        fill_price = self.slippage.adjust(price, -shares)
        commission = self.commission.calculate(shares, price)
        proceeds = shares * fill_price - commission

        # Prospective state for the margin check (post-trade values).
        prospective_short = dict(self.short_holdings)
        prospective_short[ticker] = prospective_short.get(ticker, 0) + shares
        prospective_cash = self.cash + proceeds

        gross_short_value = 0.0
        for t, qty in prospective_short.items():
            if t not in current_prices:
                raise KeyError(f"price missing for ticker {t!r} (needed for margin check)")
            gross_short_value += qty * current_prices[t]

        long_value = 0.0
        for t, qty in self.long_holdings.items():
            if t not in current_prices:
                raise KeyError(f"price missing for held ticker {t!r}")
            long_value += qty * current_prices[t]

        pv = prospective_cash + long_value - gross_short_value
        if pv <= 0 or gross_short_value > _INITIAL_MARGIN_RATIO * pv:
            raise ValueError(
                f"initial margin breached: gross short {gross_short_value:.2f} would be "
                f"{gross_short_value / pv * 100:.0f}% of PV {pv:.2f} (limit {int(_INITIAL_MARGIN_RATIO*100)}%)"
            )

        # Commit.
        self.cash = prospective_cash
        self.short_holdings = prospective_short
        self.total_commission += commission
        self.trades.append({
            "ticker": ticker,
            "shares": -shares,  # negative in log = exit / short open from a long-centric view
            "fill_price": fill_price,
            "commission": commission,
            "side": "short",
        })

    def execute_short_cover(
        self,
        ticker: str,
        shares: int,
        price: float,
        force: bool = False,
    ) -> None:
        """Buy back to close (or reduce) a short position.

        `force=True` is used by `liquidate_all_shorts` during margin calls —
        the cover proceeds even if cash goes negative (the account is being
        forcibly unwound; user chose this behavior per the M10 spec).

        Raises:
          ValueError if `allow_short` is False.
          ValueError if there's no short position to cover.
          ValueError if `shares` exceeds the open short.
          ValueError on insufficient cash (only when force=False).
        """
        if not self.allow_short:
            raise ValueError("short selling disabled (allow_short=False)")
        if not isinstance(shares, int) or isinstance(shares, bool):
            raise TypeError(f"shares must be int, got {type(shares).__name__}")
        if shares <= 0:
            raise ValueError(f"shares must be positive, got {shares}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")

        open_short = self.short_holdings.get(ticker, 0)
        if open_short <= 0:
            raise ValueError(f"no open short for {ticker!r} to cover")
        if shares > open_short:
            raise ValueError(
                f"cover {shares} exceeds open short {open_short} for {ticker}"
            )

        # Buying-direction slippage.
        fill_price = self.slippage.adjust(price, shares)
        commission = self.commission.calculate(shares, price)
        cost = shares * fill_price + commission

        if not force and cost > self.cash:
            raise ValueError(
                f"insufficient cash to cover: need {cost:.2f}, have {self.cash:.2f}"
            )

        self.cash -= cost  # may go negative if force=True
        new_qty = open_short - shares
        if new_qty == 0:
            del self.short_holdings[ticker]
        else:
            self.short_holdings[ticker] = new_qty

        self.total_commission += commission
        self.trades.append({
            "ticker": ticker,
            "shares": shares,  # positive in log = entry from long-centric view (we're buying)
            "fill_price": fill_price,
            "commission": commission,
            "side": "cover",
        })

    # ---------------------------------------------------------------------
    # Mark-to-market + margin

    def total_value(self, prices: dict[str, float]) -> float:
        """Return cash + long MTM - short MTM.

        Raises KeyError if any held ticker (long OR short) is missing from
        `prices` — silent zero would mask data gaps and break the extended
        cash-conservation invariant.
        """
        long_value = 0.0
        for ticker, shares in self.long_holdings.items():
            if ticker not in prices:
                raise KeyError(f"price missing for held long ticker {ticker!r}")
            long_value += shares * prices[ticker]

        short_value = 0.0
        for ticker, shares in self.short_holdings.items():
            if ticker not in prices:
                raise KeyError(f"price missing for held short ticker {ticker!r}")
            short_value += shares * prices[ticker]

        return self.cash + long_value - short_value

    def maintenance_margin_ok(self, prices: dict[str, float]) -> bool:
        """Reg T maintenance: equity / total exposure >= 25%.

        Total exposure = |sum(long × price)| + |sum(short × price)|. No shorts
        and no longs → no margin risk → True. Equity at or below zero with any
        exposure → False (account is busted, force liquidate).
        """
        if not self.short_holdings:
            return True  # no margin risk without shorts

        long_value = sum(qty * prices[t] for t, qty in self.long_holdings.items())
        short_value = sum(qty * prices[t] for t, qty in self.short_holdings.items())
        exposure = abs(long_value) + abs(short_value)
        if exposure == 0:
            return True
        equity = self.cash + long_value - short_value
        return equity / exposure >= _MAINTENANCE_MARGIN_RATIO

    def liquidate_all_shorts(
        self,
        prices: dict[str, float],
        current_date: date,
    ) -> list[dict]:
        """Forced cover of every open short at the supplied prices.

        Used by the engine when `maintenance_margin_ok` returns False. Each
        forced trade is tagged `note='margin_call'` and the date is set by
        the broker (caller convenience — engine doesn't need to re-tag).
        Cash may go negative — the account is being unwound regardless.

        Returns the list of new trade dicts added to `self.trades` (so the
        engine can do any post-liquidation bookkeeping).
        """
        forced: list[dict] = []
        # Snapshot keys — execute_short_cover mutates short_holdings.
        for ticker in list(self.short_holdings.keys()):
            qty = self.short_holdings[ticker]
            self.execute_short_cover(ticker, qty, prices[ticker], force=True)
            # Tag the trade we just appended.
            tr = self.trades[-1]
            tr["date"] = current_date
            tr["note"] = "margin_call"
            forced.append(tr)
        return forced

    # ---------------------------------------------------------------------
    # M9 limit-order book (long-only in v1.1; short-side limits → v1.x)

    def place_limit_order(self, order: LimitOrder, placed_on: date) -> None:
        """Add a limit order to the open book. Does NOT fill immediately.

        Cash is NOT reserved at placement (documented caveat — IB reserves;
        we don't). If at fill time the broker can't afford the buy, the fill
        silently fails and the order stays in the book until expiry.
        """
        if order.placed_on is not None:
            raise ValueError("LimitOrder has already been placed")
        order.placed_on = placed_on
        self.open_orders.append(order)

    def process_open_orders(
        self,
        current_date: date,
        ohlc: dict[str, dict],
    ) -> None:
        """Try to fill each open limit order against today's price range.

        `ohlc` is `{ticker: {'open', 'high', 'low', 'close'}}`. Orders are
        eligible only if `current_date > placed_on` (so an order placed today
        cannot fill today — see LimitOrder docstring on look-ahead).

        Fill rules:
          - buy:  low  <= limit_price → fill at limit_price.
          - sell: high >= limit_price → fill at limit_price.
        Sells fill before buys (consistent with market-order convention).
        On insufficient-cash buys, the fill is skipped and the order stays
        in the book; lifetime expiration is handled by expire_orders().
        """
        # Sort: sells first, buys second; preserve placement order within each.
        sells = [o for o in self.open_orders if o.side == "sell"]
        buys = [o for o in self.open_orders if o.side == "buy"]
        filled: list[LimitOrder] = []

        for order in sells + buys:
            if order.placed_on is None or current_date <= order.placed_on:
                continue  # not eligible until the day AFTER placement
            bars = ohlc.get(order.ticker)
            if not bars:
                continue  # no price today (delisting/halt); leave on book

            high = bars.get("high")
            low = bars.get("low")
            if high is None or low is None:
                continue

            if order.side == "buy" and low <= order.limit_price:
                if self._fill_limit(order, order.limit_price):
                    filled.append(order)
            elif order.side == "sell" and high >= order.limit_price:
                if self._fill_limit(order, order.limit_price):
                    filled.append(order)

        for o in filled:
            self.open_orders.remove(o)

    def _fill_limit(self, order: LimitOrder, fill_price: float) -> bool:
        """Execute a limit order at `fill_price` (no slippage, commission charged).

        Returns True if filled, False if rejected (e.g., insufficient cash on
        a buy, insufficient holdings on a sell — long-only rules still apply).
        Mirrors execute_order's accounting but tags trades with side='buy_limit'
        or 'sell_limit' so the trade log distinguishes the channel.
        """
        commission = self.commission.calculate(order.shares, fill_price)

        if order.side == "buy":
            cost = order.shares * fill_price + commission
            if cost > self.cash:
                return False  # leave on book; may fill later or expire
            self.cash -= cost
            self.long_holdings[order.ticker] = (
                self.long_holdings.get(order.ticker, 0) + order.shares
            )
            recorded_shares = order.shares
            side_label = "buy_limit"
        else:  # sell
            held = self.long_holdings.get(order.ticker, 0)
            if order.shares > held:
                return False  # long-only; can't oversell
            proceeds = order.shares * fill_price - commission
            self.cash += proceeds
            new_qty = held - order.shares
            if new_qty == 0:
                del self.long_holdings[order.ticker]
            else:
                self.long_holdings[order.ticker] = new_qty
            recorded_shares = -order.shares
            side_label = "sell_limit"

        self.total_commission += commission
        self.trades.append({
            "ticker": order.ticker,
            "shares": recorded_shares,
            "fill_price": fill_price,
            "commission": commission,
            "side": side_label,
        })
        return True

    def expire_orders(self, current_date: date) -> None:
        """Drop orders past their lifetime. Called by Engine at end of day.

          - 'day':         drop if current_date > placed_on.
          - 'gtc_expire':  drop if current_date >= expiration_date.
          - 'gtc':         never auto-expires.
        """
        remaining: list[LimitOrder] = []
        for o in self.open_orders:
            placed = o.placed_on
            if o.lifetime == "day":
                if placed is not None and current_date > placed:
                    continue  # expired
            elif o.lifetime == "gtc_expire":
                if o.expiration_date is not None and current_date >= o.expiration_date:
                    continue  # expired
            # 'gtc' never expires.
            remaining.append(o)
        self.open_orders = remaining
