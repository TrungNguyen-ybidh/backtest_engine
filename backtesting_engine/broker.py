"""Broker — simulates trade execution.

Long-only, market-orders-only for v1. Limit/stop orders land in v1.1.

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
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be positive, got {initial_cash}")

        self.cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.holdings: dict[str, int] = {}

        # Trade log — engine tags with date if needed. Broker is date-agnostic.
        self.trades: list[dict] = []
        self.total_commission: float = 0.0

    def execute_order(self, ticker: str, shares: int, price: float) -> None:
        """Execute a market order. Positive shares = buy, negative = sell.

        Raises ValueError on insufficient cash (buy) or insufficient holdings (sell).
        Long-only: cannot sell more than currently held.
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
            self.holdings[ticker] = self.holdings.get(ticker, 0) + shares
            side = "buy"
        else:
            sell_qty = -shares  # positive
            held = self.holdings.get(ticker, 0)
            if sell_qty > held:
                raise ValueError(
                    f"insufficient holdings to sell {sell_qty} {ticker}: hold {held} (long-only)"
                )
            proceeds = sell_qty * fill_price - commission
            self.cash += proceeds
            new_qty = held - sell_qty
            if new_qty == 0:
                del self.holdings[ticker]
            else:
                self.holdings[ticker] = new_qty
            side = "sell"

        self.total_commission += commission
        self.trades.append({
            "ticker": ticker,
            "shares": shares,
            "fill_price": fill_price,
            "commission": commission,
            "side": side,
        })

    def total_value(self, prices: dict[str, float]) -> float:
        """Return cash + mark-to-market value of all holdings.

        Raises KeyError if a held ticker is missing from `prices` — silent zero
        would mask data gaps and break the cash-conservation invariant check.
        """
        holdings_value = 0.0
        for ticker, shares in self.holdings.items():
            if ticker not in prices:
                raise KeyError(f"price missing for held ticker {ticker!r}")
            holdings_value += shares * prices[ticker]
        return self.cash + holdings_value
