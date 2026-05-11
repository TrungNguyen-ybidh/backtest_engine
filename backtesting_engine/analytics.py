"""Analytics — performance metrics and visualizations.

Inputs:
- history: list of per-day dicts from Engine.run().
- trades: optional reference to broker.trades for trade-level analytics.
- config: optional BacktestConfig for benchmark name + initial capital.

Reports always disclose the caveats from .claude/rules/output-limitations.md
(survivorship bias, adj-close retrospection, no intraday, same-day fills).
"""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 252 trading days per year — standard convention for annualization.
_TRADING_DAYS_PER_YEAR = 252

_CAVEATS = (
    "Caveats (per output-limitations.md):\n"
    "  - Survivorship bias: DB contains only currently-listed companies.\n"
    "  - Adj-close retrospection: prices change with future splits/dividends.\n"
    "  - No intraday data: daily bars only.\n"
    "  - Same-day fills: trades execute at the signal day's close (optimistic)."
)


class Analytics:
    def __init__(
        self,
        history: list[dict],
        trades: Optional[list[dict]] = None,
        config=None,
    ):
        if not history:
            raise ValueError("history is empty — was Engine.run() called?")
        self.history = pd.DataFrame(history)
        self.history["date"] = pd.to_datetime(self.history["date"])
        self.history = self.history.sort_values("date").reset_index(drop=True)
        self.trades = trades or []
        self.config = config

    # ---------- metrics ----------

    def _daily_returns(self) -> pd.Series:
        return self.history["total_value"].pct_change().dropna()

    def _metrics_for_series(self, values: pd.Series, baseline: float | None = None) -> dict:
        """Compute core metrics from an equity series.

        `baseline` anchors the total/annualized return calculation. If provided
        (typically config.initial_capital), it represents the dollar at risk
        BEFORE the first trade; otherwise values.iloc[0] is used. Using initial
        capital is important — values.iloc[0] is the EOD value AFTER the first
        rebalance, so commission/slippage on the entry would otherwise be
        invisible in the reported return.
        """
        values = values.dropna()
        if len(values) < 2:
            return {
                "final": float(values.iloc[-1]) if len(values) else float("nan"),
                "total_return": float("nan"),
                "annual_return": float("nan"),
                "annual_vol": float("nan"),
                "sharpe": float("nan"),
                "max_drawdown": float("nan"),
            }

        initial = float(baseline) if baseline is not None else float(values.iloc[0])
        final = float(values.iloc[-1])
        total_return = final / initial - 1.0

        # Use number of return observations (= len(values) - 1) for annualization.
        n_days = len(values) - 1
        annual_return = (final / initial) ** (_TRADING_DAYS_PER_YEAR / n_days) - 1.0

        daily_ret = values.pct_change().dropna()
        annual_vol = float(daily_ret.std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
        # Sharpe with RFR=0 for v1. Real RFR via get_factors lands in v1.x.
        sharpe = (
            float(daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR))
            if daily_ret.std(ddof=1) > 0
            else float("nan")
        )

        running_peak = values.cummax()
        drawdown = values / running_peak - 1.0
        max_drawdown = float(drawdown.min())

        return {
            "final": final,
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
        }

    def _benchmark_equity(self) -> Optional[pd.Series]:
        """Rescale benchmark prices to start at the same dollar as the strategy."""
        if "benchmark_price" not in self.history.columns:
            return None
        bench = self.history["benchmark_price"].dropna()
        if bench.empty:
            return None
        initial_strategy = float(self.history["total_value"].iloc[0])
        return bench / float(bench.iloc[0]) * initial_strategy

    # ---------- public API ----------

    def print_summary(self) -> None:
        # Anchor returns to initial_capital so commission/slippage on the entry
        # trade is reflected in the headline total return.
        initial = self.config.initial_capital if self.config else None
        m = self._metrics_for_series(self.history["total_value"], baseline=initial)

        start = self.history["date"].iloc[0].date()
        end = self.history["date"].iloc[-1].date()
        n_days = len(self.history)

        total_commission = sum(t.get("commission", 0.0) for t in self.trades)
        n_trades = len(self.trades)

        print("=" * 60)
        print(f"Backtest Summary  {start} -> {end}  ({n_days} trading days)")
        print("=" * 60)
        print(f"  Final value         ${m['final']:>14,.2f}")
        print(f"  Total return        {m['total_return']:>15.2%}")
        print(f"  Annualized return   {m['annual_return']:>15.2%}")
        print(f"  Annualized vol      {m['annual_vol']:>15.2%}")
        print(f"  Sharpe (RFR=0)      {m['sharpe']:>15.2f}")
        print(f"  Max drawdown        {m['max_drawdown']:>15.2%}")
        print(f"  Trades              {n_trades:>15d}")
        print(f"  Total commission    ${total_commission:>14,.2f}")

        bench_eq = self._benchmark_equity()
        if bench_eq is not None and self.config is not None:
            # Benchmark series is already rescaled to start at initial_capital,
            # so anchor metrics to the same baseline for an apples-to-apples view.
            bm = self._metrics_for_series(bench_eq, baseline=initial)
            name = self.config.benchmark or "benchmark"
            print(f"\n  Benchmark: {name}")
            print(f"    Total return      {bm['total_return']:>15.2%}")
            print(f"    Annualized return {bm['annual_return']:>15.2%}")
            print(f"    Annualized vol    {bm['annual_vol']:>15.2%}")
            print(f"    Sharpe            {bm['sharpe']:>15.2f}")
            print(f"    Max drawdown      {bm['max_drawdown']:>15.2%}")
            print(
                f"  Active return       "
                f"{(m['total_return'] - bm['total_return']):>15.2%}"
            )

        print()
        print(_CAVEATS)
        print("=" * 60)

    def plot_equity_curve(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.history["date"],
                y=self.history["total_value"],
                mode="lines",
                name="Strategy",
            )
        )
        bench_eq = self._benchmark_equity()
        if bench_eq is not None:
            name = (self.config.benchmark if self.config else "Benchmark") or "Benchmark"
            fig.add_trace(
                go.Scatter(
                    x=self.history["date"],
                    y=bench_eq,
                    mode="lines",
                    name=name,
                    line={"dash": "dash"},
                )
            )
        fig.update_layout(
            title="Equity curve",
            xaxis_title="Date",
            yaxis_title="Portfolio value ($)",
            hovermode="x unified",
        )
        return fig

    def plot_trades(self, ticker: str) -> go.Figure:
        """Line chart of adjusted close with buy/sell markers at fill prices.

        Uses the adjusted close (from `history`-time prices is not available, so
        we rebuild from trades + history endpoints — instead we just plot the
        trade fills against the trade-time price). For a full price series,
        re-pull via DataInterface outside Analytics.
        """
        ticker = ticker.upper()
        trades_t = [
            t for t in self.trades
            if t.get("ticker", "").upper() == ticker and "date" in t
        ]
        if not trades_t:
            raise ValueError(f"no trades found for ticker {ticker!r}")

        df = pd.DataFrame(trades_t)
        df["date"] = pd.to_datetime(df["date"])
        buys = df[df["side"] == "buy"]
        sells = df[df["side"] == "sell"]

        fig = go.Figure()
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys["date"],
                    y=buys["fill_price"],
                    mode="markers",
                    name="Buy",
                    marker={"symbol": "triangle-up", "size": 12, "color": "green"},
                )
            )
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells["date"],
                    y=sells["fill_price"],
                    mode="markers",
                    name="Sell",
                    marker={"symbol": "triangle-down", "size": 12, "color": "red"},
                )
            )
        fig.update_layout(
            title=f"Trades — {ticker} (fills at adjusted-close-based prices)",
            xaxis_title="Date",
            yaxis_title="Fill price ($)",
        )
        return fig
