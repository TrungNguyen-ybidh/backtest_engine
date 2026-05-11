"""Analytics — performance metrics and visualizations.

Inputs:
- history: list of per-day dicts from Engine.run().
- trades: optional reference to broker.trades for trade-level analytics.
- config: optional BacktestConfig for benchmark name + initial capital.

Reports always disclose the caveats from .claude/rules/output-limitations.md
(survivorship bias, adj-close retrospection, no intraday, same-day fills).
"""

from collections import deque
from pathlib import Path
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

        # FIFO round-trip P&L (M5.1). Computed once; trade-stat methods read this.
        self.trade_pnl, self.open_positions = self._compute_trade_pnl()

    # ---------- trade matching (M5.1) ----------

    _CLOSED_COLS = (
        "ticker", "entry_date", "exit_date", "shares",
        "entry_price", "exit_price", "pnl", "pct_return", "duration_days",
    )
    _OPEN_COLS = ("ticker", "entry_date", "shares_remaining", "entry_price")

    def _compute_trade_pnl(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Match sells to buys FIFO, per ticker, to build closed round-trips.

        Commission is allocated per-share at entry and at exit, so reported
        pnl is net of frictions (slippage is already baked into fill_price by
        the Broker). Anything still open at the end of the trade stream goes
        to `open_positions` instead.
        """
        closed_rows: list[dict] = []
        open_rows: list[dict] = []

        # Group trades by ticker; skip un-dated entries (engine tags `date`).
        by_ticker: dict[str, list[dict]] = {}
        for t in self.trades:
            if "date" not in t or "ticker" not in t:
                continue
            by_ticker.setdefault(t["ticker"].upper(), []).append(t)

        for ticker, ticker_trades in by_ticker.items():
            ticker_trades = sorted(ticker_trades, key=lambda r: r["date"])
            lots: deque = deque()  # FIFO of open buy lots

            for tr in ticker_trades:
                shares = int(tr["shares"])
                price = float(tr["fill_price"])
                comm = float(tr.get("commission", 0.0))
                cps = comm / abs(shares) if shares != 0 else 0.0
                tdate = pd.to_datetime(tr["date"])

                if shares > 0:
                    lots.append({
                        "entry_date": tdate,
                        "shares_remaining": shares,
                        "entry_price": price,
                        "entry_cps": cps,
                    })
                else:
                    qty_to_close = -shares
                    while qty_to_close > 0 and lots:
                        lot = lots[0]
                        take = min(lot["shares_remaining"], qty_to_close)
                        entry_price = lot["entry_price"]
                        exit_price = price
                        gross = (exit_price - entry_price) * take
                        costs = (lot["entry_cps"] + cps) * take
                        pnl = gross - costs
                        entry_notional = entry_price * take
                        closed_rows.append({
                            "ticker": ticker,
                            "entry_date": lot["entry_date"],
                            "exit_date": tdate,
                            "shares": int(take),
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "pct_return": pnl / entry_notional if entry_notional else float("nan"),
                            "duration_days": (tdate - lot["entry_date"]).days,
                        })
                        lot["shares_remaining"] -= take
                        qty_to_close -= take
                        if lot["shares_remaining"] == 0:
                            lots.popleft()
                    # If qty_to_close > 0 here, the broker would have rejected the
                    # sell (long-only). Leaving the loop is safe — no orphan rows.

            for lot in lots:
                open_rows.append({
                    "ticker": ticker,
                    "entry_date": lot["entry_date"],
                    "shares_remaining": int(lot["shares_remaining"]),
                    "entry_price": lot["entry_price"],
                })

        closed_df = (
            pd.DataFrame(closed_rows, columns=list(self._CLOSED_COLS))
            if closed_rows else pd.DataFrame(columns=list(self._CLOSED_COLS))
        )
        open_df = (
            pd.DataFrame(open_rows, columns=list(self._OPEN_COLS))
            if open_rows else pd.DataFrame(columns=list(self._OPEN_COLS))
        )
        return closed_df, open_df

    # ---------- trade stats (M5.1) ----------

    def total_trades(self) -> int:
        return len(self.trade_pnl)

    def win_rate(self) -> float:
        if self.trade_pnl.empty:
            return float("nan")
        return float((self.trade_pnl["pnl"] > 0).mean())

    def avg_win(self) -> float:
        wins = self.trade_pnl.loc[self.trade_pnl["pnl"] > 0, "pnl"]
        return float(wins.mean()) if not wins.empty else float("nan")

    def avg_loss(self) -> float:
        # Returned as a negative number (do not abs).
        losses = self.trade_pnl.loc[self.trade_pnl["pnl"] < 0, "pnl"]
        return float(losses.mean()) if not losses.empty else float("nan")

    def profit_factor(self) -> float:
        if self.trade_pnl.empty:
            return float("nan")
        wins_sum = float(self.trade_pnl.loc[self.trade_pnl["pnl"] > 0, "pnl"].sum())
        losses_sum = float(self.trade_pnl.loc[self.trade_pnl["pnl"] < 0, "pnl"].sum())
        denom = abs(losses_sum)
        if denom == 0:
            return float("inf") if wins_sum > 0 else float("nan")
        return wins_sum / denom

    def avg_trade_duration(self) -> float:
        if self.trade_pnl.empty:
            return float("nan")
        return float(self.trade_pnl["duration_days"].mean())

    # ---------- risk metrics (M5.1) ----------

    def sortino_ratio(self) -> float:
        """Sharpe-like, but denominator is std of negative daily returns only.

        RFR=0 for v1 (consistent with `_metrics_for_series`'s Sharpe).
        """
        daily = self._daily_returns()
        if len(daily) < 2:
            return float("nan")
        downside = daily[daily < 0]
        if len(downside) < 2:
            return float("nan")
        d_std = downside.std(ddof=1)
        if d_std == 0:
            return float("nan")
        return float(daily.mean() / d_std * np.sqrt(_TRADING_DAYS_PER_YEAR))

    def calmar_ratio(self) -> float:
        initial = self.config.initial_capital if self.config else None
        m = self._metrics_for_series(self.history["total_value"], baseline=initial)
        mdd = m["max_drawdown"]
        if mdd == 0 or mdd != mdd:  # 0 or NaN
            return float("nan")
        ann = m["annual_return"]
        if ann != ann:
            return float("nan")
        return float(ann / abs(mdd))

    def exposure_time(self) -> float:
        if "holdings_value" not in self.history.columns:
            return float("nan")
        return float((self.history["holdings_value"] > 0).mean())

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
        # M5.1 — additional risk metrics in the same block.
        sortino = self.sortino_ratio()
        calmar = self.calmar_ratio()
        exposure = self.exposure_time()
        print(f"  Sortino (RFR=0)     {sortino:>15.2f}")
        print(f"  Calmar              {calmar:>15.2f}")
        print(f"  Exposure time       {exposure:>15.2%}")
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

        # M5.1 — closed-round-trip stats. Only counts FIFO-matched pairs; still
        # -open positions are excluded (see self.open_positions).
        print()
        print("  TRADES (closed round-trips, FIFO-matched)")
        n_closed = self.total_trades()
        wr = self.win_rate()
        aw = self.avg_win()
        al = self.avg_loss()
        pf = self.profit_factor()
        dur = self.avg_trade_duration()
        print(f"    Closed round-trips  {n_closed:>13d}")
        print(f"    Win rate            {wr:>15.2%}" if wr == wr
              else f"    Win rate            {'n/a':>15s}")
        print(f"    Avg win             ${aw:>14,.2f}" if aw == aw
              else f"    Avg win             {'n/a':>15s}")
        print(f"    Avg loss            ${al:>14,.2f}" if al == al
              else f"    Avg loss            {'n/a':>15s}")
        if pf == pf:  # not NaN
            pf_str = "inf" if pf == float("inf") else f"{pf:.2f}"
            print(f"    Profit factor       {pf_str:>15s}")
        else:
            print(f"    Profit factor       {'n/a':>15s}")
        print(f"    Avg duration (days) {dur:>15.1f}" if dur == dur
              else f"    Avg duration (days) {'n/a':>15s}")

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

    # ---------- plots (M5.1) ----------

    def plot_drawdown(self) -> go.Figure:
        """Drawdown series as a red filled area, y-axis in percent."""
        values = self.history["total_value"]
        running_peak = values.cummax()
        drawdown = values / running_peak - 1.0

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.history["date"],
                y=drawdown,
                mode="lines",
                name="Drawdown",
                fill="tozeroy",
                line={"color": "red"},
                fillcolor="rgba(220, 60, 60, 0.30)",
            )
        )
        fig.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown",
            yaxis={"tickformat": ".0%"},
            hovermode="x unified",
        )
        return fig

    def plot_monthly_returns(self) -> go.Figure:
        """Heatmap of monthly compounded returns. Years on y, months on x."""
        daily = self._daily_returns()
        if daily.empty:
            return go.Figure()

        # Index daily returns by date so we can resample by calendar month.
        # history is already sorted; pct_change drops the first row.
        dated = pd.Series(
            daily.values,
            index=self.history["date"].iloc[1 : len(daily) + 1].values,
        )
        monthly = (1.0 + dated).resample("ME").prod() - 1.0
        df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        })
        pivot = df.pivot(index="year", columns="month", values="ret").sort_index()
        # Ensure all 12 month columns are present for a tidy grid.
        for m in range(1, 13):
            if m not in pivot.columns:
                pivot[m] = np.nan
        pivot = pivot[list(range(1, 13))]

        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        text = [[("" if v != v else f"{v * 100:.1f}%") for v in row]
                for row in pivot.values]

        fig = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=month_labels,
                y=[str(y) for y in pivot.index],
                colorscale="RdYlGn",
                zmid=0.0,
                text=text,
                texttemplate="%{text}",
                hovertemplate="%{y} %{x}: %{z:.2%}<extra></extra>",
                colorbar={"tickformat": ".0%"},
            )
        )
        fig.update_layout(
            title="Monthly returns",
            xaxis_title="Month",
            yaxis_title="Year",
            yaxis={"autorange": "reversed"},
        )
        return fig

    # ---------- report (M5.1) ----------

    def generate_report(self, output_path: str) -> str:
        """Build a self-contained HTML report (metrics + 3 plots + caveats)."""
        initial = self.config.initial_capital if self.config else None
        m = self._metrics_for_series(self.history["total_value"], baseline=initial)
        start = self.history["date"].iloc[0].date()
        end = self.history["date"].iloc[-1].date()
        n_days = len(self.history)
        total_commission = sum(t.get("commission", 0.0) for t in self.trades)
        n_trades = len(self.trades)

        def fmt_pct(x: float) -> str:
            return "n/a" if x != x else f"{x:.2%}"

        def fmt_num(x: float, places: int = 2) -> str:
            return "n/a" if x != x else f"{x:.{places}f}"

        def fmt_money(x: float) -> str:
            return "n/a" if x != x else f"${x:,.2f}"

        pf = self.profit_factor()
        pf_str = (
            "n/a" if pf != pf else ("inf" if pf == float("inf") else f"{pf:.2f}")
        )

        rows = [
            ("Window", f"{start} → {end} ({n_days} trading days)"),
            ("Final value", fmt_money(m["final"])),
            ("Total return", fmt_pct(m["total_return"])),
            ("Annualized return", fmt_pct(m["annual_return"])),
            ("Annualized vol", fmt_pct(m["annual_vol"])),
            ("Sharpe (RFR=0)", fmt_num(m["sharpe"])),
            ("Sortino (RFR=0)", fmt_num(self.sortino_ratio())),
            ("Calmar", fmt_num(self.calmar_ratio())),
            ("Max drawdown", fmt_pct(m["max_drawdown"])),
            ("Exposure time", fmt_pct(self.exposure_time())),
            ("Trades (fills)", str(n_trades)),
            ("Total commission", fmt_money(total_commission)),
            ("Closed round-trips", str(self.total_trades())),
            ("Win rate", fmt_pct(self.win_rate())),
            ("Avg win", fmt_money(self.avg_win())),
            ("Avg loss", fmt_money(self.avg_loss())),
            ("Profit factor", pf_str),
            ("Avg trade duration (days)", fmt_num(self.avg_trade_duration(), 1)),
        ]
        table_html = (
            "<table><tbody>"
            + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
            + "</tbody></table>"
        )

        # First plot bundles plotly.js from the CDN; the rest reuse it.
        equity_html = self.plot_equity_curve().to_html(
            include_plotlyjs="cdn", full_html=False
        )
        drawdown_html = self.plot_drawdown().to_html(
            include_plotlyjs=False, full_html=False
        )
        monthly_html = self.plot_monthly_returns().to_html(
            include_plotlyjs=False, full_html=False
        )

        caveats_html = f"<pre class='caveats'>{_CAVEATS}</pre>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Backtest report — {start} → {end}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif;
          max-width: 1100px; margin: 24px auto; padding: 0 16px; color: #222; }}
  h1 {{ margin-bottom: 4px; }}
  h2 {{ margin-top: 32px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; margin: 12px 0 24px; }}
  th, td {{ padding: 6px 14px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f7f7f7; font-weight: 600; }}
  pre.caveats {{ background: #fff8e1; border-left: 4px solid #f0c14b;
                 padding: 12px 16px; white-space: pre-wrap; }}
</style>
</head>
<body>
<h1>Backtest report</h1>
<p>{start} → {end} ({n_days} trading days)</p>

<h2>Summary</h2>
{table_html}

<h2>Equity curve</h2>
{equity_html}

<h2>Drawdown</h2>
{drawdown_html}

<h2>Monthly returns</h2>
{monthly_html}

<h2>Caveats</h2>
{caveats_html}
</body>
</html>
"""
        out = Path(output_path)
        out.write_text(html, encoding="utf-8")
        return str(out)
