# backtest_engine

A modular Python backtesting engine for systematic equity strategies, built on a MySQL financial database. Simulates daily-bar execution with realistic cost models, enforces look-ahead safety at the strategy layer, and ships with both rule-based and fundamental-screening strategies.

## What it does

Runs a strategy (rule-based or fundamental) against historical price + fundamentals data, simulates trade execution with commission and slippage, and outputs performance metrics plus interactive plots. The strategy layer is swappable — the same engine runs simple MA crossovers today and ML model predictions tomorrow without engine changes.

## Architecture

```
config.py          → BacktestConfig (settings)
data_interface.py  → DataInterface (MySQL → DataFrames)
models.py          → CommissionModel, SlippageModel (pluggable)
broker.py          → Broker (simulates trade execution)
strategy.py        → Strategy base + AllCash, BuyAndHold,
                     MovingAverageCrossover, ValueScreen
engine.py          → Engine (daily loop orchestrator)
analytics.py       → Analytics (metrics + plots)
main.py            → Entry point
```

Each module talks only to its immediate neighbors. Strategy never touches Broker. Broker never touches DataInterface. The Engine is the only module that wires everything together.

## Setup

Create the environment and install dependencies:

```bash
conda create -n backtest python=3.11
conda activate backtest
pip install -r requirements.txt
```

Create a `.env` file at the repo root with your MySQL connection (this file is gitignored):

```
sql_path=mysql+pymysql://user:password@host:3306/dbname
```

The DB schema expected by `DataInterface` is described in [.claude/rules/database.md](.claude/rules/database.md) — `daily_prices`, `income_stmt`, `balance_sheet`, `cashflow`, `factors_daily`, `dividends`. Price data should be loaded from yfinance with `auto_adjust=True` so `close` is already split- and dividend-adjusted.

Confirm the DB connection works before running a backtest:

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); \
from sqlalchemy import create_engine; import os; \
from backtesting_engine.data_interface import DataInterface; \
eng = create_engine(os.environ['sql_path']); \
print(DataInterface(eng).__enter__().get_prices(['AAPL'], '2023-01-03', '2023-01-10').head())"
```

If you see a DataFrame with `date`, `ticker`, `open`, `high`, `low`, `close`, `volume` columns, you're wired up.

## Running a backtest

The shortest path:

```bash
python -m backtesting_engine.main
```

That runs the configuration hard-coded in [`backtesting_engine/main.py`](backtesting_engine/main.py). To run your own, copy this pattern — every line maps to one module:

```python
from datetime import date
from dotenv import load_dotenv
from sqlalchemy import create_engine

from backtesting_engine.analytics import Analytics
from backtesting_engine.broker import Broker
from backtesting_engine.config import BacktestConfig
from backtesting_engine.data_interface import DataInterface
from backtesting_engine.engine import Engine
from backtesting_engine.models import FixedBpsSlippage, PerShareCommission
from backtesting_engine.strategy import MovingAverageCrossover

load_dotenv()  # reads sql_path from .env into the environment

# 1. Settings — universe, window, capital, cost model parameters.
config = BacktestConfig(
    start_date=date(2020, 1, 1),
    end_date=date(2024, 1, 1),
    initial_capital=100_000.0,
    tickers=["AAPL", "MSFT", "GOOGL"],
)

# 2. Strategy — decides target weights each day. Swap this line to swap strategies.
strategy = MovingAverageCrossover(config.tickers, fast_window=50, slow_window=200)

# 3. Broker — simulates fills, tracks cash and holdings, applies costs.
broker = Broker(
    initial_cash=config.initial_capital,
    commission=PerShareCommission(config.commission_per_share, config.commission_min),
    slippage=FixedBpsSlippage(config.slippage_bps),
)

# 4. Data + Engine — context manager closes the SQL connection on exit.
sql_engine = create_engine(config.db_url)
with DataInterface(sql_engine) as data:
    engine = Engine(config, data, strategy, broker)
    engine.run()  # populates engine.history and broker.trades

# 5. Analytics — print metrics, optionally render plots.
analytics = Analytics(engine.history, trades=broker.trades, config=config)
analytics.print_summary()
analytics.plot_equity_curve().show()         # opens in browser
analytics.plot_trades("AAPL").show()         # optional: per-ticker fills
```

Run the full validation suite (requires DB):

```bash
python smoke_validation.py
```

Run offline strategy unit tests (no DB):

```bash
python smoke_offline.py
```

## How to configure a backtest

Every run is driven by a `BacktestConfig`. Validation happens in `__post_init__` — invalid values raise immediately.

| Field | Type | Default | Notes |
|---|---|---|---|
| `start_date` | `date` | required | Must be strictly before `end_date`. |
| `end_date` | `date` | required | Exclusive of intraday — daily bars only. |
| `initial_capital` | `float` | `100_000.0` | Must be > 0. Used as the baseline for total/annualized return. |
| `tickers` | `list[str]` | required | At least one symbol. Auto-uppercased. Empty strings rejected. |
| `benchmark` | `str \| None` | `None` | If set, must appear in `daily_prices` for the window or `Engine.run()` raises. |
| `commission_per_share` | `float` | `0.005` | $/share, ≥ 0. IB tier-1 default. |
| `commission_min` | `float` | `1.0` | $ minimum per trade, ≥ 0. Floor that dominates small trades (< 200 shares). |
| `slippage_bps` | `float` | `5.0` | Basis points each side, ≥ 0. 5 bps in + 5 bps out = 10 bps round-trip. |
| `db_url` | `str \| None` | `None` | If not set, falls back to `$sql_path` then `$DATABASE_URL`. |

## Strategies

All strategies inherit from `Strategy` and return `{ticker: weight}` where weights are in `[0, 1]` and sum to ≤ 1.0. Any remaining weight sits in cash. The base class enforces look-ahead safety and weight validity before returning, so subclasses only implement `_compute`.

### `AllCash`

```python
AllCash()
```

Holds 100% cash. Returns `{}` every day. Reference for validation test #1 — a backtest with this strategy should produce 0% return and 0 trades.

### `BuyAndHold(tickers)`

```python
BuyAndHold(["AAPL", "MSFT"])
```

Equal-weight across `tickers`. With N tickers, each gets weight `1/N`. With daily rebalancing this means the engine trades once on day 1 and then mostly idles for a single-ticker run; for multi-ticker portfolios, price drift causes small rebalance trades every day (documented limitation — see the rebalance-band item on the roadmap).

### `MovingAverageCrossover(tickers, fast_window=50, slow_window=200)`

```python
MovingAverageCrossover(["AAPL", "MSFT", "NVDA"], fast_window=50, slow_window=200)
```

For each ticker, compares the mean of the last `fast_window` closes against the mean of the last `slow_window` closes. Bullish (fast > slow) tickers hold a fixed `1/N` slot; bearish or insufficient-history tickers sit in cash. So with 5 tickers and 3 bullish, the portfolio is 60% invested and 40% cash. A ticker without `slow_window` bars of history is skipped — no NaN propagation.

Constructor validates `0 < fast_window < slow_window`.

### `ValueScreen(tickers, top_n=3)`

```python
ValueScreen(["AAPL", "MSFT", "JPM", "KO", "XOM"], top_n=3)
```

Declares `requires_fundamentals = ("income_stmt",)`, so the engine pre-fetches and per-day-slices income statements by `filing_date`. Computes earnings yield = `EPS / latest_close` for each ticker, ranks descending, picks the top `top_n`, equal-weights them at `1/top_n`. Tickers with negative EPS, missing filings, or missing prices are excluded.

Limitation: uses single-quarter EPS rather than trailing-twelve-months. Ranking is consistent across tickers at a fixed quarterly cadence, but the absolute earnings-yield number is quarter-annualized. A one-line change in `_compute` sums the last 4 quarters if TTM is needed.

### Writing your own strategy

Subclass `Strategy` and implement `_compute`. The base class handles look-ahead checks and weight validation, so the subclass body stays short:

```python
from backtesting_engine.strategy import Strategy

class MyStrategy(Strategy):
    # Optional — only set if you need fundamentals. Engine pre-fetches these.
    requires_fundamentals = ()  # e.g. ("income_stmt", "balance_sheet")

    def __init__(self, tickers: list[str]):
        self.tickers = [t.upper() for t in tickers]

    def _compute(self, current_date, price_data, fundamentals):
        # price_data contains rows up to and including current_date (no future).
        # Return a dict of {ticker: weight}. Empty dict = all cash.
        return {self.tickers[0]: 1.0}
```

Hard rules: never touch `DataInterface` or `Broker` from inside a strategy, never read past `current_date`, never return negative weights or weights summing above 1.0. The base class will raise `ValueError` if you do.

## How the daily loop works

`Engine.run()` does the following, in order:

1. **One-shot data fetch.** Pulls the full price window for `config.tickers + [benchmark]` in a single DB call, and for each statement in `strategy.requires_fundamentals`, pulls all filings up to `end_date`. No per-day DB hits.
2. **Fail-fast checks.** If any configured ticker has zero rows in the window, or if `config.benchmark` is set but missing from the data, raises `ValueError` before iterating. Silent `None` propagation would mask config errors.
3. **Iterate trading dates** present in the price DataFrame (not the calendar — skips weekends and holidays automatically).
4. **Per day:**
   - Slice `price_hist` through `current_date` inclusive.
   - Slice fundamentals by `filing_date <= current_date` (critical rule #2 — filing date, not period-end date).
   - Call `strategy.generate_signals(current_date, price_hist, fundamentals)` to get target weights.
   - Mark to market: `pv = broker.total_value(today's prices)`.
   - Compute target shares per ticker: `floor(weight * pv / price)`.
   - Diff against current holdings; positions the strategy dropped target 0 (full exit).
   - Execute **sells first** to free cash, then **buys**. Each fill is tagged with `current_date` on `broker.trades[-1]`.
   - Append a history row: `{date, total_value, cash, holdings_value, holdings, benchmark_price, n_trades}`.
5. **Edge-case handling.**
   - *Cash-safe buy retry:* if slippage + commission tip a buy's cost just past available cash, the engine catches `ValueError` from `Broker`, drops one share, and retries. Keeps cost-model math in `Broker` as the single source of truth.
   - *Missing-price fallback:* if a held ticker has no row for today (data gap mid-window), `broker.total_value` raises `KeyError`; the engine catches it, skips the rebalance for that day, and records a `note="missing_price"` row instead of silently zeroing the position.

After `run()`, `engine.history` is a list of per-day dicts and `broker.trades` is the full fill log.

## Reading the output

### `Analytics.print_summary()`

Prints a block like:

```
============================================================
Backtest Summary  2023-01-03 -> 2023-12-29  (250 trading days)
============================================================
  Final value         $    154,651.60
  Total return                 54.65%
  Annualized return            55.47%
  Annualized vol               19.94%
  Sharpe (RFR=0)                 2.32
  Max drawdown                -14.92%
  Sortino (RFR=0)                3.10
  Calmar                         3.72
  Exposure time               100.00%
  Trades                            2
  Total commission    $          4.05

  TRADES (closed round-trips, FIFO-matched)
    Closed round-trips              0
    Win rate                      n/a
    Avg win                       n/a
    Avg loss                      n/a
    Profit factor                 n/a
    Avg duration (days)           n/a

Caveats (per output-limitations.md):
  ...
```

Field by field:

- **Final value** — `total_value` from the last history row (cash + mark-to-market holdings).
- **Total return** — `final / initial_capital - 1`. Anchored to `config.initial_capital`, not the first history row, so commission and slippage on the entry trade are reflected in the headline number.
- **Annualized return** — `(final / initial_capital) ** (252 / n_return_obs) - 1`. 252 trading-day convention.
- **Annualized vol** — `std(daily_returns) * sqrt(252)`. Sample std (ddof=1).
- **Sharpe (RFR=0)** — `mean(daily_returns) / std(daily_returns) * sqrt(252)`. Risk-free rate is 0 for v1 — a real RFR via `get_factors` lands in v1.x. `NaN` when std is 0 (e.g., AllCash).
- **Max drawdown** — most negative point on `value / cummax(value) - 1`. Always ≤ 0.
- **Sortino (RFR=0)** — Sharpe-style ratio with downside-only deviation as denominator: `mean(daily_returns) / std(daily_returns_where_<0) * sqrt(252)`. NaN when there are fewer than 2 negative-return days.
- **Calmar** — `annualized_return / abs(max_drawdown)`. NaN if max drawdown is 0.
- **Exposure time** — fraction of history rows with `holdings_value > 0`. AllCash → 0, BuyAndHold → ≈ 1.
- **Trades** — `len(broker.trades)`. Each fill (one ticker, one side) counts as one.
- **Total commission** — sum of `commission` across all trades.

The trailing **TRADES** block reports only **closed round-trips** matched via FIFO (see [Trade-level analytics](#trade-level-analytics-fifo-round-trips) below). Open lots are excluded from these stats and surfaced separately on `Analytics.open_positions`.

If `config.benchmark` is set, a second block reports the same metrics for the benchmark (rescaled to start at `initial_capital`) plus an **Active return** line = `strategy_total_return - benchmark_total_return`.

### Trade-level analytics (FIFO round-trips)

`Analytics` builds a closed-round-trip DataFrame at construction time via FIFO matching. Buy lots are pushed onto a per-ticker FIFO queue; sells consume them front-first. Commission is allocated per share on both entry and exit, so reported `pnl` is net of frictions (slippage is already baked into `fill_price` by the Broker).

Two attributes:

- `analytics.trade_pnl` — `DataFrame[ticker, entry_date, exit_date, shares, entry_price, exit_price, pnl, pct_return, duration_days]`. `pct_return` is anchored to entry notional, not portfolio value.
- `analytics.open_positions` — `DataFrame[ticker, entry_date, shares_remaining, entry_price]`. The FIFO leftover at the end of the trade stream.

Helper methods (all empty-safe — return `nan` / `0` rather than raise):

| Method | Returns |
|---|---|
| `total_trades()` | count of closed round-trips |
| `win_rate()` | fraction of rows with `pnl > 0` |
| `avg_win()` | mean of positive `pnl` |
| `avg_loss()` | mean of negative `pnl` (returned as a negative number, **not** `abs`) |
| `profit_factor()` | `sum(wins) / abs(sum(losses))`; `inf` if no losses + ≥1 win; `nan` if both empty |
| `avg_trade_duration()` | mean of `duration_days` |
| `sortino_ratio()` | downside-std variant of Sharpe |
| `calmar_ratio()` | `annual_return / abs(max_drawdown)` |
| `exposure_time()` | fraction of rows holding any position |

FIFO is chosen over LIFO/avg-cost: it matches the typical brokerage tax-lot default and gives deterministic, intuitive entry→exit pairs for trade review. If a tax-aware or wash-sale variant is ever needed, `_compute_trade_pnl` is the single chokepoint to swap.

### `Analytics.plot_equity_curve()`

Returns a plotly `go.Figure` with the strategy's `total_value` as a solid line; if a benchmark is configured, overlays it as a dashed line rescaled to the same starting dollar. The caller decides when to render — call `.show()` to open in a browser, or embed with `.to_html()`.

### `Analytics.plot_trades(ticker)`

Returns a plotly `go.Figure` with green triangle-up markers at buy fills and red triangle-down markers at sell fills for the given ticker. No candlestick overlay — `close` is split- and dividend-adjusted, which would make candles misleading. Raises `ValueError` if no trades exist for the ticker.

### `Analytics.plot_drawdown()`

Returns a plotly `go.Figure` with the underwater drawdown series (`value / cummax(value) - 1`) rendered as a red filled area, y-axis formatted as percent.

### `Analytics.plot_monthly_returns()`

Heatmap of monthly compounded returns. Years on the y-axis (top → bottom = oldest → newest), months on the x-axis, percent values shown inline, RdYlGn colorscale centered at zero.

### `Analytics.generate_report(output_path)`

Writes a self-contained HTML report to `output_path` containing: a metrics summary table (Sharpe, Sortino, Calmar, exposure, win rate, profit factor, …), the equity curve, the drawdown chart, the monthly-returns heatmap, and the standard caveats block. The first plot bundles plotly.js from the CDN; subsequent plots reuse it, so the file stays compact (≈ 30 KB for a one-year run).

```python
analytics.generate_report("backtest_report.html")
```

## Cost model

Realistic, not free:

- **Commission** — per-share with a per-trade minimum (Interactive Brokers tier-1 default: $0.005/share, $1 minimum).
- **Slippage** — 5 bps each side (10 bps round-trip), direction-aware so buys fill higher and sells fill lower.

### Worked example

Buying 100 AAPL at quoted $150:

```
slippage adjust    : 150 * (1 + 0.0005)  = 150.075   (5 bps up for a buy)
commission         : max(100 * 0.005, 1) = 1.00      ($1 floor wins)
total cost         : 100 * 150.075 + 1   = 15,008.50
cash after         : 100,000 - 15,008.50 = 84,991.50
```

Selling those 100 shares back at $150:

```
slippage adjust    : 150 * (1 - 0.0005)  = 149.925   (5 bps down for a sell)
commission         : max(100 * 0.005, 1) = 1.00
proceeds           : 100 * 149.925 - 1   = 14,991.50
```

Round-trip cost on $15k notional: `15,008.50 - 14,991.50 = $17.00` = $15 slippage + $2 commission. This is the canonical reference number; any change to commission/slippage defaults will change it.

A $100k single-stock entry pays roughly $50 slippage + $1–$5 commission. The 0.15% drag in test 2 (BuyAndHold AAPL 2023) is the canonical reference number for these defaults.

## Headline validation results

All 7 tests in the project's validation contract pass against a live MySQL DB of yfinance-sourced data.

| Test | Strategy | Window | Result |
|---|---|---|---|
| 1 | AllCash | AAPL 2023 | 0% return, 0 trades, flat equity curve |
| 2 | BuyAndHold | AAPL 2023 | **54.65%** (raw AAPL 54.80%, 0.15% cost drag) |
| 3 | BuyAndHold | 5-tech basket 2023 | 82.10% |
| 4 | MA Crossover 50/200 | AAPL 2020–2024 | 38.10%, 5 trades (including a real Oct 2022 whipsaw) |
| 5 | Look-ahead enforcement | — | `ValueError` raised on future prices and future `filing_date`s |
| 6 | Cash conservation | every run | `cash + holdings_value == total_value` asserted on every history row |
| 7 | Commission verification | every run | sum > 0, matches per-share commission model |

Plus ValueScreen on AAPL/MSFT/JPM/KO/XOM 2022–2024: 19.67% return, filing_date look-ahead spot-checked clean on 2023-06-30 (most recent filing in slice was 2023-05-03).

## Critical correctness invariants

Enforced in code, never violated:

1. **No look-ahead bias** — `price_data.date.max() <= current_date` on every Strategy call.
2. **`filing_date`, not `date`** — fundamentals filter by when the report became public, not the quarter end.
3. **`close` is already adjusted** — yfinance `auto_adjust=True`, use directly for returns.
4. **Strategies return target weights**, not share counts. Engine handles conversion and floor-rounding.
5. **Cash conservation** — `cash + sum(shares * price) == total_value` at every step.

## Roadmap

v1 is feature-complete. Next, in priority order:

- **v1.1** — Rebalance band: only trade when target/current diverges by more than a threshold. The single biggest cost-drag fix for multi-ticker portfolios; followed by limit and stop orders.
- **v1.2** — Volume-based slippage.
- **v1.3** — Position sizing models (Kelly, risk parity, inverse-volatility).
- **v1.4** — Multi-timeframe support.
- **v1.5** — Walk-forward analysis.
- **v2.0** — ML model integration: a `MLStrategy` subclass that takes a trained model and emits weights. Engine does not change.

## Known limitations

Always disclosed in `Analytics.print_summary()` and worth understanding before reading any backtest result:

- **Survivorship bias** — the DB contains only currently-listed companies. Strategies that pick "cheap" stocks (low P/E) will be biased because many cheap stocks went bankrupt and aren't in the data.
- **Adj-close retrospection** — adjusted prices change retroactively when companies do new splits or pay new dividends.
- **No intraday data** — daily bars only.
- **Same-day fills** — trades execute at the signal day's close. Real markets fill at next-day open or worse.

## Tech stack

- Python 3.10+
- MySQL 8 via SQLAlchemy 2 + PyMySQL (`caching_sha2_password` auth → `cryptography` required)
- pandas, numpy
- plotly (interactive charts), matplotlib (static)
- python-dotenv

## License

For personal portfolio use.
