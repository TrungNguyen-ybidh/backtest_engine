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

## Quick start

Setup:

```bash
conda create -n backtest python=3.11
conda activate backtest
pip install -r requirements.txt
```

Create a `.env` file at the repo root with your MySQL connection (this file is gitignored):

```
sql_path=mysql+pymysql://user:password@host:3306/dbname
```

The DB schema expected by `DataInterface` is described in `.claude/rules/database.md` — `daily_prices`, `income_stmt`, `balance_sheet`, `cashflow`, `factors_daily`, `dividends`. Price data should be loaded from yfinance with `auto_adjust=True` so `close` is already split- and dividend-adjusted.

Run a backtest:

```bash
python -m backtesting_engine.main
```

Run the full validation suite (requires DB):

```bash
python smoke_validation.py
```

Run offline strategy unit tests (no DB):

```bash
python smoke_offline.py
```

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

## Cost model

Realistic, not free:

- **Commission**: per-share with $1 minimum (Interactive Brokers tier-1 default).
- **Slippage**: 5 bps each side (10 bps round-trip), direction-aware so buys fill higher and sells fill lower.

A $100k single-stock entry pays roughly $50 slippage + $1–$5 commission. The 0.15% drag in test 2 (BuyAndHold AAPL 2023) is the canonical reference number for these defaults.

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
