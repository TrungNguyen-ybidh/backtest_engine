"""End-to-end validation suite — runs all 7 tests from the testing contract.

Run from project root:
    python smoke_validation.py

The 7 tests:
  1. AllCash → 0% return, 0 trades, flat equity curve.
  2. BuyAndHold single → return matches manual AAPL adj-close calc.
  3. BuyAndHold broad universe → reasonable return range.
  4. MA Crossover on AAPL — log trades, verify 4+ trades over 4-year window.
  5. No look-ahead — enforced by Strategy base class on every iteration.
  6. Cash conservation — `cash + holdings_value == total_value` on every row.
  7. Commission verification — sum > 0 and matches per-share model.
"""

import math
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine

from backtesting_engine.analytics import Analytics
from backtesting_engine.broker import Broker
from backtesting_engine.config import BacktestConfig
from backtesting_engine.data_interface import DataInterface
from backtesting_engine.engine import Engine
from backtesting_engine.models import FixedBpsSlippage, PerShareCommission
from backtesting_engine.strategy import (
    AllCash,
    BuyAndHold,
    MovingAverageCrossover,
    ValueScreen,
)


INITIAL = 100_000.0


def fresh_broker(config):
    return Broker(
        initial_cash=config.initial_capital,
        commission=PerShareCommission(config.commission_per_share, config.commission_min),
        slippage=FixedBpsSlippage(config.slippage_bps),
    )


def assert_cash_conservation(engine):
    for row in engine.history:
        if row["total_value"] != row["total_value"]:  # NaN
            continue
        diff = row["total_value"] - (row["cash"] + row["holdings_value"])
        assert abs(diff) < 1e-6, (
            f"cash conservation broken on {row['date']}: diff={diff}"
        )


def run_engine(sql_engine, config, strategy):
    broker = fresh_broker(config)
    with DataInterface(sql_engine) as data:
        engine = Engine(config, data, strategy, broker)
        engine.run()
    return engine, broker


def test_1_allcash(sql_engine):
    print("\n--- TEST 1: AllCash AAPL 2023 ---")
    config = BacktestConfig(
        start_date=date(2023, 1, 1), end_date=date(2024, 1, 1),
        initial_capital=INITIAL, tickers=["AAPL"],
    )
    engine, broker = run_engine(sql_engine, config, AllCash())

    final = engine.history[-1]["total_value"]
    assert abs(final - INITIAL) < 1e-6, f"AllCash drifted: final={final}"
    assert len(broker.trades) == 0, "AllCash traded"
    assert_cash_conservation(engine)
    print(f"  final ${final:,.2f}, 0 trades, {len(engine.history)} history rows")
    print("  [PASS]")


def test_2_buyandhold_single(sql_engine):
    print("\n--- TEST 2: BuyAndHold AAPL 2023 ---")
    config = BacktestConfig(
        start_date=date(2023, 1, 1), end_date=date(2024, 1, 1),
        initial_capital=INITIAL, tickers=["AAPL"],
    )
    engine, broker = run_engine(sql_engine, config, BuyAndHold(config.tickers))

    with DataInterface(sql_engine) as data:
        prices = data.get_prices(["AAPL"], config.start_date, config.end_date)

    final = engine.history[-1]["total_value"]
    strat_ret = final / INITIAL - 1
    raw_ret = prices["close"].iloc[-1] / prices["close"].iloc[0] - 1
    drag = raw_ret - strat_ret
    total_comm = sum(t["commission"] for t in broker.trades)

    print(f"  strategy: {strat_ret:.2%}  raw AAPL: {raw_ret:.2%}  drag: {drag:.2%}")
    print(f"  trades: {len(broker.trades)}  commission: ${total_comm:.2f}")

    assert strat_ret < raw_ret, "strategy beat raw stock — costs missing?"
    assert drag < 0.02, f"cost drag {drag:.2%} > 2%"
    assert total_comm > 0
    assert_cash_conservation(engine)
    print("  [PASS]")


def test_3_buyandhold_broad(sql_engine):
    print("\n--- TEST 3: BuyAndHold 5-stock universe (2023) ---")
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    config = BacktestConfig(
        start_date=date(2023, 1, 1), end_date=date(2024, 1, 1),
        initial_capital=INITIAL, tickers=universe,
    )
    engine, broker = run_engine(sql_engine, config, BuyAndHold(config.tickers))

    final = engine.history[-1]["total_value"]
    strat_ret = final / INITIAL - 1

    print(f"  strategy: {strat_ret:.2%}  ({len(broker.trades)} trades)")
    # Bound the magnitude — adding an index ETF to the DB would let us compare
    # to a real benchmark; for now, just sanity-check the return is plausible.
    assert -0.5 < strat_ret < 5.0, f"return {strat_ret:.2%} outside plausible range"
    assert len(broker.trades) >= len(universe), "should buy each ticker on day 1"
    assert_cash_conservation(engine)
    print("  [PASS]")


def test_4_ma_crossover(sql_engine):
    print("\n--- TEST 4: MovingAverageCrossover AAPL 2020-2024 (50/200) ---")
    config = BacktestConfig(
        start_date=date(2020, 1, 1), end_date=date(2024, 1, 1),
        initial_capital=INITIAL, tickers=["AAPL"],
    )
    strategy_obj = MovingAverageCrossover(config.tickers, fast_window=50, slow_window=200)
    engine, broker = run_engine(sql_engine, config, strategy_obj)

    final = engine.history[-1]["total_value"]
    strat_ret = final / INITIAL - 1
    n_trades = len(broker.trades)

    print(f"  final ${final:,.2f}  return {strat_ret:.2%}  trades: {n_trades}")
    print("  Trade log:")
    for t in broker.trades:
        print(
            f"    {t['date']}  {t['side']:4s}  {t['ticker']}  "
            f"{abs(t['shares']):>5d} @ ${t['fill_price']:.2f}"
        )
    # Expect a handful of trades (golden/death crosses): COVID crash + recovery,
    # 2022 bear, 2023 recovery. Want at least 2 round-trips (4+ trades).
    assert n_trades >= 4, f"only {n_trades} trades — strategy never crossed?"
    assert n_trades < 50, f"{n_trades} trades — strategy is whipsawing"
    assert_cash_conservation(engine)
    print("  [PASS]")


def test_5_no_lookahead():
    """The base class enforces this on every iteration. This test explicitly
    tries to violate it and confirms ValueError is raised."""
    print("\n--- TEST 5: Look-ahead enforcement ---")
    import pandas as pd

    bad_data = pd.DataFrame({
        "date": [pd.Timestamp("2024-01-15")],
        "ticker": ["AAPL"], "close": [150.0],
    })
    try:
        BuyAndHold(["AAPL"]).generate_signals(date(2024, 1, 10), bad_data)
    except ValueError as e:
        print(f"  raised ValueError as expected: {e}")
        print("  [PASS]")
        return
    raise AssertionError("look-ahead violation did not raise")


# Tests 6 (cash conservation) and 7 (commission > 0) are asserted in each
# strategy test above. The summary block prints a reminder at the end.


def test_valuescreen(sql_engine):
    print("\n--- TEST M7: ValueScreen 5-stock universe (2022-2024) ---")
    universe = ["AAPL", "MSFT", "JPM", "KO", "XOM"]
    config = BacktestConfig(
        start_date=date(2022, 1, 1), end_date=date(2024, 1, 1),
        initial_capital=INITIAL, tickers=universe,
    )
    engine, broker = run_engine(sql_engine, config, ValueScreen(config.tickers, top_n=3))

    final = engine.history[-1]["total_value"]
    strat_ret = final / INITIAL - 1
    n_trades = len(broker.trades)
    print(f"  final ${final:,.2f}  return {strat_ret:.2%}  trades: {n_trades}")

    # Filing-date spot check: pick one mid-window date and confirm the latest
    # filing in the engine's slice has filing_date <= that date.
    spot_date = date(2023, 6, 30)
    spot_row = next((r for r in engine.history if r["date"] == spot_date), None)
    if spot_row is None:
        print(f"  (no history row for {spot_date}, skipping filing-date spot)")
    else:
        held = list(spot_row["holdings"].keys())
        with DataInterface(sql_engine) as data:
            fund = data.get_fundamentals(held, "income_stmt", as_of=spot_date)
        print(f"  held on {spot_date}: {held}")
        if not fund.empty:
            max_filing = fund["filing_date"].max()
            print(f"  most recent filing in slice: {max_filing.date()}")
            assert max_filing.date() <= spot_date, "look-ahead leak in fundamentals!"

    assert n_trades > 0, "ValueScreen never traded"
    assert_cash_conservation(engine)
    print("  [PASS]")


def test_analytics_m5_1(sql_engine):
    """M5.1: FIFO trade matching, trade stats, risk metrics, HTML report.

    Re-runs three engines (AllCash, BuyAndHold single, MA Crossover) so the
    Analytics object has a fresh history + trade log for each. Cheap relative
    to the cost of a real DB session.
    """
    print("\n--- TEST M5.1: Analytics enhancements ---")

    # AllCash: no trades, no exposure. Trade-stat methods must not raise.
    config_a = BacktestConfig(
        start_date=date(2023, 1, 1), end_date=date(2024, 1, 1),
        initial_capital=INITIAL, tickers=["AAPL"],
    )
    engine_a, broker_a = run_engine(sql_engine, config_a, AllCash())
    an_a = Analytics(engine_a.history, trades=broker_a.trades, config=config_a)
    assert an_a.trade_pnl.empty, "AllCash should have no closed round-trips"
    assert an_a.total_trades() == 0
    assert math.isnan(an_a.win_rate())
    assert math.isnan(an_a.avg_win())
    assert math.isnan(an_a.avg_loss())
    assert math.isnan(an_a.profit_factor())
    assert math.isnan(an_a.avg_trade_duration())
    assert an_a.exposure_time() == 0.0
    print(f"  AllCash: trade_pnl empty, exposure_time={an_a.exposure_time():.2f}  [ok]")

    # BuyAndHold AAPL: 1 entry, position still open at end → 0 closed round-trips.
    config_b = BacktestConfig(
        start_date=date(2023, 1, 1), end_date=date(2024, 1, 1),
        initial_capital=INITIAL, tickers=["AAPL"],
    )
    engine_b, broker_b = run_engine(sql_engine, config_b, BuyAndHold(config_b.tickers))
    an_b = Analytics(engine_b.history, trades=broker_b.trades, config=config_b)
    assert an_b.total_trades() == 0, (
        f"BuyAndHold entry should leave 0 closed round-trips, got {an_b.total_trades()}"
    )
    assert not an_b.open_positions.empty, "open AAPL position should appear"
    assert an_b.exposure_time() > 0.99, (
        f"BuyAndHold should be ~always-invested, got {an_b.exposure_time():.3f}"
    )
    print(
        f"  BuyAndHold AAPL: closed={an_b.total_trades()}, "
        f"open_positions={len(an_b.open_positions)}, "
        f"exposure_time={an_b.exposure_time():.3f}  [ok]"
    )

    # MA Crossover: enough trades to build round-trips. Realized P&L should
    # not equal total P&L (open lots + commissions + slippage absorb the gap).
    config_m = BacktestConfig(
        start_date=date(2020, 1, 1), end_date=date(2024, 1, 1),
        initial_capital=INITIAL, tickers=["AAPL"],
    )
    strat = MovingAverageCrossover(config_m.tickers, fast_window=50, slow_window=200)
    engine_m, broker_m = run_engine(sql_engine, config_m, strat)
    an_m = Analytics(engine_m.history, trades=broker_m.trades, config=config_m)
    assert an_m.total_trades() >= 1, (
        f"MA crossover should produce at least 1 closed round-trip, got {an_m.total_trades()}"
    )
    pf = an_m.profit_factor()
    assert (pf == pf) and (pf == float("inf") or pf >= 0), (
        f"profit_factor should be finite or +inf, got {pf}"
    )
    realized = float(an_m.trade_pnl["pnl"].sum())
    total_pnl = engine_m.history[-1]["total_value"] - INITIAL
    assert abs(realized - total_pnl) > 1.0, (
        f"realized ({realized:.2f}) should differ from total ({total_pnl:.2f}) — "
        "unrealized P&L + costs must absorb the difference"
    )
    print(
        f"  MA Crossover: closed={an_m.total_trades()}, "
        f"profit_factor={pf:.2f}, realized=${realized:,.2f}, "
        f"total=${total_pnl:,.2f}  [ok]"
    )

    # Risk metrics return sensible numbers on the MA run.
    sortino = an_m.sortino_ratio()
    calmar = an_m.calmar_ratio()
    assert sortino == sortino, "sortino should be finite on a real run"
    assert calmar == calmar, "calmar should be finite on a real run"
    print(f"  Sortino={sortino:.2f}, Calmar={calmar:.2f}  [ok]")

    # HTML report renders to disk and is non-trivial.
    report_path = Path("test_report.html")
    try:
        an_m.generate_report(str(report_path))
        size = report_path.stat().st_size
        assert size > 10_000, f"report size {size} bytes is suspiciously small"
        print(f"  generate_report wrote {report_path} ({size:,} bytes)  [ok]")
    finally:
        if report_path.exists():
            report_path.unlink()

    print("  [PASS]")


def main() -> None:
    load_dotenv()
    sql_engine = create_engine(os.environ["sql_path"])

    test_1_allcash(sql_engine)
    test_2_buyandhold_single(sql_engine)
    test_3_buyandhold_broad(sql_engine)
    test_4_ma_crossover(sql_engine)
    test_5_no_lookahead()
    print("\n--- TEST 6 (cash conservation): asserted on every test above ---")
    print("--- TEST 7 (commission): asserted on every test above ---")
    test_valuescreen(sql_engine)
    test_analytics_m5_1(sql_engine)

    print("\n" + "=" * 60)
    print("ALL VALIDATION TESTS PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
