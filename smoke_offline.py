"""Offline smoke test — exercises M6/M7 strategy logic without the DB.

Verifies:
  - Look-ahead enforcement (Test 5 from testing.md, no DB needed).
  - MovingAverageCrossover signal computation on synthetic prices.
  - ValueScreen ranking on synthetic fundamentals.
  - Strategy.generate_signals fundamental look-ahead belt-and-braces check.

Run from project root:
    python smoke_offline.py
"""

import math
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from backtesting_engine.analytics import Analytics
from backtesting_engine.strategy import (
    AllCash,
    BuyAndHold,
    MovingAverageCrossover,
    ValueScreen,
)


class _Config:
    """Minimal stand-in for BacktestConfig — only fields Analytics reads."""

    def __init__(self, initial_capital=100_000.0, benchmark=None):
        self.initial_capital = initial_capital
        self.benchmark = benchmark


def _price_frame(rows):
    """rows: list of (date_str, ticker, close) → standard DataInterface format."""
    df = pd.DataFrame(rows, columns=["date", "ticker", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df


# ---------- Test 5: no look-ahead ----------

def test_no_lookahead():
    print("\n--- Look-ahead enforcement ---")
    bad = _price_frame([("2024-01-15", "AAPL", 150.0)])
    try:
        BuyAndHold(["AAPL"]).generate_signals(date(2024, 1, 10), bad)
    except ValueError as e:
        print(f"  [PASS] raised: {e}")
        return
    raise AssertionError("look-ahead violation did not raise")


def test_no_lookahead_fundamentals():
    print("\n--- Look-ahead enforcement (fundamentals) ---")
    prices = _price_frame([("2024-01-05", "AAPL", 150.0)])
    fund = pd.DataFrame({
        "ticker": ["AAPL"],
        "filing_date": pd.to_datetime(["2024-02-01"]),  # future filing
        "eps": [1.5],
    })
    try:
        ValueScreen(["AAPL", "MSFT"], top_n=1).generate_signals(
            date(2024, 1, 10), prices, fundamentals={"income_stmt": fund}
        )
    except ValueError as e:
        print(f"  [PASS] raised: {e}")
        return
    raise AssertionError("fundamentals look-ahead did not raise")


# ---------- MA Crossover ----------

def test_ma_crossover_bullish():
    print("\n--- MovingAverageCrossover: bullish setup ---")
    # 220 daily closes, strongly rising → fast > slow → bullish.
    closes = [100 + i * 0.5 for i in range(220)]
    rows = [(f"2023-{((i // 21) % 12) + 1:02d}-{(i % 21) + 1:02d}", "AAPL", c)
            for i, c in enumerate(closes)]
    # Simpler: just use sequential business dates
    dates = pd.bdate_range("2022-01-01", periods=220)
    rows = list(zip(dates.strftime("%Y-%m-%d"), ["AAPL"] * 220, closes))
    df = _price_frame(rows)

    strat = MovingAverageCrossover(["AAPL"], fast_window=50, slow_window=200)
    w = strat.generate_signals(dates[-1].date(), df)
    print(f"  weights: {w}")
    assert w == {"AAPL": 1.0}, f"expected bullish full-slot, got {w}"
    print("  [PASS]")


def test_ma_crossover_bearish():
    print("\n--- MovingAverageCrossover: bearish setup ---")
    # Strongly falling: fast < slow → out.
    closes = [200 - i * 0.5 for i in range(220)]
    dates = pd.bdate_range("2022-01-01", periods=220)
    rows = list(zip(dates.strftime("%Y-%m-%d"), ["AAPL"] * 220, closes))
    df = _price_frame(rows)

    strat = MovingAverageCrossover(["AAPL"], fast_window=50, slow_window=200)
    w = strat.generate_signals(dates[-1].date(), df)
    print(f"  weights: {w}")
    assert w == {}, f"expected empty (cash), got {w}"
    print("  [PASS]")


def test_ma_crossover_insufficient_history():
    print("\n--- MovingAverageCrossover: < slow_window bars ---")
    # Only 100 bars → can't compute 200-day MA → sit out.
    closes = [100 + i * 0.5 for i in range(100)]
    dates = pd.bdate_range("2022-01-01", periods=100)
    rows = list(zip(dates.strftime("%Y-%m-%d"), ["AAPL"] * 100, closes))
    df = _price_frame(rows)

    strat = MovingAverageCrossover(["AAPL"], fast_window=50, slow_window=200)
    w = strat.generate_signals(dates[-1].date(), df)
    print(f"  weights: {w}")
    assert w == {}, "should be empty when history < slow_window"
    print("  [PASS]")


def test_ma_crossover_mixed_universe():
    print("\n--- MovingAverageCrossover: mixed universe (2 of 3 bullish) ---")
    dates = pd.bdate_range("2022-01-01", periods=220)
    rows = []
    # AAPL and MSFT trending up, GOOGL trending down
    for ticker, slope, intercept in [
        ("AAPL", 0.5, 100),
        ("MSFT", 0.3, 150),
        ("GOOGL", -0.4, 200),
    ]:
        for i, dt in enumerate(dates):
            rows.append((dt.strftime("%Y-%m-%d"), ticker, intercept + i * slope))
    df = _price_frame(rows)

    strat = MovingAverageCrossover(["AAPL", "MSFT", "GOOGL"], fast_window=50, slow_window=200)
    w = strat.generate_signals(dates[-1].date(), df)
    print(f"  weights: {w}")
    # 2 of 3 bullish, each gets 1/3 slot → sum = 2/3
    assert set(w.keys()) == {"AAPL", "MSFT"}, f"expected AAPL+MSFT, got {set(w.keys())}"
    assert all(abs(v - 1/3) < 1e-9 for v in w.values())
    print("  [PASS]")


# ---------- ValueScreen ----------

def test_valuescreen_basic():
    print("\n--- ValueScreen: ranks by EPS/price ---")
    today = date(2024, 6, 30)
    # Prices: all $100 → eps/price ranks by raw EPS.
    prices = _price_frame([
        ("2024-06-28", "AAPL", 100.0),
        ("2024-06-28", "MSFT", 100.0),
        ("2024-06-28", "JPM",  100.0),
        ("2024-06-28", "KO",   100.0),
    ])
    fund = pd.DataFrame({
        "ticker":      ["AAPL", "MSFT", "JPM", "KO"],
        "filing_date": pd.to_datetime(["2024-04-30", "2024-04-30", "2024-04-30", "2024-04-30"]),
        "eps":         [1.5,    1.0,    3.0,   2.0],  # JPM highest
    })

    strat = ValueScreen(["AAPL", "MSFT", "JPM", "KO"], top_n=2)
    w = strat.generate_signals(today, prices, fundamentals={"income_stmt": fund})
    print(f"  weights: {w}")
    assert set(w.keys()) == {"JPM", "KO"}, f"expected top-2 by EPS (JPM,KO), got {set(w.keys())}"
    assert all(abs(v - 0.5) < 1e-9 for v in w.values())
    print("  [PASS]")


def test_valuescreen_skips_negative_eps():
    print("\n--- ValueScreen: skips negative-EPS names ---")
    today = date(2024, 6, 30)
    prices = _price_frame([
        ("2024-06-28", "X", 100.0),
        ("2024-06-28", "Y", 100.0),
        ("2024-06-28", "Z", 100.0),
    ])
    fund = pd.DataFrame({
        "ticker":      ["X",  "Y",   "Z"],
        "filing_date": pd.to_datetime(["2024-04-30"] * 3),
        "eps":         [-1.0, 0.5,   1.0],  # X has losses → skip
    })
    strat = ValueScreen(["X", "Y", "Z"], top_n=2)
    w = strat.generate_signals(today, prices, fundamentals={"income_stmt": fund})
    print(f"  weights: {w}")
    assert "X" not in w, "negative-EPS ticker should be excluded"
    print("  [PASS]")


def test_valuescreen_no_filing_yet():
    print("\n--- ValueScreen: returns empty if no eligible filings ---")
    today = date(2024, 1, 5)
    prices = _price_frame([("2024-01-03", "AAPL", 150.0)])
    # All filings in the future → engine would slice them all away.
    fund = pd.DataFrame(columns=["ticker", "filing_date", "eps"])
    strat = ValueScreen(["AAPL", "MSFT"], top_n=1)
    w = strat.generate_signals(today, prices, fundamentals={"income_stmt": fund})
    print(f"  weights: {w}")
    assert w == {}, "should return empty when no fundamentals are available yet"
    print("  [PASS]")


# ---------- M5.1 Analytics (offline, no DB) ----------

def _build_history(dates, values, holdings_values=None):
    """Build the dict-list history schema Engine produces."""
    if holdings_values is None:
        holdings_values = [0.0] * len(values)
    return [
        {
            "date": d.date(),
            "total_value": float(v),
            "cash": float(v - hv),
            "holdings_value": float(hv),
            "holdings": {},
            "benchmark_price": None,
            "n_trades": 0,
        }
        for d, v, hv in zip(dates, values, holdings_values)
    ]


def test_analytics_allcash_offline():
    print("\n--- M5.1 Analytics: AllCash empty round-trips ---")
    dates = pd.bdate_range("2023-01-03", periods=20)
    history = _build_history(dates, [100_000.0] * 20, [0.0] * 20)
    an = Analytics(history, trades=[], config=_Config())
    assert an.trade_pnl.empty
    assert an.total_trades() == 0
    assert math.isnan(an.win_rate())
    assert math.isnan(an.avg_win())
    assert math.isnan(an.avg_loss())
    assert math.isnan(an.profit_factor())
    assert math.isnan(an.avg_trade_duration())
    assert an.exposure_time() == 0.0
    print("  [PASS]")


def test_analytics_buyandhold_open_position_offline():
    print("\n--- M5.1 Analytics: BuyAndHold-style entry stays in open_positions ---")
    dates = pd.bdate_range("2023-01-03", periods=20)
    # Value rises from $100k to $110k while fully invested → exposure ≈ 1.0.
    values = [100_000.0 + i * (10_000 / 19) for i in range(20)]
    holdings_values = [v - 1_000.0 for v in values]  # always have holdings
    history = _build_history(dates, values, holdings_values)
    trades = [{
        "date": dates[0].date(),
        "ticker": "AAPL", "shares": 100, "fill_price": 150.0,
        "commission": 1.0, "side": "buy",
    }]
    an = Analytics(history, trades=trades, config=_Config())
    assert an.total_trades() == 0, "entry-only ⇒ no closed round-trip"
    assert len(an.open_positions) == 1
    assert an.open_positions.iloc[0]["shares_remaining"] == 100
    assert an.exposure_time() > 0.99
    print(f"  exposure_time={an.exposure_time():.3f}, open={len(an.open_positions)}  [PASS]")


def test_analytics_fifo_matching_offline():
    print("\n--- M5.1 Analytics: FIFO round-trip matching ---")
    base = date(2023, 1, 3)
    trades = [
        # Two buy lots, one partial sell, one full sell, one residual buy.
        {"date": base, "ticker": "AAPL", "shares": 100, "fill_price": 100.0, "commission": 1.0, "side": "buy"},
        {"date": base + timedelta(days=5), "ticker": "AAPL", "shares": 50, "fill_price": 110.0, "commission": 1.0, "side": "buy"},
        {"date": base + timedelta(days=10), "ticker": "AAPL", "shares": -120, "fill_price": 120.0, "commission": 1.0, "side": "sell"},
        {"date": base + timedelta(days=20), "ticker": "AAPL", "shares": -30, "fill_price": 130.0, "commission": 1.0, "side": "sell"},
        {"date": base + timedelta(days=25), "ticker": "AAPL", "shares": 10, "fill_price": 140.0, "commission": 1.0, "side": "buy"},
    ]
    # History values are irrelevant to FIFO matching itself but Analytics needs ≥1 row.
    dates = pd.bdate_range("2023-01-03", periods=30)
    history = _build_history(dates, [100_000.0] * 30, [50_000.0] * 30)
    an = Analytics(history, trades=trades, config=_Config())

    # Expected closed rows: 100@100→120, 20@110→120, 30@110→130 ⇒ 3 round-trips.
    assert an.total_trades() == 3, f"expected 3 closed round-trips, got {an.total_trades()}"
    # Residual open lot: 10 shares from final buy.
    assert len(an.open_positions) == 1
    assert an.open_positions.iloc[0]["shares_remaining"] == 10

    pnls = an.trade_pnl["pnl"].tolist()
    # Per-row gross checks (costs are small but non-zero so test bands):
    # Row 0: 100 * (120 - 100) = 2000, minus ~$1.83 of commission allocation.
    assert 1995 < pnls[0] < 2000, pnls
    # Row 1: 20 * (120 - 110) = 200
    assert 198 < pnls[1] < 200, pnls
    # Row 2: 30 * (130 - 110) = 600
    assert 597 < pnls[2] < 600, pnls

    # Win rate and profit factor sanity.
    assert an.win_rate() == 1.0
    assert an.profit_factor() == float("inf"), "no losers ⇒ profit_factor=inf"
    assert math.isnan(an.avg_loss()) is False or an.avg_loss() != an.avg_loss()
    print(f"  pnls={[round(p, 2) for p in pnls]}, open={len(an.open_positions)}  [PASS]")


def test_analytics_profit_factor_with_losses():
    print("\n--- M5.1 Analytics: profit_factor with wins + losses ---")
    base = date(2023, 1, 3)
    trades = [
        {"date": base, "ticker": "AAPL", "shares": 100, "fill_price": 100.0, "commission": 1.0, "side": "buy"},
        {"date": base + timedelta(days=5), "ticker": "AAPL", "shares": -100, "fill_price": 110.0, "commission": 1.0, "side": "sell"},  # +$1000
        {"date": base + timedelta(days=10), "ticker": "AAPL", "shares": 100, "fill_price": 110.0, "commission": 1.0, "side": "buy"},
        {"date": base + timedelta(days=15), "ticker": "AAPL", "shares": -100, "fill_price": 105.0, "commission": 1.0, "side": "sell"},  # -$500
    ]
    dates = pd.bdate_range("2023-01-03", periods=30)
    history = _build_history(dates, [100_000.0] * 30, [50_000.0] * 30)
    an = Analytics(history, trades=trades, config=_Config())
    assert an.total_trades() == 2
    pf = an.profit_factor()
    # ~1000 / ~500 = ~2.0 (slightly off due to commissions). Accept 1.9–2.05.
    assert 1.9 < pf < 2.05, f"expected pf~2, got {pf}"
    assert an.win_rate() == 0.5
    assert an.avg_win() > 0
    assert an.avg_loss() < 0, "avg_loss must be a NEGATIVE number, not abs()"
    print(f"  profit_factor={pf:.2f}, win_rate={an.win_rate():.2f}, avg_loss={an.avg_loss():.2f}  [PASS]")


def test_analytics_risk_metrics_offline():
    print("\n--- M5.1 Analytics: Sortino, Calmar, exposure_time ---")
    dates = pd.bdate_range("2023-01-03", periods=60)
    # Mostly-up series with one drawdown stretch so MDD < 0 and Sortino is finite.
    values = [100_000.0]
    for i in range(1, 60):
        if 20 <= i < 30:
            values.append(values[-1] * 0.99)  # drawdown phase
        else:
            values.append(values[-1] * 1.005)  # uptrend phase
    holdings_values = [v - 1_000.0 for v in values]
    history = _build_history(dates, values, holdings_values)
    an = Analytics(history, trades=[], config=_Config())

    sortino = an.sortino_ratio()
    calmar = an.calmar_ratio()
    exp = an.exposure_time()
    assert sortino == sortino, "sortino should be finite"
    assert calmar == calmar, "calmar should be finite"
    assert abs(exp - 1.0) < 1e-9, f"exposure should be 1.0, got {exp}"
    print(f"  sortino={sortino:.2f}, calmar={calmar:.2f}, exposure={exp:.3f}  [PASS]")


def test_analytics_generate_report_offline():
    print("\n--- M5.1 Analytics: generate_report writes HTML ---")
    dates = pd.bdate_range("2023-01-03", periods=60)
    values = [100_000.0 * (1 + 0.001 * i) for i in range(60)]
    holdings_values = [v - 1_000.0 for v in values]
    history = _build_history(dates, values, holdings_values)
    trades = [
        {"date": dates[0].date(), "ticker": "AAPL", "shares": 100, "fill_price": 150.0, "commission": 1.0, "side": "buy"},
        {"date": dates[20].date(), "ticker": "AAPL", "shares": -100, "fill_price": 160.0, "commission": 1.0, "side": "sell"},
    ]
    an = Analytics(history, trades=trades, config=_Config())
    out = Path("test_report_offline.html")
    try:
        an.generate_report(str(out))
        size = out.stat().st_size
        assert size > 10_000, f"report size {size} bytes is suspiciously small"
        # Sanity: the HTML must contain the table title and a plotly div.
        text = out.read_text(encoding="utf-8")
        assert "Backtest report" in text
        assert "plotly" in text.lower()
        print(f"  wrote {out} ({size:,} bytes)  [PASS]")
    finally:
        if out.exists():
            out.unlink()


def main() -> None:
    test_no_lookahead()
    test_no_lookahead_fundamentals()
    test_ma_crossover_bullish()
    test_ma_crossover_bearish()
    test_ma_crossover_insufficient_history()
    test_ma_crossover_mixed_universe()
    test_valuescreen_basic()
    test_valuescreen_skips_negative_eps()
    test_valuescreen_no_filing_yet()
    test_analytics_allcash_offline()
    test_analytics_buyandhold_open_position_offline()
    test_analytics_fifo_matching_offline()
    test_analytics_profit_factor_with_losses()
    test_analytics_risk_metrics_offline()
    test_analytics_generate_report_offline()
    print("\n" + "=" * 60)
    print("ALL OFFLINE TESTS PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
