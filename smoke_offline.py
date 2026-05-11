"""Offline smoke test — exercises M6/M7 strategy logic without the DB.

Verifies:
  - Look-ahead enforcement (Test 5 from testing.md, no DB needed).
  - MovingAverageCrossover signal computation on synthetic prices.
  - ValueScreen ranking on synthetic fundamentals.
  - Strategy.generate_signals fundamental look-ahead belt-and-braces check.

Run from project root:
    python smoke_offline.py
"""

from datetime import date

import pandas as pd

from backtesting_engine.strategy import (
    AllCash,
    BuyAndHold,
    MovingAverageCrossover,
    ValueScreen,
)


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
    print("\n" + "=" * 60)
    print("ALL OFFLINE TESTS PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
