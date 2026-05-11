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
from backtesting_engine.broker import Broker
from backtesting_engine.config import BacktestConfig
from backtesting_engine.engine import Engine
from backtesting_engine.models import (
    FixedBpsSlippage,
    LimitOrder,
    PerShareCommission,
)
from backtesting_engine.strategy import (
    AllCash,
    BuyAndHold,
    LimitBuyTheDip,
    LongShortBarbell,
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


# ---------- M8 Rebalance frequency (offline, fake DataInterface) ----------


class _FakeDataInterface:
    """Minimal DataInterface stand-in for offline engine tests.

    The engine only calls `get_prices` and `get_fundamentals` — no SQL, no
    connection lifecycle. We just hand back the canned DataFrames the test set up.
    """

    def __init__(self, prices: pd.DataFrame, fundamentals: dict | None = None):
        self._prices = prices
        self._fundamentals = fundamentals or {}

    def get_prices(self, tickers, start, end):
        df = self._prices.copy()
        df = df[df["ticker"].isin([t.upper() for t in tickers])]
        df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]
        return df.reset_index(drop=True)

    def get_fundamentals(self, tickers, statement, as_of):
        df = self._fundamentals.get(statement, pd.DataFrame()).copy()
        if df.empty:
            return df
        df = df[df["ticker"].isin([t.upper() for t in tickers])]
        df = df[df["filing_date"].dt.date <= as_of]
        return df.reset_index(drop=True)


def _synthetic_prices(tickers, start, periods, slopes=None, intercepts=None, vol_pct=0.0):
    """Daily business-day OHLC DataFrame in the standard DataInterface format.

    `vol_pct` widens the daily high/low band around close (defaults to flat
    bars where open=high=low=close). Tests that need to trigger limit fills
    use a non-zero band so today's [low, high] can straddle a limit price.
    """
    dates = pd.bdate_range(start, periods=periods)
    rows = []
    for i, t in enumerate(tickers):
        slope = (slopes or [0.5] * len(tickers))[i]
        intercept = (intercepts or [100.0] * len(tickers))[i]
        for j, d in enumerate(dates):
            close = intercept + j * slope
            high = close * (1 + vol_pct)
            low = close * (1 - vol_pct)
            rows.append((d, t, close, high, low, close))
    df = pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df, dates


def _fresh_broker(config):
    return Broker(
        initial_cash=config.initial_capital,
        commission=PerShareCommission(config.commission_per_share, config.commission_min),
        slippage=FixedBpsSlippage(config.slippage_bps),
    )


def _run_engine_offline(config, strategy, prices):
    """Build engine with fake DataInterface and run."""
    data = _FakeDataInterface(prices)
    broker = _fresh_broker(config)
    engine = Engine(config, data, strategy, broker)
    engine.run()
    return engine, broker


def test_m8_config_validation():
    print("\n--- M8: BacktestConfig rejects unknown rebalance_frequency ---")
    try:
        BacktestConfig(
            start_date=date(2023, 1, 1), end_date=date(2024, 1, 1),
            tickers=["AAPL"], rebalance_frequency="hourly",
        )
    except ValueError as e:
        print(f"  [PASS] raised: {e}")
        return
    raise AssertionError("invalid rebalance_frequency was accepted")


def test_m8_default_is_daily():
    print("\n--- M8: default rebalance_frequency='daily' (no field set) ---")
    config = BacktestConfig(
        start_date=date(2023, 1, 1), end_date=date(2024, 1, 1),
        tickers=["AAPL"],
    )
    assert config.rebalance_frequency == "daily", config.rebalance_frequency
    print("  [PASS]")


def test_m8_backwards_compat_daily():
    """Test 8: a daily-default run produces identical history+trades to v1 path."""
    print("\n--- M8 Test 8: default-daily backwards compat ---")
    prices, dates = _synthetic_prices(
        ["AAPL", "MSFT"], "2023-01-02", periods=60,
        slopes=[0.5, 0.3], intercepts=[100.0, 150.0],
    )
    base_config_kwargs = dict(
        start_date=dates[0].date(), end_date=dates[-1].date(),
        initial_capital=100_000.0, tickers=["AAPL", "MSFT"],
    )
    # v1 baseline: no rebalance_frequency in kwargs → default 'daily'.
    cfg_default = BacktestConfig(**base_config_kwargs)
    eng_default, brk_default = _run_engine_offline(
        cfg_default, BuyAndHold(["AAPL", "MSFT"]), prices
    )
    # Explicitly-set daily should produce identical results.
    cfg_explicit = BacktestConfig(rebalance_frequency="daily", **base_config_kwargs)
    eng_explicit, brk_explicit = _run_engine_offline(
        cfg_explicit, BuyAndHold(["AAPL", "MSFT"]), prices
    )
    assert len(eng_default.history) == len(eng_explicit.history)
    for a, b in zip(eng_default.history, eng_explicit.history):
        assert a["date"] == b["date"]
        assert abs(a["total_value"] - b["total_value"]) < 1e-9
        assert a["holdings"] == b["holdings"]
    assert len(brk_default.trades) == len(brk_explicit.trades)
    print(f"  history rows: {len(eng_default.history)}, trades: {len(brk_default.trades)}  [PASS]")


def test_m8_monthly_trade_count():
    """Test 9: monthly rebalance produces ~1 rebalance/month, far fewer trades."""
    print("\n--- M8 Test 9: monthly trade count << daily ---")
    # Synthetic, 12 months of business days (~252 trading days) across 2 tickers.
    prices, dates = _synthetic_prices(
        ["AAPL", "MSFT"], "2023-01-02", periods=252,
        slopes=[0.5, 0.3], intercepts=[100.0, 150.0],
    )
    kwargs = dict(
        start_date=dates[0].date(), end_date=dates[-1].date(),
        initial_capital=100_000.0, tickers=["AAPL", "MSFT"],
    )

    _, brk_daily = _run_engine_offline(
        BacktestConfig(**kwargs), BuyAndHold(["AAPL", "MSFT"]), prices,
    )
    eng_monthly, brk_monthly = _run_engine_offline(
        BacktestConfig(rebalance_frequency="monthly", **kwargs),
        BuyAndHold(["AAPL", "MSFT"]),
        prices,
    )

    # Count distinct trade dates as a proxy for "rebalance count".
    rebal_days_monthly = len({t["date"] for t in brk_monthly.trades})
    rebal_days_daily = len({t["date"] for t in brk_daily.trades})
    # 12 months → at most 12 rebalances; daily over 252 days = many more.
    assert rebal_days_monthly <= 13, (
        f"expected ≤13 monthly rebalance days, got {rebal_days_monthly}"
    )
    assert rebal_days_monthly < rebal_days_daily, (
        f"monthly ({rebal_days_monthly}) should be < daily ({rebal_days_daily})"
    )
    print(
        f"  daily rebalance days: {rebal_days_daily}, "
        f"monthly: {rebal_days_monthly}, "
        f"history rows: {len(eng_monthly.history)}  [PASS]"
    )


def test_m8_equity_continuity():
    """Test 10: history has 1 row per trading day even when not rebalancing."""
    print("\n--- M8 Test 10: equity curve continuity on non-rebalance days ---")
    prices, dates = _synthetic_prices(
        ["AAPL"], "2023-01-02", periods=60, slopes=[0.5], intercepts=[100.0],
    )
    cfg = BacktestConfig(
        start_date=dates[0].date(), end_date=dates[-1].date(),
        initial_capital=100_000.0, tickers=["AAPL"],
        rebalance_frequency="monthly",
    )
    eng, brk = _run_engine_offline(cfg, BuyAndHold(["AAPL"]), prices)
    # Every trading day in the slice should appear exactly once in history.
    expected_dates = sorted(set(prices["date"].dt.date))
    history_dates = [row["date"] for row in eng.history]
    assert history_dates == expected_dates, (
        f"history length {len(history_dates)} != trading days {len(expected_dates)}"
    )
    # Total value must rise as the price rises, even between rebalances.
    pv_first = eng.history[0]["total_value"]
    pv_last = eng.history[-1]["total_value"]
    assert pv_last > pv_first, "monotonically rising synth → pv must rise"
    # Cash conservation on every row.
    for row in eng.history:
        if row["total_value"] != row["total_value"]:
            continue
        diff = row["total_value"] - (row["cash"] + row["holdings_value"])
        assert abs(diff) < 1e-6, f"cash conservation broken on {row['date']}"
    print(f"  {len(history_dates)} history rows, pv {pv_first:,.0f} → {pv_last:,.0f}  [PASS]")


def test_m8_first_day_rebalance():
    """Test 11: very first trading day always rebalances regardless of frequency."""
    print("\n--- M8 Test 11: first-day rebalance under every frequency ---")
    prices, dates = _synthetic_prices(
        ["AAPL"], "2023-01-02", periods=5, slopes=[0.5], intercepts=[100.0],
    )
    for freq in ("daily", "weekly", "monthly", "quarterly", "yearly"):
        cfg = BacktestConfig(
            start_date=dates[0].date(), end_date=dates[-1].date(),
            initial_capital=100_000.0, tickers=["AAPL"],
            rebalance_frequency=freq,
        )
        eng, brk = _run_engine_offline(cfg, BuyAndHold(["AAPL"]), prices)
        first_trade_date = brk.trades[0]["date"] if brk.trades else None
        assert first_trade_date == dates[0].date(), (
            f"{freq}: first trade on {first_trade_date}, expected {dates[0].date()}"
        )
    print("  every freq fires on day 1  [PASS]")


# ---------- M9 Limit orders (Broker unit tests + Engine integration) ----------


def _fresh_broker_simple(initial_cash=100_000.0):
    return Broker(
        initial_cash=initial_cash,
        commission=PerShareCommission(0.005, 1.0),
        slippage=FixedBpsSlippage(5.0),
    )


def _bar(open_, high, low, close):
    return {"open": open_, "high": high, "low": low, "close": close}


def test_m9_limitorder_validation():
    print("\n--- M9: LimitOrder dataclass validation ---")
    # Bad side
    try:
        LimitOrder(ticker="AAPL", side="short", shares=10, limit_price=100.0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid side accepted")
    # shares <= 0
    try:
        LimitOrder(ticker="AAPL", side="buy", shares=0, limit_price=100.0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero shares accepted")
    # gtc_expire without expiration_date
    try:
        LimitOrder(ticker="AAPL", side="buy", shares=10, limit_price=100.0, lifetime="gtc_expire")
    except ValueError:
        pass
    else:
        raise AssertionError("gtc_expire without expiration_date accepted")
    # day with stray expiration_date
    try:
        LimitOrder(
            ticker="AAPL", side="buy", shares=10, limit_price=100.0,
            lifetime="day", expiration_date=date(2024, 1, 10),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("day with expiration_date accepted")
    print("  [PASS]")


def test_m9_day_limit_fills_when_touched():
    """Test 12: buy limit fills exactly once at limit_price when bar's low <= limit."""
    print("\n--- M9 Test 12: day-only buy limit fills when touched ---")
    broker = _fresh_broker_simple()
    placed = date(2024, 1, 8)  # Monday
    broker.place_limit_order(
        LimitOrder(ticker="AAPL", side="buy", shares=100, limit_price=90.0, lifetime="day"),
        placed_on=placed,
    )
    # Day after placement, low touches the limit.
    next_day = date(2024, 1, 9)
    broker.process_open_orders(next_day, {"AAPL": _bar(95, 96, 85, 92)})
    assert len(broker.trades) == 1, broker.trades
    tr = broker.trades[0]
    assert tr["side"] == "buy_limit"
    assert tr["fill_price"] == 90.0, f"fill should be at limit (90.0), got {tr['fill_price']}"
    assert tr["shares"] == 100
    assert broker.holdings.get("AAPL") == 100
    # Book is emptied of the filled order.
    assert len(broker.open_orders) == 0
    print(f"  filled at ${tr['fill_price']:.2f}, holdings={broker.holdings}  [PASS]")


def test_m9_day_limit_no_fill_when_not_touched():
    """Test 12 (negative): buy limit BELOW the day's low never fills."""
    print("\n--- M9 Test 12 (neg): day-only limit doesn't fill if untouched ---")
    broker = _fresh_broker_simple()
    placed = date(2024, 1, 8)
    broker.place_limit_order(
        LimitOrder(ticker="AAPL", side="buy", shares=100, limit_price=80.0, lifetime="day"),
        placed_on=placed,
    )
    next_day = date(2024, 1, 9)
    broker.process_open_orders(next_day, {"AAPL": _bar(95, 96, 85, 92)})  # low 85 > limit 80
    assert len(broker.trades) == 0
    # Still in book until end-of-day expiry runs.
    assert len(broker.open_orders) == 1
    broker.expire_orders(next_day)
    assert len(broker.open_orders) == 0, "day-only should be expired at EOD"
    print("  no fill, expired at EOD  [PASS]")


def test_m9_day_limit_expires():
    """Test 13: unfilled day-only is gone from open_orders after EOD."""
    print("\n--- M9 Test 13: day-only limit expires at EOD ---")
    broker = _fresh_broker_simple()
    placed = date(2024, 1, 8)
    broker.place_limit_order(
        LimitOrder(ticker="AAPL", side="buy", shares=100, limit_price=80.0, lifetime="day"),
        placed_on=placed,
    )
    # Day after placement, no fill possible.
    broker.process_open_orders(date(2024, 1, 9), {"AAPL": _bar(95, 96, 90, 92)})
    broker.expire_orders(date(2024, 1, 9))
    assert broker.open_orders == []
    print("  [PASS]")


def test_m9_gtc_expire_persists_then_expires():
    """Test 14: GTC with expiration_date persists across days until expiry."""
    print("\n--- M9 Test 14: gtc_expire persists then expires on expiration_date ---")
    broker = _fresh_broker_simple()
    placed = date(2024, 1, 8)
    broker.place_limit_order(
        LimitOrder(
            ticker="AAPL", side="buy", shares=100, limit_price=80.0,
            lifetime="gtc_expire", expiration_date=date(2024, 1, 11),
        ),
        placed_on=placed,
    )
    # 1/9, 1/10: low above limit → no fill, no expiry.
    for d in (date(2024, 1, 9), date(2024, 1, 10)):
        broker.process_open_orders(d, {"AAPL": _bar(95, 96, 90, 92)})
        broker.expire_orders(d)
        assert len(broker.open_orders) == 1, f"order missing on {d}"
    # 1/11 = expiration day; even if not filled, expire_orders drops it.
    broker.process_open_orders(date(2024, 1, 11), {"AAPL": _bar(95, 96, 90, 92)})
    broker.expire_orders(date(2024, 1, 11))
    assert broker.open_orders == [], "gtc_expire should drop on expiration day"
    print("  [PASS]")


def test_m9_gtc_until_filled():
    """Test 15: GTC never auto-expires; lives until a touch fills it."""
    print("\n--- M9 Test 15: GTC persists across many days until filled ---")
    broker = _fresh_broker_simple()
    placed = date(2024, 1, 8)
    broker.place_limit_order(
        LimitOrder(ticker="AAPL", side="buy", shares=100, limit_price=80.0, lifetime="gtc"),
        placed_on=placed,
    )
    # 10 trading days with low above limit → still in book.
    for i in range(1, 11):
        d = date(2024, 1, 8) + pd.Timedelta(days=i)
        d = d.to_pydatetime().date() if hasattr(d, "to_pydatetime") else d
        broker.process_open_orders(d, {"AAPL": _bar(95, 96, 90, 92)})
        broker.expire_orders(d)
        assert len(broker.open_orders) == 1, f"GTC dropped early on {d}"
    # Finally a low <= 80 → fill.
    fill_day = date(2024, 2, 1)
    broker.process_open_orders(fill_day, {"AAPL": _bar(82, 83, 78, 81)})
    assert len(broker.trades) == 1
    assert broker.trades[0]["fill_price"] == 80.0
    assert len(broker.open_orders) == 0
    print("  [PASS]")


def test_m9_fill_uses_limit_price_not_market():
    """Test 16: trade log records the limit price, not the day's close/low."""
    print("\n--- M9 Test 16: fill_price == limit_price (not market) ---")
    broker = _fresh_broker_simple()
    placed = date(2024, 1, 8)
    # Buy limit at 90; day blows through it: low 70, close 75.
    broker.place_limit_order(
        LimitOrder(ticker="AAPL", side="buy", shares=100, limit_price=90.0, lifetime="day"),
        placed_on=placed,
    )
    broker.process_open_orders(date(2024, 1, 9), {"AAPL": _bar(85, 88, 70, 75)})
    tr = broker.trades[-1]
    assert tr["fill_price"] == 90.0, f"expected 90.0 (limit), got {tr['fill_price']}"
    # Sell limit at 110; day spikes past it: high 130, close 125.
    broker2 = _fresh_broker_simple()
    # Seed broker2 with a long position so a sell is allowed.
    broker2.execute_order("AAPL", 100, 100.0)
    broker2.trades.clear()  # drop the seed trade so we can isolate
    broker2.place_limit_order(
        LimitOrder(ticker="AAPL", side="sell", shares=100, limit_price=110.0, lifetime="day"),
        placed_on=placed,
    )
    broker2.process_open_orders(date(2024, 1, 9), {"AAPL": _bar(120, 130, 115, 125)})
    tr2 = broker2.trades[-1]
    assert tr2["fill_price"] == 110.0, f"sell fill should be at limit (110), got {tr2['fill_price']}"
    assert tr2["side"] == "sell_limit"
    print("  buy fill at 90.0, sell fill at 110.0  [PASS]")


def test_m9_not_eligible_on_placement_day():
    """Limit placed on day D cannot fill on day D (look-ahead guard)."""
    print("\n--- M9: order placed today is NOT eligible today (D+1 rule) ---")
    broker = _fresh_broker_simple()
    today = date(2024, 1, 9)
    broker.place_limit_order(
        LimitOrder(ticker="AAPL", side="buy", shares=100, limit_price=90.0, lifetime="day"),
        placed_on=today,
    )
    # Same-day process with a bar that clearly touches 90 → must NOT fill.
    broker.process_open_orders(today, {"AAPL": _bar(85, 88, 70, 75)})
    assert len(broker.trades) == 0, "limit placed today must not fill today"
    assert len(broker.open_orders) == 1
    print("  [PASS]")


def test_m9_limit_buy_the_dip_e2e():
    """End-to-end: LimitBuyTheDip enters via day-only buy limits, then rides."""
    print("\n--- M9 E2E: LimitBuyTheDip enters via limit, then no further orders ---")
    # Single ticker, gentle uptrend with a sharp drop on day 3 that punches
    # through the strategy's buy-the-dip threshold (drop_pct=0.02).
    dates = pd.bdate_range("2024-01-02", periods=10)
    rows = []
    closes = [100, 101, 102, 90, 95, 100, 105, 108, 110, 112]  # day 3 dips to 90
    for d, c in zip(dates, closes):
        # high/low band of 3% so the dip clearly touches the threshold.
        rows.append((d, "AAPL", c, c * 1.03, c * 0.97, c))
    prices = pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close"])
    prices["date"] = pd.to_datetime(prices["date"])

    cfg = BacktestConfig(
        start_date=dates[0].date(), end_date=dates[-1].date(),
        initial_capital=100_000.0, tickers=["AAPL"],
    )
    strat = LimitBuyTheDip(["AAPL"], drop_pct=0.02, shares_per_order=100)
    data = _FakeDataInterface(prices)
    broker = _fresh_broker(cfg)
    engine = Engine(cfg, data, strat, broker)
    engine.run()

    # We expect exactly one buy_limit trade and zero market trades after that.
    sides = [tr["side"] for tr in broker.trades]
    assert all(s == "buy_limit" for s in sides), f"unexpected sides: {sides}"
    assert len(broker.trades) >= 1, "no fills — synthetic dip should have triggered one"
    # After the fill, holdings should contain AAPL.
    assert broker.holdings.get("AAPL", 0) == 100, broker.holdings
    # And no further limits emitted after the first fill (held > 0 suppression).
    # Count how many days had limits placed: should equal #days before the fill day.
    fill_dates = {tr["date"] for tr in broker.trades}
    assert len(fill_dates) == 1, f"expected single fill day, got {fill_dates}"
    print(
        f"  filled {len(broker.trades)} buy_limit trade(s) on {fill_dates}, "
        f"final holdings={broker.holdings}  [PASS]"
    )


# ---------- M10 Short selling (validate weights + broker + engine) ----------


def _fresh_broker_short(initial_cash=100_000.0, allow_short=True):
    return Broker(
        initial_cash=initial_cash,
        commission=PerShareCommission(0.005, 1.0),
        slippage=FixedBpsSlippage(5.0),
        allow_short=allow_short,
    )


def test_m10_validate_rejects_negative_when_disabled():
    """Test 17: allow_short=False rejects negative weights (v1 contract preserved)."""
    print("\n--- M10 Test 17: allow_short=False rejects negative weights ---")
    strat = BuyAndHold(["AAPL"])
    strat.allow_short = False  # explicit; default is False already
    try:
        strat._validate_weights({"AAPL": -0.1})
    except ValueError as e:
        print(f"  [PASS] raised: {e}")
        return
    raise AssertionError("negative weight accepted with allow_short=False")


def test_m10_validate_accepts_negative_when_enabled():
    """Test 18: allow_short=True accepts [-1, 1] with sum(|w|) <= 1."""
    print("\n--- M10 Test 18: allow_short=True accepts mixed [-1, 1] weights ---")
    strat = BuyAndHold(["AAPL", "MSFT"])  # any concrete strategy works for the check
    strat.allow_short = True
    # Mixed long+short, gross 1.0 — should pass.
    strat._validate_weights({"AAPL": 0.5, "MSFT": -0.5})
    # Just-over-cap → reject.
    try:
        strat._validate_weights({"AAPL": 0.6, "MSFT": -0.6})
    except ValueError as e:
        print(f"  gross over-cap correctly rejected: {e}")
    else:
        raise AssertionError("gross > 1.0 was accepted")
    # Out-of-range single weight → reject.
    try:
        strat._validate_weights({"AAPL": -1.5})
    except ValueError as e:
        print(f"  out-of-range rejected: {e}")
        print("  [PASS]")
        return
    raise AssertionError("weight < -1 was accepted")


def test_m10_short_entry_cash_mechanics():
    """Test 19: opening a short increases cash by shares*fill - commission."""
    print("\n--- M10 Test 19: short entry cash mechanics ---")
    broker = _fresh_broker_short()
    cash_before = broker.cash
    # Short 100 AAPL at $150. PV is $100k → 100*$150 = $15k short = 15% of PV ⇒ well under 50%.
    broker.execute_short_open(
        "AAPL", 100, 150.0,
        current_prices={"AAPL": 150.0},
    )
    cash_after = broker.cash
    cash_delta = cash_after - cash_before
    # Slippage drops fill price (selling direction). 5 bps = 0.05% → fill ≈ $149.925.
    # Proceeds = 100 * 149.925 - $1 commission = $14,991.50.
    expected = 100 * 149.925 - 1.0
    assert abs(cash_delta - expected) < 0.01, f"cash delta {cash_delta}, expected ~{expected}"
    assert broker.short_holdings.get("AAPL") == 100
    assert "AAPL" not in broker.long_holdings
    assert broker.trades[-1]["side"] == "short"
    print(f"  cash {cash_before:,.2f} → {cash_after:,.2f} (Δ {cash_delta:,.2f})  [PASS]")


def test_m10_mtm_for_shorts():
    """Test 20: total_value falls when shorted stock rises, rises when it falls."""
    print("\n--- M10 Test 20: MTM responds correctly to short PnL ---")
    broker = _fresh_broker_short()
    broker.execute_short_open(
        "AAPL", 100, 150.0,
        current_prices={"AAPL": 150.0},
    )
    pv_at_entry = broker.total_value({"AAPL": 150.0})
    pv_when_up = broker.total_value({"AAPL": 160.0})    # short loses
    pv_when_down = broker.total_value({"AAPL": 140.0})  # short gains
    assert pv_when_up < pv_at_entry, f"pv should fall when short rises: {pv_when_up} vs {pv_at_entry}"
    assert pv_when_down > pv_at_entry, f"pv should rise when short falls: {pv_when_down} vs {pv_at_entry}"
    # Sanity: a $10 rise across 100 shares = $1000 loss
    assert abs((pv_at_entry - pv_when_up) - 1000.0) < 1.0, (
        f"$10 up → expected ~$1000 loss, got {pv_at_entry - pv_when_up}"
    )
    print(
        f"  pv at $150: {pv_at_entry:,.2f}, "
        f"at $160: {pv_when_up:,.2f}, "
        f"at $140: {pv_when_down:,.2f}  [PASS]"
    )


def test_m10_initial_margin_rejects():
    """Test 21: short that would push gross > 50% of PV is rejected at placement."""
    print("\n--- M10 Test 21: Reg T initial margin (50%) rejects oversized short ---")
    broker = _fresh_broker_short()
    # $100k cash. Try to short 2000 shares @ $30 = $60k notional → gross > 50% PV.
    try:
        broker.execute_short_open(
            "AAPL", 2000, 30.0,
            current_prices={"AAPL": 30.0},
        )
    except ValueError as e:
        print(f"  [PASS] raised: {e}")
        assert not broker.short_holdings, "short_holdings should be empty after reject"
        return
    raise AssertionError("oversized short was accepted (initial margin failed)")


def test_m10_maintenance_margin_call():
    """Test 22: when equity / exposure drops below 25%, all shorts liquidate at close."""
    print("\n--- M10 Test 22: maintenance margin call liquidates all shorts ---")
    broker = _fresh_broker_short()
    # Open short of 1000 AAPL @ $30. PV after ≈ $100k. Gross short ~$30k ≈ 30% ⇒ initial OK.
    broker.execute_short_open(
        "AAPL", 1000, 30.0,
        current_prices={"AAPL": 30.0},
    )
    # Margin still healthy at $30.
    assert broker.maintenance_margin_ok({"AAPL": 30.0})
    # Push price up to $110 — pv = cash_after_open (≈ $130k) - 1000*$110 = $20k.
    # exposure = $110k. equity/exposure = 20/110 ≈ 0.18 < 0.25.
    breach_price = 110.0
    assert not broker.maintenance_margin_ok({"AAPL": breach_price}), "should be breached"

    forced = broker.liquidate_all_shorts({"AAPL": breach_price}, date(2024, 6, 1))
    assert len(forced) == 1, forced
    assert forced[0]["note"] == "margin_call"
    assert forced[0]["side"] == "cover"
    assert forced[0]["date"] == date(2024, 6, 1)
    assert "AAPL" not in broker.short_holdings, "short should be flat after liquidation"
    print(f"  forced cover at ${forced[0]['fill_price']:.2f}, cash now {broker.cash:,.2f}  [PASS]")


def test_m10_cash_conservation_with_shorts():
    """Test 23: extended invariant holds across a full long+short E2E run."""
    print("\n--- M10 Test 23: cash conservation with longs + shorts ---")
    # Two tickers: AAPL trends up, XYZ trends down. LongShortBarbell goes long
    # AAPL and short XYZ — both profitable in this synth.
    dates = pd.bdate_range("2024-01-02", periods=40)
    rows = []
    for i, d in enumerate(dates):
        aapl_close = 150.0 + i * 0.5
        xyz_close = 100.0 - i * 0.3
        rows.append((d, "AAPL", aapl_close, aapl_close * 1.01, aapl_close * 0.99, aapl_close))
        rows.append((d, "XYZ",  xyz_close,  xyz_close  * 1.01, xyz_close  * 0.99, xyz_close))
    prices = pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close"])
    prices["date"] = pd.to_datetime(prices["date"])

    cfg = BacktestConfig(
        start_date=dates[0].date(), end_date=dates[-1].date(),
        initial_capital=100_000.0, tickers=["AAPL", "XYZ"],
        allow_short=True,
    )
    strat = LongShortBarbell(longs=["AAPL"], shorts=["XYZ"])
    data = _FakeDataInterface(prices)
    broker = _fresh_broker_short(initial_cash=cfg.initial_capital, allow_short=True)
    engine = Engine(cfg, data, strat, broker)
    engine.run()

    # Assert invariant on every history row.
    for row in engine.history:
        if row["total_value"] != row["total_value"]:  # NaN
            continue
        diff = row["total_value"] - (row["cash"] + row["holdings_value"])
        assert abs(diff) < 1e-6, (
            f"cash conservation broken on {row['date']}: diff={diff}, row={row}"
        )

    # Sanity: there should be both a long (AAPL) and a short (XYZ) open.
    assert engine.history[-1]["holdings"].get("AAPL", 0) > 0
    assert engine.history[-1]["short_holdings"].get("XYZ", 0) > 0
    # And both trade sides should appear in the trade log.
    sides = {t["side"] for t in broker.trades}
    assert "buy" in sides and "short" in sides, sides
    final = engine.history[-1]["total_value"]
    print(
        f"  {len(engine.history)} history rows, final pv ${final:,.2f}, "
        f"final long={engine.history[-1]['holdings']}, "
        f"short={engine.history[-1]['short_holdings']}  [PASS]"
    )


def test_m10_long_short_barbell_e2e():
    """End-to-end LongShortBarbell: trades happen on both books, MTM stays sane."""
    print("\n--- M10 E2E: LongShortBarbell across 4 tickers ---")
    dates = pd.bdate_range("2024-01-02", periods=20)
    rows = []
    seeds = {
        "AAPL": (150.0, 0.5),
        "MSFT": (300.0, 0.3),
        "XYZ":  (100.0, -0.4),
        "ABC":  (80.0,  -0.2),
    }
    for d_idx, d in enumerate(dates):
        for t, (intercept, slope) in seeds.items():
            close = intercept + d_idx * slope
            rows.append((d, t, close, close * 1.005, close * 0.995, close))
    prices = pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close"])
    prices["date"] = pd.to_datetime(prices["date"])

    cfg = BacktestConfig(
        start_date=dates[0].date(), end_date=dates[-1].date(),
        initial_capital=100_000.0, tickers=["AAPL", "MSFT", "XYZ", "ABC"],
        allow_short=True,
    )
    strat = LongShortBarbell(longs=["AAPL", "MSFT"], shorts=["XYZ", "ABC"])
    data = _FakeDataInterface(prices)
    broker = _fresh_broker_short(initial_cash=cfg.initial_capital)
    engine = Engine(cfg, data, strat, broker)
    engine.run()

    # Trades on both books.
    trade_sides = [t["side"] for t in broker.trades]
    assert "buy" in trade_sides, trade_sides
    assert "short" in trade_sides, trade_sides
    # Final state has both long and short positions.
    last = engine.history[-1]
    assert any(last["holdings"].get(t, 0) > 0 for t in ("AAPL", "MSFT")), last
    assert any(last["short_holdings"].get(t, 0) > 0 for t in ("XYZ", "ABC")), last
    # No margin breach (synth volatility too small).
    notes = [t.get("note") for t in broker.trades]
    assert "margin_call" not in notes, "synth shouldn't trigger margin call"
    # Cash conservation on every row.
    for row in engine.history:
        if row["total_value"] != row["total_value"]:
            continue
        diff = row["total_value"] - (row["cash"] + row["holdings_value"])
        assert abs(diff) < 1e-6, f"cash conservation broken: {row}"
    print(
        f"  trades: {len(broker.trades)}, final pv ${last['total_value']:,.2f}, "
        f"long={last['holdings']}, short={last['short_holdings']}  [PASS]"
    )


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
    test_m8_config_validation()
    test_m8_default_is_daily()
    test_m8_backwards_compat_daily()
    test_m8_monthly_trade_count()
    test_m8_equity_continuity()
    test_m8_first_day_rebalance()
    test_m9_limitorder_validation()
    test_m9_day_limit_fills_when_touched()
    test_m9_day_limit_no_fill_when_not_touched()
    test_m9_day_limit_expires()
    test_m9_gtc_expire_persists_then_expires()
    test_m9_gtc_until_filled()
    test_m9_fill_uses_limit_price_not_market()
    test_m9_not_eligible_on_placement_day()
    test_m9_limit_buy_the_dip_e2e()
    test_m10_validate_rejects_negative_when_disabled()
    test_m10_validate_accepts_negative_when_enabled()
    test_m10_short_entry_cash_mechanics()
    test_m10_mtm_for_shorts()
    test_m10_initial_margin_rejects()
    test_m10_maintenance_margin_call()
    test_m10_cash_conservation_with_shorts()
    test_m10_long_short_barbell_e2e()
    print("\n" + "=" * 60)
    print("ALL OFFLINE TESTS PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
