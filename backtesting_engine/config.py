"""BacktestConfig — central settings for a backtest run."""

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class BacktestConfig:
    start_date: date
    end_date: date
    initial_capital: float = 100_000.0
    tickers: list[str] = field(default_factory=list)
    benchmark: Optional[str] = None

    commission_per_share: float = 0.005
    commission_min: float = 1.0
    slippage_bps: float = 5.0

    db_url: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.start_date, date) or not isinstance(self.end_date, date):
            raise TypeError("start_date and end_date must be datetime.date instances")
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be before end_date ({self.end_date})"
            )
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        if not self.tickers:
            raise ValueError("tickers must contain at least one symbol")
        if any(not isinstance(t, str) or not t for t in self.tickers):
            raise ValueError("every ticker must be a non-empty string")
        if self.commission_per_share < 0:
            raise ValueError("commission_per_share must be >= 0")
        if self.commission_min < 0:
            raise ValueError("commission_min must be >= 0")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")

        self.tickers = [t.upper() for t in self.tickers]
        if self.benchmark is not None:
            self.benchmark = self.benchmark.upper()

        if self.db_url is None:
            self.db_url = os.environ.get("sql_path") or os.environ.get("DATABASE_URL")
