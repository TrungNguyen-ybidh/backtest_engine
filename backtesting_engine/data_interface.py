"""DataInterface — translation layer from MySQL to pandas DataFrames.

Rules:
- No business logic here (no signal computation, no aggregation beyond SQL).
- Other modules NEVER write SQL directly; they go through this class.
- Fundamental tables filter by `filing_date`, not `date`.
- daily_prices.close is already split- and dividend-adjusted (yfinance
  auto_adjust=True default). Use it for return calculations directly.
"""

from datetime import date

import pandas as pd
from sqlalchemy import Engine, bindparam, inspect, text

_PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]

_FUNDAMENTAL_TABLES = {"income_stmt", "balance_sheet", "cashflow"}


class DataInterface:
    def __init__(self, engine: Engine):
        self.engine = engine
        self._conn = None

    def __enter__(self) -> "DataInterface":
        self._conn = self.engine.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self.engine.dispose()

    def _bind(self):
        return self._conn if self._conn is not None else self.engine

    def get_prices(
        self, tickers: list[str], start: date, end: date
    ) -> pd.DataFrame:
        if not tickers:
            return _empty(_PRICE_COLUMNS)

        sql = text(
            """
            SELECT date, ticker, open, high, low, close, volume
            FROM daily_prices
            WHERE ticker IN :tickers
              AND date >= :start
              AND date <= :end
            ORDER BY date ASC, ticker ASC
            """
        ).bindparams(bindparam("tickers", expanding=True))

        df = pd.read_sql(
            sql,
            self._bind(),
            params={"tickers": [t.upper() for t in tickers], "start": start, "end": end},
        )
        return _normalize(df, _PRICE_COLUMNS, numeric_cols=[
            "open", "high", "low", "close", "volume"
        ])

    def get_fundamentals(
        self, tickers: list[str], statement: str, as_of: date
    ) -> pd.DataFrame:
        if statement not in _FUNDAMENTAL_TABLES:
            raise ValueError(
                f"statement must be one of {sorted(_FUNDAMENTAL_TABLES)}, got {statement!r}"
            )

        cols = self._fundamental_columns(statement)
        if not tickers:
            return _empty(cols)

        sql = text(
            f"""
            SELECT *
            FROM {statement}
            WHERE ticker IN :tickers
              AND filing_date <= :as_of
            ORDER BY ticker ASC, filing_date ASC
            """
        ).bindparams(bindparam("tickers", expanding=True))

        df = pd.read_sql(
            sql,
            self._bind(),
            params={"tickers": [t.upper() for t in tickers], "as_of": as_of},
        )
        date_cols = [c for c in ("date", "filing_date", "accepted_date") if c in df.columns]
        numeric_cols = [
            c for c in df.columns
            if c not in {"ticker", "fiscal_year", "period", *date_cols}
        ]
        return _normalize(df, cols, numeric_cols=numeric_cols, date_cols=date_cols)

    def get_factors(self, start: date, end: date) -> pd.DataFrame:
        sql = text(
            """
            SELECT date, market_excess_return, size_factor, value_factor,
                   profitability_factor, inveatment_factor, risk_free_rate
            FROM factors_daily
            WHERE date >= :start AND date <= :end
            ORDER BY date ASC
            """
        )
        df = pd.read_sql(sql, self._bind(), params={"start": start, "end": end})
        df = df.rename(columns={"inveatment_factor": "investment_factor"})
        cols = [
            "date", "market_excess_return", "size_factor", "value_factor",
            "profitability_factor", "investment_factor", "risk_free_rate",
        ]
        numeric_cols = [c for c in cols if c != "date"]
        return _normalize(df, cols, numeric_cols=numeric_cols)

    def _fundamental_columns(self, table: str) -> list[str]:
        return [c["name"] for c in inspect(self.engine).get_columns(table)]


def _empty(columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({c: [] for c in columns})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _normalize(
    df: pd.DataFrame,
    columns: list[str],
    *,
    numeric_cols: list[str] = (),
    date_cols: list[str] = ("date",),
) -> pd.DataFrame:
    if df.empty:
        return _empty(columns)

    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    sort_cols = [c for c in ("date", "filing_date", "ticker") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df
