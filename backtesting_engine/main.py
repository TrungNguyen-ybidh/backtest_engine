"""Entry point — wire up a backtest and run it."""

from datetime import date

from dotenv import load_dotenv
from sqlalchemy import create_engine

from .analytics import Analytics
from .broker import Broker
from .config import BacktestConfig
from .data_interface import DataInterface
from .engine import Engine
from .models import FixedBpsSlippage, PerShareCommission
from .strategy import BuyAndHold


def main() -> None:
    load_dotenv()

    config = BacktestConfig(
        start_date=date(2000, 1, 1),
        end_date=date(2024, 1, 1),
        initial_capital=100_000.0,
        tickers=["NVDA"],
    )

    strategy = BuyAndHold(config.tickers)
    broker = Broker(
        initial_cash=config.initial_capital,
        commission=PerShareCommission(config.commission_per_share, config.commission_min),
        slippage=FixedBpsSlippage(config.slippage_bps),
    )

    sql_engine = create_engine(config.db_url)
    with DataInterface(sql_engine) as data:
        engine = Engine(config, data, strategy, broker)
        engine.run()

    Analytics(engine.history, trades=broker.trades, config=config).print_summary()


if __name__ == "__main__":
    main()
