"""Entry point — wire up a backtest and run it."""

from datetime import date

from .broker import Broker
from .config import BacktestConfig
from .engine import Engine
from .models import FixedBpsSlippage, PerShareCommission
from .strategy import BuyAndHold


def main() -> None:
    config = BacktestConfig(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 1, 1),
        initial_capital=100_000.0,
        tickers=["AAPL"],
    )

    # data = DataInterface(create_engine(config.db_url))
    strategy = BuyAndHold(config.tickers)
    broker = Broker(
        initial_cash=config.initial_capital,
        commission=PerShareCommission(config.commission_per_share, config.commission_min),
        slippage=FixedBpsSlippage(config.slippage_bps),
    )

    # engine = Engine(config, data, strategy, broker)
    # engine.run()
    # Analytics(engine.history).print_summary()


if __name__ == "__main__":
    main()
