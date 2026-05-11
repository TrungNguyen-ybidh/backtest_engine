"""Modular Python backtesting engine for systematic trading strategies.

Top-level re-exports for ergonomic usage:

    import backtesting_engine as bt
    cfg = bt.BacktestConfig(start_date=..., end_date=..., tickers=["AAPL"])

Long-form module imports also work and are equivalent:

    from backtesting_engine.config import BacktestConfig

Importing this package does NOT open any database connections. DataInterface
only connects when used as a context manager (`with DataInterface(engine):`).
"""

from .analytics import Analytics
from .broker import Broker
from .config import BacktestConfig
from .data_interface import DataInterface
from .engine import Engine
from .models import (
    CommissionModel,
    FixedBpsSlippage,
    LimitOrder,
    PerShareCommission,
    SlippageModel,
)
from .strategy import (
    AllCash,
    BuyAndHold,
    LimitBuyTheDip,
    LongShortBarbell,
    MovingAverageCrossover,
    Strategy,
    ValueScreen,
)

__version__ = "1.1.0"

__all__ = [
    # Configuration + orchestration
    "BacktestConfig",
    "DataInterface",
    "Engine",
    "Broker",
    "Analytics",
    # Cost models + order types
    "CommissionModel",
    "PerShareCommission",
    "SlippageModel",
    "FixedBpsSlippage",
    "LimitOrder",
    # Strategy base + concrete implementations
    "Strategy",
    "AllCash",
    "BuyAndHold",
    "MovingAverageCrossover",
    "ValueScreen",
    "LimitBuyTheDip",
    "LongShortBarbell",
    # Metadata
    "__version__",
]
