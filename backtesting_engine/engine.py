"""Engine — orchestrator that runs the daily loop.

Wires DataInterface, Strategy, and Broker. Engine does NOT execute trades
itself (that's Broker) and does NOT compute signals (that's Strategy).
"""

from .broker import Broker
from .config import BacktestConfig
from .data_interface import DataInterface
from .strategy import Strategy


class Engine:
    def __init__(
        self,
        config: BacktestConfig,
        data: DataInterface,
        strategy: Strategy,
        broker: Broker,
    ):
        self.config = config
        self.data = data
        self.strategy = strategy
        self.broker = broker
        self.history: list[dict] = []

    def run(self) -> None:
        raise NotImplementedError
