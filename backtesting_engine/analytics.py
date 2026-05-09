"""Analytics — performance metrics and visualizations.

Reports must disclose caveats per .claude/rules/output-limitations.md
(survivorship bias, etc.).
"""

import pandas as pd


class Analytics:
    def __init__(self, history: list[dict]):
        self.history = pd.DataFrame(history)

    def print_summary(self) -> None:
        raise NotImplementedError

    def plot_equity_curve(self):
        raise NotImplementedError

    def plot_trades(self, ticker: str):
        raise NotImplementedError
