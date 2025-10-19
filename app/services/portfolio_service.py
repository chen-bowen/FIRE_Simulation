"""Portfolio management service."""

import numpy as np
import pandas as pd
from typing import Optional
from app.schemas import PortfolioState
from app.utils import per_period_amount, periods_per_year


class PortfolioService:
    """Service for portfolio management operations."""

    def __init__(self):
        pass

    def annual_rebalance_needed(self, period_index: pd.Timestamp, prev_index: Optional[pd.Timestamp]) -> bool:
        """Check if annual rebalancing is needed."""
        if prev_index is None:
            return True
        return period_index.year != prev_index.year

    def step_portfolio(
        self,
        state: PortfolioState,
        contrib: float,
        spend: float,
        inflation_rate_annual: float,
        period_return: np.ndarray,
        period_index: pd.Timestamp,
        prev_index: Optional[pd.Timestamp],
        freq: str,
    ) -> PortfolioState:
        """
        Step portfolio forward one period.

        Args:
            state: Current portfolio state
            contrib: Contribution amount for this period
            spend: Spending amount for this period
            inflation_rate_annual: Annual inflation rate
            period_return: Asset returns for this period
            period_index: Current period timestamp
            prev_index: Previous period timestamp
            freq: Frequency ('daily' or 'monthly')

        Returns:
            Updated portfolio state
        """
        # Apply contribution (pre-retire) or spending (retire) at start of period
        net_flow = contrib - spend
        new_balance = max(0.0, state.balance + net_flow)

        # Apply portfolio return (weighted)
        portfolio_ret = float(np.dot(state.weights, period_return))
        new_balance = max(0.0, new_balance * (1.0 + portfolio_ret))

        # Note: In this simplified model, we keep target weights
        # In a more sophisticated model, we would rebalance here

        return PortfolioState(balance=new_balance, weights=state.weights)

    def calculate_period_amounts(self, annual_contrib: float, annual_spend: float, freq: str) -> tuple[float, float]:
        """Calculate per-period contribution and spending amounts."""
        contrib_pp = per_period_amount(annual_contrib, freq)
        spend_pp = per_period_amount(annual_spend, freq)
        return contrib_pp, spend_pp

    def calculate_inflation_factor(self, inflation_rate_annual: float, freq: str) -> float:
        """Calculate per-period inflation factor."""
        ppy = periods_per_year(freq)
        return (1.0 + inflation_rate_annual) ** (1.0 / ppy) - 1.0
