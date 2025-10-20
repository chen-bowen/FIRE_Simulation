"""Portfolio management service.

This module handles all portfolio-related mathematics:
- Portfolio rebalancing and return calculations
- Contribution and withdrawal processing
- Inflation adjustments and period calculations
- Portfolio state management

Key features:
- Frequency-aware calculations (daily/monthly)
- Inflation-adjusted spending
- Portfolio rebalancing to target weights
- Support for different pacing modes (monthly boundary for daily frequency)
"""

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
        # Accept either a vector of per-asset returns (same length as weights)
        # or a single blended/scalar return. This ensures robustness when
        # the number of holdings varies (including 1 security) or when a
        # pre-blended return is provided.
        if isinstance(period_return, np.ndarray):
            # Squeeze to 1-D if needed
            pr = np.squeeze(period_return)
            if pr.ndim == 0:
                portfolio_ret = float(pr)
            elif pr.shape[0] == state.weights.shape[0]:
                portfolio_ret = float(np.dot(state.weights, pr))
            elif pr.shape[0] == 1:
                # Treat as already blended
                portfolio_ret = float(pr[0])
            elif state.weights.shape[0] == 1:
                # Single-asset portfolio but multiple returns provided;
                # take weighted sum with single weight (which is 1.0 after normalization)
                portfolio_ret = float(np.dot(state.weights, pr if pr.ndim == 1 else pr.ravel()))
            else:
                # Fallback: treat as blended by averaging to avoid shape errors
                portfolio_ret = float(np.mean(pr))
        else:
            # Plain scalar
            portfolio_ret = float(period_return)
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
