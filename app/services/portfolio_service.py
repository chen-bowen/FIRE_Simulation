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

from typing import Dict, Optional

import numpy as np
import pandas as pd

from app.schemas import PortfolioState, WithdrawalParams
from app.utils import per_period_amount, periods_per_year


class PortfolioService:
    """Service for portfolio management operations."""

    def __init__(self):
        pass

    def annual_rebalance_needed(
        self, period_index: pd.Timestamp, prev_index: Optional[pd.Timestamp]
    ) -> bool:
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
                portfolio_ret = float(
                    np.dot(state.weights, pr if pr.ndim == 1 else pr.ravel())
                )
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

    def calculate_period_amounts(
        self, annual_contrib: float, annual_spend: float, freq: str
    ) -> tuple[float, float]:
        """Calculate per-period contribution and spending amounts."""
        contrib_pp = per_period_amount(annual_contrib, freq)
        spend_pp = per_period_amount(annual_spend, freq)
        return contrib_pp, spend_pp

    def calculate_inflation_factor(
        self, inflation_rate_annual: float, freq: str
    ) -> float:
        """Calculate per-period inflation factor."""
        ppy = periods_per_year(freq)
        return (1.0 + inflation_rate_annual) ** (1.0 / ppy) - 1.0

    def calculate_initial_category_spending(
        self, withdrawal_params: WithdrawalParams
    ) -> Dict[str, float]:
        """
        Calculate initial annual spending per category from withdrawal parameters.

        Args:
            withdrawal_params: Withdrawal parameters with expense categories

        Returns:
            Dictionary mapping category names to annual spending amounts
        """
        category_spending = {}

        for category in withdrawal_params.expense_categories:
            if category.current_amount is not None:
                # Direct dollar amount specified
                category_spending[category.name] = category.current_amount
            elif (
                category.percentage is not None
                and withdrawal_params.total_annual_expense
            ):
                # Percentage of total specified
                category_spending[category.name] = (
                    category.percentage / 100.0
                ) * withdrawal_params.total_annual_expense
            else:
                # Should not happen due to validation, but handle gracefully
                category_spending[category.name] = 0.0

        return category_spending

    def calculate_category_spending(
        self,
        initial_category_spending: Dict[str, float],
        years_into_retirement: float,
        inflation_rate_annual: float,
        freq: str,
        category_inflation_rates: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Calculate spending per category adjusted for inflation.

        Args:
            initial_category_spending: Initial annual spending per category
            years_into_retirement: Number of years into retirement
            inflation_rate_annual: Annual inflation rate to apply (fallback if category rates unavailable)
            freq: Frequency ('daily' or 'monthly')
            category_inflation_rates: Optional dict mapping category names to their specific inflation rates

        Returns:
            Dictionary mapping category names to adjusted annual spending amounts
        """
        adjusted_spending = {}

        for category, amount in initial_category_spending.items():
            # Use category-specific inflation rate if available, otherwise fall back to general rate
            if category_inflation_rates and category in category_inflation_rates:
                category_rate = category_inflation_rates[category]
            else:
                category_rate = inflation_rate_annual

            # Apply cumulative inflation: (1 + r)^t
            inflation_multiplier = (1.0 + category_rate) ** years_into_retirement
            adjusted_spending[category] = amount * inflation_multiplier

        return adjusted_spending

    def calculate_dynamic_withdrawal(
        self,
        withdrawal_params: WithdrawalParams,
        initial_category_spending: Dict[str, float],
        years_into_retirement: float,
        inflation_rate_annual: float,
        freq: str,
        category_inflation_rates: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate total dynamic withdrawal amount for a period.

        Args:
            withdrawal_params: Withdrawal parameters
            initial_category_spending: Initial annual spending per category
            years_into_retirement: Number of years into retirement
            inflation_rate_annual: Annual inflation rate (fallback if category rates unavailable)
            freq: Frequency ('daily' or 'monthly')
            category_inflation_rates: Optional dict mapping category names to their specific inflation rates

        Returns:
            Total annual withdrawal amount
        """
        if not withdrawal_params.use_cpi_adjustment:
            # Fallback to simple inflation if CPI adjustment disabled
            inflation_multiplier = (
                1.0 + inflation_rate_annual
            ) ** years_into_retirement
            total = sum(initial_category_spending.values()) * inflation_multiplier
            return total

        # Calculate adjusted spending per category (with category-specific rates if available)
        adjusted_category_spending = self.calculate_category_spending(
            initial_category_spending,
            years_into_retirement,
            inflation_rate_annual,
            freq,
            category_inflation_rates,
        )

        # Sum all categories
        total_withdrawal = sum(adjusted_category_spending.values())

        # Apply wage growth adjustment if enabled (future feature)
        if withdrawal_params.use_wage_adjustment:
            # TODO: Implement wage growth adjustment when wage data is available
            pass

        return total_withdrawal

    def calculate_period_withdrawal(
        self,
        annual_withdrawal: float,
        freq: str,
        pacing: str = "pro-rata",
        period_index: Optional[int] = None,
    ) -> float:
        """
        Convert annual withdrawal to per-period withdrawal.

        Args:
            annual_withdrawal: Annual withdrawal amount
            freq: Frequency ('daily' or 'monthly')
            pacing: Pacing mode ('pro-rata' or 'monthly-boundary')
            period_index: Period index (for monthly-boundary mode)

        Returns:
            Per-period withdrawal amount
        """
        ppy = periods_per_year(freq)

        if freq == "daily" and pacing == "monthly-boundary":
            # For monthly-boundary, apply full month's worth on boundary periods
            # This is handled in simulation service, so return monthly amount
            return annual_withdrawal / 12.0
        else:
            # Pro-rata: divide annual by periods per year
            return annual_withdrawal / ppy
