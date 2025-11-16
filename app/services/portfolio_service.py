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
        period_index: Optional[pd.Timestamp],
        prev_index: Optional[pd.Timestamp],
        freq: str,
        period_number: Optional[int] = None,
    ) -> tuple[PortfolioState, Optional[str]]:
        """
        Step portfolio forward one period.

        Args:
            state: Current portfolio state
            contrib: Contribution amount for this period
            spend: Spending amount for this period
            inflation_rate_annual: Annual inflation rate
            period_return: Asset returns for this period (per-asset returns)
            period_index: Current period timestamp
            prev_index: Previous period timestamp
            freq: Frequency ('daily' or 'monthly')

        Returns:
            Updated portfolio state
        """
        # Apply contribution (pre-retire) or spending (retire) at start of period
        net_flow = contrib - spend
        balance_before_return = max(0.0, state.balance + net_flow)

        # Initialize asset values if not present (first period or after rebalancing)
        if state.asset_values is None:
            # Initialize asset values based on target weights
            state.asset_values = balance_before_return * state.weights

        # Apply asset returns to individual assets
        if isinstance(period_return, np.ndarray):
            pr = np.squeeze(period_return)
            if pr.ndim == 0:
                # Scalar return - apply to all assets equally
                asset_returns = np.full(len(state.weights), float(pr))
            elif pr.shape[0] == state.weights.shape[0]:
                # Per-asset returns
                asset_returns = pr.astype(float)
            elif pr.shape[0] == 1:
                # Single return value - apply to all assets
                asset_returns = np.full(len(state.weights), float(pr[0]))
            elif state.weights.shape[0] == 1:
                # Single asset portfolio
                asset_returns = np.array(
                    [float(pr[0] if pr.ndim == 1 else pr.ravel()[0])]
                )
            else:
                # Fallback: use average return for all assets
                asset_returns = np.full(len(state.weights), float(np.mean(pr)))
        else:
            # Plain scalar - apply to all assets
            asset_returns = np.full(len(state.weights), float(period_return))

        # Apply returns to individual asset values
        new_asset_values = state.asset_values * (1.0 + asset_returns)
        new_balance = float(np.sum(new_asset_values))

        # Check if annual rebalancing is needed
        should_rebalance = False
        if period_index is not None and prev_index is not None:
            # Historical simulation with dates - rebalance annually
            should_rebalance = self.annual_rebalance_needed(period_index, prev_index)
        elif period_number is not None and period_number > 0:
            # Monte Carlo simulation - rebalance annually based on period number
            # Rebalance at the END of each year (period ppy-1, 2*ppy-1, etc.)
            # Skip period 0 to allow initial allocation to drift
            from app.utils import periods_per_year

            ppy = periods_per_year(freq)
            # Rebalance at the end of each year (not at the start)
            should_rebalance = ((period_number + 1) % ppy) == 0

        rebalancing_message = None
        if should_rebalance:
            # Rebalance to target weights
            new_asset_values = new_balance * state.weights
            # Create rebalancing event message
            if period_index is not None:
                rebalancing_message = f"Portfolio rebalanced on {period_index.strftime('%Y-%m-%d')} - Reset all assets to target weights"
            elif period_number is not None:
                from app.utils import periods_per_year

                ppy = periods_per_year(freq)
                year = (period_number + 1) // ppy
                rebalancing_message = f"Portfolio rebalanced at end of year {year} (period {period_number}) - Reset all assets to target weights"

        return (
            PortfolioState(
                balance=new_balance,
                weights=state.weights,
                asset_values=new_asset_values,
            ),
            rebalancing_message,
        )

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

        Note: If education_level is set in withdrawal_params, the expense category
        percentages already reflect education-based adjustments (applied in the UI).
        These adjustments are maintained throughout retirement via proportional
        inflation adjustments.

        Args:
            withdrawal_params: Withdrawal parameters with expense categories
                (may include education_level for education-based spending adjustments)

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
                # Note: If education_level is set, these percentages already reflect
                # education-based adjustments applied in the sidebar
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

        Note: This method maintains education-based spending adjustments throughout
        retirement by applying inflation proportionally to each category. The relative
        proportions established by education level (if set) are preserved.

        Args:
            initial_category_spending: Initial annual spending per category
                (may already reflect education-based adjustments)
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
            # This maintains the relative proportions established by education adjustments
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

        Note: Education-based spending adjustments (if education_level is set in
        withdrawal_params) are maintained throughout retirement. The initial_category_spending
        already reflects education-adjusted category distributions, and inflation adjustments
        preserve these relative proportions.

        Args:
            withdrawal_params: Withdrawal parameters (may include education_level)
            initial_category_spending: Initial annual spending per category
                (may already reflect education-based adjustments)
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
