"""Data models and type definitions for Financial Independence, Retire Early (FIRE).

This module defines all data structures used throughout the application:
- Simulation parameters and results
- Portfolio state management
- Market data structures
- Custom exception classes

Key features:
- Type-safe data models with validation
- Comprehensive simulation result tracking
- Error handling with custom exceptions
- Support for different simulation frequencies
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PortfolioState:
    """Portfolio state at a point in time."""

    balance: float
    weights: np.ndarray  # target weights, sum to 1
    asset_values: Optional[np.ndarray] = None  # actual asset values (for rebalancing)


@dataclass
class ExpenseCategory:
    """Expense category with current amount or percentage."""

    name: str
    current_amount: Optional[float] = None  # Dollar amount if specified
    percentage: Optional[float] = None  # Percentage of total if specified

    def __post_init__(self):
        """Validate that either amount or percentage is provided."""
        if self.current_amount is None and self.percentage is None:
            raise ValueError(
                f"ExpenseCategory '{self.name}' must have either current_amount or percentage"
            )


@dataclass
class SavingsRateProfile:
    """Age-based savings rate profile for variable savings rates over time."""

    age_ranges: List[Tuple[int, int]]  # (start_age, end_age) inclusive
    rates: List[float]  # savings rate for each range (0.0-1.0)

    def __post_init__(self):
        """Validate that age ranges and rates match."""
        if len(self.age_ranges) != len(self.rates):
            raise ValueError("age_ranges and rates must have the same length")
        if not all(0.0 <= rate <= 1.0 for rate in self.rates):
            raise ValueError("All rates must be between 0.0 and 1.0")
        # Validate age ranges don't overlap (simplified check)
        sorted_ranges = sorted(self.age_ranges, key=lambda x: x[0])
        for i in range(len(sorted_ranges) - 1):
            if sorted_ranges[i][1] >= sorted_ranges[i + 1][0]:
                raise ValueError("Age ranges must not overlap")

    def get_rate_for_age(self, age: int) -> Optional[float]:
        """Get savings rate for a specific age."""
        for (start_age, end_age), rate in zip(self.age_ranges, self.rates):
            if start_age <= age <= end_age:
                return rate
        return None


@dataclass
class WithdrawalParams:
    """Parameters for dynamic withdrawal calculation."""

    expense_categories: List[ExpenseCategory]
    current_wage: Optional[float] = None
    education_level: Optional[str] = None
    use_cpi_adjustment: bool = True
    use_wage_adjustment: bool = False  # For future implementation
    total_annual_expense: Optional[float] = None  # Total if using percentage mode

    def __post_init__(self):
        """Validate expense categories."""
        if not self.expense_categories:
            raise ValueError("At least one expense category must be provided")

        # If using percentage mode, ensure percentages sum to 100
        if any(cat.percentage is not None for cat in self.expense_categories):
            total_pct = sum(
                cat.percentage
                for cat in self.expense_categories
                if cat.percentage is not None
            )
            if not np.isclose(total_pct, 100.0, atol=0.1):
                raise ValueError(
                    f"Expense category percentages must sum to 100%, got {total_pct}%"
                )


@dataclass
class SimulationParams:
    """Parameters for retirement simulation."""

    initial_balance: float
    annual_contrib: float
    annual_spend: float  # Kept for backward compatibility
    pre_retire_years: int
    retire_years: int
    inflation_rate_annual: float
    frequency: str
    pacing: str = "pro-rata"
    withdrawal_params: Optional[
        WithdrawalParams
    ] = None  # New dynamic withdrawal params
    use_wage_based_savings: bool = False  # Use wage growth for contributions
    savings_rate: Optional[
        float
    ] = None  # Percentage of wage to save (0.0-1.0) - used if savings_rate_profile is None
    savings_rate_profile: Optional[
        SavingsRateProfile
    ] = None  # Age-based savings rate profile
    education_level: Optional[str] = None  # Education level for wage growth
    current_age: Optional[int] = None  # Current age for wage projections
    current_year: Optional[int] = None  # Current year for wage projections
    use_wage_based_spending: bool = (
        False  # Calculate retirement spending from final wage
    )
    replacement_ratio: Optional[
        float
    ] = None  # Percentage of final wage for retirement spending (default 0.80)
    pre_retire_spending_tracked: bool = (
        False  # Whether to track pre-retirement spending
    )


@dataclass
class SimulationResult:
    """Results from a simulation run."""

    success_rate: float
    terminal_balances: np.ndarray
    median_path: np.ndarray
    p10_path: np.ndarray
    p90_path: np.ndarray
    periods_per_year: int
    horizon_periods: int
    requested_periods: Optional[int] = None
    data_limited: bool = False
    available_years: Optional[float] = None
    sample_paths: Optional[
        np.ndarray
    ] = None  # Sample individual paths for visualization
    spending_over_time: Optional[
        np.ndarray
    ] = None  # Median spending per period (for visualization)
    returns_over_time: Optional[
        np.ndarray
    ] = None  # Median returns per period (for visualization)
    rebalancing_events: Optional[List[str]] = None  # List of rebalancing event messages
    pre_retire_avg_spending: Optional[
        float
    ] = None  # Average annual spending during accumulation
    pre_retire_spending_by_year: Optional[
        np.ndarray
    ] = None  # Median spending by accumulation year
    earliest_retirement_ages: Optional[
        np.ndarray
    ] = None  # Earliest retirement age for each path (based on 25x expenses rule)


@dataclass
class DataConfig:
    """Configuration for data fetching."""

    tickers: List[str]
    start_date: str
    end_date: str
    frequency: str
    weights: Optional[np.ndarray] = None


@dataclass
class MarketData:
    """Market data container."""

    returns_df: pd.DataFrame
    means: np.ndarray
    cov: np.ndarray
    weights: np.ndarray
    frequency: str

    def __post_init__(self):
        """Validate data after initialization."""
        if len(self.weights) != len(self.returns_df.columns):
            raise ValueError(
                f"Weights length ({len(self.weights)}) must match "
                f"number of assets ({len(self.returns_df.columns)})"
            )

        if not np.isclose(self.weights.sum(), 1.0, atol=1e-6):
            raise ValueError("Weights must sum to 1.0")

        if self.means.shape[0] != len(self.returns_df.columns):
            raise ValueError(
                f"Means length ({self.means.shape[0]}) must match "
                f"number of assets ({len(self.returns_df.columns)})"
            )


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class DataError(Exception):
    """Custom exception for data-related errors."""

    pass


class SimulationError(Exception):
    """Custom exception for simulation errors."""

    pass
