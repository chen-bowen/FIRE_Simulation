"""Utility functions and constants for Financial Independence, Retire Early (FIRE).

This module provides helper functions used throughout the application:
- Input validation and data alignment
- Financial calculations and conversions
- Frequency and period calculations
- Weight validation and normalization

Key features:
- Comprehensive input validation
- Financial math utilities
- Data alignment and normalization
- Error handling with custom exceptions
"""

from typing import List, Tuple, Union

import numpy as np

from .schemas import ValidationError


def validate_frequency(freq: str) -> str:
    """Validate and normalize frequency string."""
    if freq not in {"daily", "monthly"}:
        raise ValidationError("Frequency must be 'daily' or 'monthly'")
    return freq


def periods_per_year(freq: str) -> int:
    """Get number of periods per year for given frequency."""
    return 252 if freq == "daily" else 12


def per_period_amount(annual_amount: float, freq: str) -> float:
    """Convert annual amount to per-period amount."""
    return annual_amount / float(periods_per_year(freq))


def validate_weights(
    weights: Union[List[float], np.ndarray], n_assets: int
) -> np.ndarray:
    """Validate and normalize weights."""
    weights = np.array(weights, dtype=float)

    if len(weights) != n_assets:
        raise ValidationError(
            f"Number of weights ({len(weights)}) must match "
            f"number of assets ({n_assets})"
        )

    if np.any(weights < 0):
        raise ValidationError("Weights must be non-negative")

    # Normalize weights
    weights = weights / weights.sum()

    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValidationError("Weights must sum to 1.0")

    return weights


def align_weights_with_data(weights: np.ndarray, data_columns: List[str]) -> np.ndarray:
    """Align weights with available data columns."""
    n_available = len(data_columns)
    n_weights = len(weights)

    # Start from a candidate weight vector that matches number of available assets
    if n_weights == n_available:
        aligned = np.array(weights, dtype=float)
    elif n_weights == 1:
        # Broadcast single weight across all assets, then normalize below
        aligned = np.repeat(float(weights[0]), n_available)
    else:
        # Mismatch: default to equal weights for all available assets
        aligned = np.repeat(1.0, n_available)

    # Normalize to sum to 1 and guard against division by zero
    total = float(np.sum(aligned))
    if total <= 0.0:
        aligned = np.repeat(1.0, n_available)
        total = float(np.sum(aligned))
    aligned = aligned / total

    return aligned


def format_currency(amount: float) -> str:
    """Format currency amount for display."""
    abs_amount = abs(amount)
    sign = "-" if amount < 0 else ""

    if abs_amount >= 1e6:
        return f"{sign}${abs_amount/1e6:.1f}M"
    elif abs_amount >= 1e3:
        return f"{sign}${abs_amount/1e3:.1f}K"
    else:
        return f"{sign}${abs_amount:.0f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage for display."""
    return f"{value*100:.{decimals}f}%"


def calculate_horizon_years(
    current_age: int, retire_age: int, plan_until_age: int
) -> Tuple[int, int]:
    """Calculate working and retirement years.

    Supports both pre-retirement and already-retired scenarios:
    - Pre-retirement: pre_retire_years = retire_age - current_age, retire_years = plan_until_age - retire_age
    - Already retired: pre_retire_years = 0, retire_years = plan_until_age - current_age
    """
    pre_retire_years = max(0, retire_age - current_age)

    # If already retired, retirement years is from current_age to plan_until_age
    if pre_retire_years == 0:
        retire_years = max(1, plan_until_age - current_age)
    else:
        retire_years = max(1, plan_until_age - retire_age)

    return pre_retire_years, retire_years


def validate_age_inputs(current_age: int, retire_age: int, plan_until_age: int) -> None:
    """Validate age inputs.

    Supports both pre-retirement and already-retired scenarios:
    - Pre-retirement: current_age < retire_age < plan_until_age
    - Already retired: current_age >= retire_age, plan_until_age > current_age
    """
    if current_age < 0 or retire_age < 0 or plan_until_age < 0:
        raise ValidationError("Ages must be non-negative")

    # Check if already retired
    is_already_retired = retire_age <= current_age

    if is_already_retired:
        # For already retired, plan_until_age must be greater than current_age
        if plan_until_age <= current_age:
            raise ValidationError(
                "Plan until age must be greater than current age (you are already retired)"
            )
    else:
        # For pre-retirement, standard validation applies
        if plan_until_age <= retire_age:
            raise ValidationError("Plan until age must be greater than retirement age")


def validate_financial_inputs(
    initial_balance: float, annual_contrib: float, annual_spend: float
) -> None:
    """Validate financial inputs."""
    if initial_balance < 0:
        raise ValidationError("Initial balance must be non-negative")

    if annual_contrib < 0:
        raise ValidationError("Annual contribution must be non-negative")

    if annual_spend < 0:
        raise ValidationError("Annual spending must be non-negative")


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    return numerator / denominator if denominator != 0 else default
