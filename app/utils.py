"""Utility functions and constants for the retirement planner.

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

import numpy as np
import pandas as pd
from typing import List, Tuple, Union
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


def validate_weights(weights: Union[List[float], np.ndarray], n_assets: int) -> np.ndarray:
    """Validate and normalize weights."""
    weights = np.array(weights, dtype=float)

    if len(weights) != n_assets:
        raise ValidationError(f"Number of weights ({len(weights)}) must match " f"number of assets ({n_assets})")

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

    if n_weights == n_available:
        return weights / weights.sum()
    elif n_weights == 1:
        # Broadcast single weight
        return np.repeat(float(weights[0]), n_available)
    else:
        # Reset to equal weights
        return np.repeat(1.0 / n_available, n_available)


def format_currency(amount: float) -> str:
    """Format currency amount for display."""
    if amount >= 1e6:
        return f"${amount/1e6:.1f}M"
    elif amount >= 1e3:
        return f"${amount/1e3:.1f}K"
    else:
        return f"${amount:.0f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage for display."""
    return f"{value*100:.{decimals}f}%"


def calculate_horizon_years(current_age: int, retire_age: int, plan_until_age: int) -> Tuple[int, int]:
    """Calculate working and retirement years."""
    pre_retire_years = max(0, retire_age - current_age)
    retire_years = max(1, plan_until_age - retire_age)
    return pre_retire_years, retire_years


def validate_age_inputs(current_age: int, retire_age: int, plan_until_age: int) -> None:
    """Validate age inputs."""
    if current_age < 0 or retire_age < 0 or plan_until_age < 0:
        raise ValidationError("Ages must be non-negative")

    if retire_age <= current_age:
        raise ValidationError("Retirement age must be greater than current age")

    if plan_until_age <= retire_age:
        raise ValidationError("Plan until age must be greater than retirement age")


def validate_financial_inputs(initial_balance: float, annual_contrib: float, annual_spend: float) -> None:
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
