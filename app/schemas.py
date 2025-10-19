"""Data models and type definitions for the retirement planner."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


@dataclass
class PortfolioState:
    """Portfolio state at a point in time."""

    balance: float
    weights: np.ndarray  # target weights, sum to 1


@dataclass
class SimulationParams:
    """Parameters for retirement simulation."""

    initial_balance: float
    annual_contrib: float
    annual_spend: float
    pre_retire_years: int
    retire_years: int
    inflation_rate_annual: float
    frequency: str
    pacing: str = "pro-rata"


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
    sample_paths: Optional[np.ndarray] = None  # Sample individual paths for visualization


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
            raise ValueError(f"Weights length ({len(self.weights)}) must match " f"number of assets ({len(self.returns_df.columns)})")

        if not np.isclose(self.weights.sum(), 1.0, atol=1e-6):
            raise ValueError("Weights must sum to 1.0")

        if self.means.shape[0] != len(self.returns_df.columns):
            raise ValueError(f"Means length ({self.means.shape[0]}) must match " f"number of assets ({len(self.returns_df.columns)})")


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class DataError(Exception):
    """Custom exception for data-related errors."""

    pass


class SimulationError(Exception):
    """Custom exception for simulation errors."""

    pass
