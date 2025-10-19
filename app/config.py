"""Configuration management for the retirement planner app."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import os


@dataclass
class AppConfig:
    """Application configuration."""

    # Data settings
    default_tickers: List[str] = None
    default_weights: List[float] = None
    default_start_date: str = "1980-01-01"
    default_end_date: str = None

    # Simulation settings
    default_frequency: str = "monthly"
    default_mc_paths: int = 1000
    default_seed: int = 42
    max_historical_starts: int = 300

    # UI settings
    default_inflation: float = 0.025
    default_current_age: int = 35
    default_retire_age: int = 65
    default_plan_until_age: int = 95
    default_initial_balance: float = 200000.0
    default_annual_contrib: float = 10000.0
    default_annual_spend: float = 40000.0

    # Ticker mappings for historical data
    historical_mappings: Dict[str, str] = None

    def __post_init__(self):
        """Set default values after initialization."""
        if self.default_tickers is None:
            self.default_tickers = ["^GSPC", "^TNX"]
        if self.default_weights is None:
            self.default_weights = [0.6, 0.4]
        if self.default_end_date is None:
            from datetime import date

            self.default_end_date = date.today().isoformat()
        if self.historical_mappings is None:
            self.historical_mappings = {
                # ETFs to indices
                "SPY": "^GSPC",
                "QQQ": "^IXIC",
                "IWM": "^RUT",
                "EFA": "^EFA",
                "EEM": "^EEM",
                # ETFs to mutual funds
                "VTI": "VTSMX",
                "BND": "VBMFX",
                "VEA": "VDMIX",
                "VWO": "VEIEX",
                "VGLT": "VUSTX",
                "VGSH": "VFISX",
            }


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the application configuration."""
    return config


def update_config(**kwargs) -> None:
    """Update configuration with new values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")
