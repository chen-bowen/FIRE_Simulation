"""Data service for fetching and processing market data.

This module handles all data operations for the retirement planner:
- Yahoo Finance data fetching with caching
- Historical data backfilling for ETFs with limited history
- Market data calibration (means, covariances, correlations)
- Data validation and error handling

Key features:
- LRU caching for improved performance
- Automatic ticker mapping for historical backfill
- Robust error handling for data fetching failures
- Frequency-aware data processing (daily/monthly)
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from app.config import get_config
from app.schemas import DataError, MarketData
from app.utils import validate_frequency


class DataService:
    """Service for fetching and processing market data."""

    def __init__(self):
        self.config = get_config()
        self._cpi_data: Optional[pd.DataFrame] = None
        self._cpi_inflation_rates: Optional[pd.Series] = None

    @lru_cache(maxsize=64)
    def _download_prices(self, ticker: str, start: str, end: str) -> pd.Series:
        """Download price data for a single ticker with caching."""
        try:
            data = yf.download(
                ticker, start=start, end=end, progress=False, auto_adjust=True
            )
            if data.empty:
                return pd.Series(dtype=float)

            s = data["Close"].copy()
            s.name = ticker
            return s
        except Exception as e:
            raise DataError(f"Failed to download data for {ticker}: {str(e)}")

    def _map_to_historical_equivalent(self, ticker: str) -> str:
        """Map modern ETFs to their historical equivalents."""
        return self.config.historical_mappings.get(ticker.upper(), ticker)

    def fetch_returns(
        self, tickers: List[str], start: str, end: str, freq: str
    ) -> pd.DataFrame:
        """
        Fetch return data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            freq: Frequency ('daily' or 'monthly')

        Returns:
            DataFrame of returns with ticker names as columns
        """
        validate_frequency(freq)

        if not tickers:
            raise DataError("No tickers provided")

        series_list = []
        successful_tickers = []

        for ticker in tickers:
            try:
                # Try historical equivalent first
                historical_ticker = self._map_to_historical_equivalent(ticker)
                s = self._download_prices(historical_ticker, start, end)

                if s.empty:
                    # Fallback to original ticker
                    s = self._download_prices(ticker, start, end)

                if not s.empty:
                    s.name = ticker  # Keep original ticker name
                    series_list.append(s)
                    successful_tickers.append(ticker)

            except Exception as e:
                print(f"Warning: Failed to fetch data for {ticker}: {str(e)}")
                continue

        if not series_list:
            raise DataError("No data returned for any ticker")

        # Combine series and compute returns
        prices = pd.concat(series_list, axis=1).dropna(how="any")

        if freq == "daily":
            returns = prices.pct_change().dropna()
        else:
            # Monthly: resample to month-end then compute returns
            monthly_prices = prices.resample("M").last()
            returns = monthly_prices.pct_change().dropna()

        if returns.empty:
            raise DataError("No return data available for the specified period")

        return returns

    def estimate_moments(
        self, returns_df: pd.DataFrame, freq: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate mean returns and covariance matrix.

        Args:
            returns_df: DataFrame of returns
            freq: Frequency for validation

        Returns:
            Tuple of (means, covariance_matrix)
        """
        validate_frequency(freq)

        if returns_df.empty:
            raise DataError("Cannot estimate moments from empty data")

        means = returns_df.mean().to_numpy()
        cov = returns_df.cov().to_numpy()

        return means, cov

    def create_market_data(
        self, returns_df: pd.DataFrame, weights: np.ndarray, freq: str
    ) -> MarketData:
        """Create a MarketData object with validation."""
        means, cov = self.estimate_moments(returns_df, freq)

        return MarketData(
            returns_df=returns_df, means=means, cov=cov, weights=weights, frequency=freq
        )

    def load_cpi_data(self) -> pd.DataFrame:
        """
        Load CPI data from CSV file.

        Returns:
            DataFrame with Year as index and Annual CPI values
        """
        if self._cpi_data is not None:
            return self._cpi_data

        try:
            # Get the project root directory (parent of app/)
            project_root = Path(__file__).parent.parent.parent
            cpi_file = project_root / "data" / "CPI.csv"

            if not cpi_file.exists():
                raise DataError(f"CPI data file not found: {cpi_file}")

            # Read CSV, skipping empty rows
            df = pd.read_csv(cpi_file)
            df = df[df["Year"].notna()]  # Remove rows with missing year

            # Extract Year and Annual columns
            cpi_df = df[["Year", "Annual"]].copy()
            cpi_df = cpi_df[
                cpi_df["Annual"].notna()
            ]  # Remove rows with missing annual data
            cpi_df["Year"] = cpi_df["Year"].astype(int)
            cpi_df["Annual"] = cpi_df["Annual"].astype(float)
            cpi_df = cpi_df.set_index("Year")
            cpi_df.columns = ["CPI"]

            self._cpi_data = cpi_df
            return cpi_df

        except Exception as e:
            raise DataError(f"Failed to load CPI data: {str(e)}")

    def calculate_inflation_rates(self) -> pd.Series:
        """
        Calculate historical annual inflation rates from CPI data.

        Returns:
            Series with Year as index and inflation rate as values
        """
        if self._cpi_inflation_rates is not None:
            return self._cpi_inflation_rates

        cpi_df = self.load_cpi_data()
        cpi_values = cpi_df["CPI"].sort_index()

        # Calculate year-over-year inflation: (CPI_t / CPI_t-1) - 1
        inflation_rates = cpi_values.pct_change().dropna()
        inflation_rates.name = "InflationRate"

        self._cpi_inflation_rates = inflation_rates
        return inflation_rates

    def get_average_inflation_rate(self) -> float:
        """
        Get historical average annual inflation rate for Monte Carlo simulations.

        Returns:
            Average annual inflation rate
        """
        inflation_rates = self.calculate_inflation_rates()
        return float(inflation_rates.mean())

    def get_inflation_rate_for_year(self, year: int) -> Optional[float]:
        """
        Get inflation rate for a specific year.

        Args:
            year: Year to get inflation rate for

        Returns:
            Inflation rate for that year, or None if not available
        """
        inflation_rates = self.calculate_inflation_rates()
        if year in inflation_rates.index:
            return float(inflation_rates.loc[year])
        return None

    def load_wage_data(self) -> pd.DataFrame:
        """
        Load wage data by education level (placeholder for future implementation).

        Returns:
            DataFrame with wage data (to be implemented)
        """
        # TODO: Implement when wage data CSV is available
        raise NotImplementedError("Wage data loading not yet implemented")
