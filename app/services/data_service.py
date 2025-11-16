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
from typing import Dict, List, Optional, Tuple

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
        self._category_cpi_data: Dict[str, pd.DataFrame] = {}
        self._category_inflation_rates: Dict[str, pd.Series] = {}
        self._income_data: Dict[str, pd.DataFrame] = {}

        # Map expense category names to CPI file names
        self.category_cpi_mapping = {
            "Food and beverages": "Food_CPI.csv",
            "Housing": "Housing CPI.csv",
            "Apparel": "Apparel_CPI.csv",
            "Transportation": "Transportation_CPI.csv",
            "Medical care": "Medical_CPI.csv",
            "Recreation": "Recreation_CPI.csv",
            "Education and communication": "Education_CPI.csv",
            "Other goods and services": "Other_CPI.csv",
        }

        # Map education level strings to income CSV filenames
        self.education_income_mapping = {
            "Less than high school": "Less_High_school_income.csv",
            "High school": "High_School_income.csv",
            "Some college": "Some_college_income.csv",
            "Bachelor's degree": "Bachelors_Income.csv",
            "Master's degree": "Advanced_degree_income.csv",
            "Professional degree": "Advanced_degree_income.csv",
            "Doctorate": "Advanced_degree_income.csv",
        }

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

    def load_category_cpi_data(self, category_name: str) -> Optional[pd.DataFrame]:
        """
        Load category-specific CPI data from CSV file.

        Args:
            category_name: Name of the expense category

        Returns:
            DataFrame with Year as index and Annual CPI values, or None if file not found
        """
        # Check cache first
        if category_name in self._category_cpi_data:
            return self._category_cpi_data[category_name]

        # Get filename from mapping
        filename = self.category_cpi_mapping.get(category_name)
        if not filename:
            return None

        try:
            # Get the project root directory (parent of app/)
            project_root = Path(__file__).parent.parent.parent
            cpi_file = project_root / "data" / "CPI" / filename

            if not cpi_file.exists():
                # Fallback: try without "CPI" subdirectory
                cpi_file = project_root / "data" / filename
                if not cpi_file.exists():
                    return None

            # Read CSV, skipping empty rows
            df = pd.read_csv(cpi_file)
            df = df[df["Year"].notna()]  # Remove rows with missing year

            # Extract Year and Annual columns (handle different column names)
            if "Annual" in df.columns:
                cpi_df = df[["Year", "Annual"]].copy()
                cpi_df = cpi_df[cpi_df["Annual"].notna()]
                cpi_df["Year"] = cpi_df["Year"].astype(int)
                cpi_df["Annual"] = cpi_df["Annual"].astype(float)
            elif len(df.columns) >= 2:
                # Assume first column is Year, second is CPI value
                year_col = df.columns[0]
                cpi_col = df.columns[1]
                cpi_df = df[[year_col, cpi_col]].copy()
                cpi_df = cpi_df[cpi_df[cpi_col].notna()]
                cpi_df[year_col] = cpi_df[year_col].astype(int)
                cpi_df[cpi_col] = cpi_df[cpi_col].astype(float)
                cpi_df.columns = ["Year", "Annual"]
            else:
                return None

            cpi_df = cpi_df.set_index("Year")
            cpi_df.columns = ["CPI"]

            # Cache the result
            self._category_cpi_data[category_name] = cpi_df
            return cpi_df

        except Exception as e:
            print(
                f"Warning: Failed to load category CPI data for {category_name}: {str(e)}"
            )
            return None

    def calculate_category_inflation_rates(
        self, category_name: str
    ) -> Optional[pd.Series]:
        """
        Calculate historical annual inflation rates for a specific category from CPI data.

        Args:
            category_name: Name of the expense category

        Returns:
            Series with Year as index and inflation rate as values, or None if data unavailable
        """
        # Check cache first
        if category_name in self._category_inflation_rates:
            return self._category_inflation_rates[category_name]

        # Load category CPI data
        cpi_df = self.load_category_cpi_data(category_name)
        if cpi_df is None or cpi_df.empty:
            return None

        cpi_values = cpi_df["CPI"].sort_index()

        # Calculate year-over-year inflation: (CPI_t / CPI_t-1) - 1
        inflation_rates = cpi_values.pct_change().dropna()
        inflation_rates.name = "InflationRate"

        # Cache the result
        self._category_inflation_rates[category_name] = inflation_rates
        return inflation_rates

    def get_category_inflation_rate_for_year(
        self, category_name: str, year: int
    ) -> Optional[float]:
        """
        Get category-specific inflation rate for a specific year.

        Args:
            category_name: Name of the expense category
            year: Year to get inflation rate for

        Returns:
            Inflation rate for that year, or None if not available
        """
        inflation_rates = self.calculate_category_inflation_rates(category_name)
        if inflation_rates is None:
            return None
        if year in inflation_rates.index:
            return float(inflation_rates.loc[year])
        return None

    def get_average_category_inflation_rate(
        self, category_name: str
    ) -> Optional[float]:
        """
        Get historical average annual inflation rate for a specific category.

        Args:
            category_name: Name of the expense category

        Returns:
            Average annual inflation rate, or None if data unavailable
        """
        inflation_rates = self.calculate_category_inflation_rates(category_name)
        if inflation_rates is None or len(inflation_rates) == 0:
            return None
        return float(inflation_rates.mean())

    def load_wage_data(self, education_level: str) -> Optional[pd.DataFrame]:
        """
        Load wage data by education level from CSV file.

        Args:
            education_level: Education level string (e.g., "Bachelor's degree")

        Returns:
            DataFrame with Year as index and Annual income values, or None if file not found
        """
        # Check cache first
        if education_level in self._income_data:
            return self._income_data[education_level]

        # Get filename from mapping
        filename = self.education_income_mapping.get(education_level)
        if not filename:
            return None

        try:
            # Get the project root directory (parent of app/)
            project_root = Path(__file__).parent.parent.parent
            income_file = project_root / "data" / "Income" / filename

            if not income_file.exists():
                return None

            # Read CSV, skipping empty rows
            df = pd.read_csv(income_file)
            df = df[df["Year"].notna()]  # Remove rows with missing year

            # Extract Year and Annual columns
            if "Annual" in df.columns:
                income_df = df[["Year", "Annual"]].copy()
                income_df = income_df[income_df["Annual"].notna()]
                income_df["Year"] = income_df["Year"].astype(int)
                income_df["Annual"] = income_df["Annual"].astype(float)
            else:
                return None

            income_df = income_df.set_index("Year")
            income_df.columns = ["Income"]

            # Cache the result
            self._income_data[education_level] = income_df
            return income_df

        except Exception as e:
            print(
                f"Warning: Failed to load income data for {education_level}: {str(e)}"
            )
            return None

    def get_income_for_education_level(
        self, education_level: str, year: Optional[int] = None
    ) -> Optional[float]:
        """
        Get income for a specific education level and optionally a specific year.

        Args:
            education_level: Education level string
            year: Optional year to get income for. If None, returns most recent available.

        Returns:
            Income value, or None if data unavailable
        """
        income_df = self.load_wage_data(education_level)
        if income_df is None or income_df.empty:
            return None

        if year is not None:
            if year in income_df.index:
                return float(income_df.loc[year, "Income"])
            return None
        else:
            # Return most recent available
            return float(income_df["Income"].iloc[-1])

    def is_crypto_ticker(self, ticker: str) -> bool:
        """
        Check if a ticker is a crypto asset.

        Args:
            ticker: Ticker symbol to check

        Returns:
            True if ticker is a crypto asset, False otherwise
        """
        ticker_upper = ticker.upper()
        # Check for common crypto ticker patterns
        crypto_keywords = ["BTC", "ETH", "CRYPTO"]
        return any(keyword in ticker_upper for keyword in crypto_keywords)
