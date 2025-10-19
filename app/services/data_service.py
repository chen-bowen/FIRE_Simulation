"""Data service for fetching and processing market data."""

import numpy as np
import pandas as pd
import yfinance as yf
from functools import lru_cache
from typing import List, Tuple, Optional
from app.schemas import MarketData, DataError
from app.config import get_config
from app.utils import validate_frequency, periods_per_year


class DataService:
    """Service for fetching and processing market data."""

    def __init__(self):
        self.config = get_config()

    @lru_cache(maxsize=64)
    def _download_prices(self, ticker: str, start: str, end: str) -> pd.Series:
        """Download price data for a single ticker with caching."""
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
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

    def fetch_returns(self, tickers: List[str], start: str, end: str, freq: str) -> pd.DataFrame:
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

    def estimate_moments(self, returns_df: pd.DataFrame, freq: str) -> Tuple[np.ndarray, np.ndarray]:
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

    def create_market_data(self, returns_df: pd.DataFrame, weights: np.ndarray, freq: str) -> MarketData:
        """Create a MarketData object with validation."""
        means, cov = self.estimate_moments(returns_df, freq)

        return MarketData(returns_df=returns_df, means=means, cov=cov, weights=weights, frequency=freq)
