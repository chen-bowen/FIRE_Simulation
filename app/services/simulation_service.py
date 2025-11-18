"""Simulation service for hybrid retirement simulations.

This module contains the hybrid simulation engine:
- Hybrid simulation: Combines historical data for accumulation phase with Monte Carlo projections for retirement phase

Key features:
- Historical data used for accumulation when available
- Monte Carlo projections calibrated from historical data for retirement
- Portfolio rebalancing and inflation adjustments
- Success rate calculations and percentile analysis
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.config import get_config
from app.schemas import (
    PortfolioState,
    SimulationError,
    SimulationParams,
    SimulationResult,
)
from app.utils import per_period_amount, periods_per_year

from .data_service import DataService
from .portfolio_service import PortfolioService


class SimulationService:
    """Service for running retirement simulations.

    This class provides the hybrid simulation method:
    - Hybrid: Combines historical data for accumulation phase with Monte Carlo projections for retirement phase
    """

    def __init__(self):
        # Initialize portfolio service for portfolio math operations
        self.portfolio_service = PortfolioService()
        # Initialize data service for CPI and other data
        self.data_service = DataService()

    # ------------------------- Helpers ------------------------- #
    def _safe_cholesky(self, cov: np.ndarray) -> np.ndarray:
        # add tiny jitter for numerical stability
        jitter = 1e-12
        return np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))

    def _is_crypto_asset(self, ticker: str) -> bool:
        """Check if a ticker is a crypto asset."""
        return self.data_service.is_crypto_ticker(ticker)

    def _cap_crypto_normal_returns(
        self, returns: np.ndarray, tickers: List[str], frequency: str
    ) -> np.ndarray:
        """Cap normal crypto returns per period to prevent unrealistic compounding.

        Args:
            returns: Array of returns (can be 1D or 2D)
            tickers: List of ticker symbols corresponding to returns
            frequency: 'daily' or 'monthly'

        Returns:
            Returns array with crypto returns capped
        """
        if not tickers:
            return returns

        config = get_config()
        max_return = (
            config.crypto_max_daily_return
            if frequency == "daily"
            else config.crypto_max_monthly_return
        )

        # Handle both 1D and 2D arrays
        if returns.ndim == 1:
            # Single period, multiple assets
            n_assets = returns.shape[0]
            for i in range(min(len(tickers), n_assets)):
                if self._is_crypto_asset(tickers[i]):
                    returns[i] = np.clip(returns[i], -max_return, max_return)
        else:
            # Multiple periods, multiple assets
            n_assets = returns.shape[1]
            for i in range(min(len(tickers), n_assets)):
                if self._is_crypto_asset(tickers[i]):
                    returns[:, i] = np.clip(returns[:, i], -max_return, max_return)

        return returns

    def _inject_extreme_volatility_events(
        self,
        returns: np.ndarray,
        tickers: List[str],
        frequency: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Inject rare extreme volatility events for crypto assets.

        Args:
            returns: Array of returns (can be 1D or 2D)
            tickers: List of ticker symbols corresponding to returns
            frequency: 'daily' or 'monthly'
            rng: Random number generator

        Returns:
            Returns array with extreme events injected for crypto assets
        """
        if not tickers:
            return returns

        config = get_config()

        # Handle both 1D and 2D arrays
        if returns.ndim == 1:
            # Single period, multiple assets
            n_assets = returns.shape[0]
            for i in range(min(len(tickers), n_assets)):
                if self._is_crypto_asset(tickers[i]):
                    # Random chance for extreme event
                    if rng.random() < config.crypto_extreme_event_prob:
                        # Most extreme events are crashes, some are rallies
                        if rng.random() < config.crypto_crash_prob_ratio:
                            # Extreme crash: -80% to -70%
                            crash_size = rng.uniform(
                                config.crypto_extreme_crash_min,
                                config.crypto_extreme_crash_max,
                            )
                            returns[i] = crash_size
                        else:
                            # Extreme rally: +150% to +200%
                            rally_size = rng.uniform(
                                config.crypto_extreme_rally_min,
                                config.crypto_extreme_rally_max,
                            )
                            returns[i] = rally_size
        else:
            # Multiple periods, multiple assets
            n_assets = returns.shape[1]
            for i in range(min(len(tickers), n_assets)):
                if self._is_crypto_asset(tickers[i]):
                    # Check each period for extreme events
                    for period_idx in range(returns.shape[0]):
                        if rng.random() < config.crypto_extreme_event_prob:
                            if rng.random() < config.crypto_crash_prob_ratio:
                                crash_size = rng.uniform(
                                    config.crypto_extreme_crash_min,
                                    config.crypto_extreme_crash_max,
                                )
                                returns[period_idx, i] = crash_size
                            else:
                                rally_size = rng.uniform(
                                    config.crypto_extreme_rally_min,
                                    config.crypto_extreme_rally_max,
                                )
                                returns[period_idx, i] = rally_size

        return returns

    def _apply_volatility_dampening(
        self,
        cov: np.ndarray,
        tickers: List[str],
        available_years: float,
        projection_years: float,
    ) -> np.ndarray:
        """Apply volatility dampening for crypto assets when projecting beyond available data.

        Args:
            cov: Covariance matrix
            tickers: List of ticker symbols
            available_years: Years of available historical data
            projection_years: Total years being projected

        Returns:
            Adjusted covariance matrix
        """
        config = get_config()

        # Only apply dampening if projecting beyond available data
        if (
            projection_years <= available_years
            or available_years < config.crypto_min_data_years
        ):
            return cov

        years_beyond = projection_years - available_years
        # Calculate dampening factor (max 50% reduction)
        dampening_factor = min(
            0.5, years_beyond * config.crypto_volatility_dampening / projection_years
        )

        # Create adjusted covariance matrix
        adjusted_cov = cov.copy()

        # Find crypto asset indices
        crypto_indices = [
            i for i, ticker in enumerate(tickers) if self._is_crypto_asset(ticker)
        ]

        # Apply dampening to crypto variance (diagonal) and covariance (off-diagonal)
        for i in crypto_indices:
            # Reduce variance for crypto assets
            adjusted_cov[i, i] *= 1.0 - dampening_factor
            # Also reduce covariance with other assets
            for j in range(len(tickers)):
                if i != j:
                    adjusted_cov[i, j] *= 1.0 - dampening_factor
                    adjusted_cov[j, i] *= 1.0 - dampening_factor

        return adjusted_cov

    # --------------------- Hybrid Simulation --------------------- #
    def run_simulation(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        params: SimulationParams,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """Run combined simulation: historical accumulation + Monte Carlo retirement.

        If there is insufficient historical data for the full accumulation period, the function
        extends accumulation with Monte Carlo returns before switching to the retirement Monte Carlo.
        """
        try:
            ppy = periods_per_year(params.frequency)
            pre_retire_periods = int(params.pre_retire_years * ppy)
            retire_periods = int(params.retire_years * ppy)
            total_periods = pre_retire_periods + retire_periods

            # Basic stats from historical data
            asset_returns = returns_df.to_numpy()
            # Handle single asset case - ensure 2D array
            if asset_returns.ndim == 1:
                asset_returns = asset_returns.reshape(-1, 1)
            available_periods = asset_returns.shape[0]
            n_assets = asset_returns.shape[1]

            # Generate hybrid paths: for each path, randomly pick historical accumulation (if any)
            rng = np.random.default_rng(seed)

            # Get tickers from returns_df
            tickers = list(returns_df.columns)
            available_years = available_periods / ppy
            projection_years = params.pre_retire_years + params.retire_years

            # Prepare Monte Carlo parameters from historical log-returns
            log_returns = np.log1p(asset_returns)
            log_means = np.mean(log_returns, axis=0)

            # Handle single asset case for covariance
            if n_assets == 1:
                cov = np.array([[np.var(log_returns.flatten())]])
            else:
                cov = np.cov(log_returns.T)

            # Apply volatility dampening for crypto if projecting beyond available data
            cov = self._apply_volatility_dampening(
                cov, tickers, available_years, projection_years
            )

            L = self._safe_cholesky(cov)

            use_dynamic_withdrawal = params.withdrawal_params is not None
            initial_category_spending: Optional[Dict[str, float]] = None
            avg_category_inflation_rates: Optional[Dict[str, float]] = None

            if use_dynamic_withdrawal:
                initial_category_spending = (
                    self.portfolio_service.calculate_initial_category_spending(
                        params.withdrawal_params, self.data_service
                    )
                )

                # Load average category-specific inflation rates if CPI adjustment is enabled
                if params.withdrawal_params.use_cpi_adjustment:
                    avg_category_inflation_rates = {}
                    for category_name in initial_category_spending.keys():
                        try:
                            avg_rate = (
                                self.data_service.get_average_category_inflation_rate(
                                    category_name
                                )
                            )
                            if avg_rate is not None:
                                avg_category_inflation_rates[category_name] = avg_rate
                        except Exception:
                            # If category-specific data unavailable, will fall back to general rate
                            pass
                    # If no category rates found, use None to fall back to general rate
                    if not avg_category_inflation_rates:
                        avg_category_inflation_rates = None

            # Determine total annual spending
            total_annual_spend = (
                sum(initial_category_spending.values())
                if use_dynamic_withdrawal
                else params.annual_spend
            )

            # Calculate period amounts (contributions may be wage-based)
            use_wage_based = (
                params.use_wage_based_savings
                and params.education_level
                and params.savings_rate is not None
                and params.current_age is not None
                and params.current_year is not None
            )

            if use_wage_based:
                # Pre-calculate annual wages for each year during accumulation (cache for performance)
                annual_wages_by_year = {}
                for year_offset in range(params.pre_retire_years + 1):
                    target_year = params.current_year + year_offset
                    target_age = params.current_age + year_offset
                    weekly_wage = self.data_service.get_wage_for_age(
                        params.education_level,
                        params.current_age,
                        params.current_year,
                        target_age,
                    )
                    if weekly_wage is None:
                        weekly_wage = self.data_service.get_income_for_education_level(
                            params.education_level
                        )
                        if weekly_wage is None:
                            break
                        growth_rate = self.data_service.calculate_wage_growth_rate(
                            params.education_level
                        )
                        if growth_rate:
                            weekly_wage = weekly_wage * (
                                (1.0 + growth_rate) ** year_offset
                            )
                    annual_wages_by_year[
                        target_year
                    ] = self.data_service.get_annual_wage(weekly_wage)
                contrib_pp = None  # Will be calculated dynamically using cached wages
            else:
                contrib_pp, _ = self.portfolio_service.calculate_period_amounts(
                    params.annual_contrib, 0.0, params.frequency
                )

            spend_pp_nominal_year1 = self.portfolio_service.calculate_period_amounts(
                0.0, total_annual_spend, params.frequency
            )[1]

            avg_inflation_rate = params.inflation_rate_annual
            try:
                avg_inflation_rate = self.data_service.get_average_inflation_rate()
            except Exception:
                avg_inflation_rate = params.inflation_rate_annual

            inflation_pp = self.portfolio_service.calculate_inflation_factor(
                avg_inflation_rate, params.frequency
            )

            balances_over_time: List[np.ndarray] = []
            spending_over_time: List[np.ndarray] = []
            returns_over_time: List[np.ndarray] = []
            terminal_balances: List[float] = []
            success_count = 0
            rebalancing_events: List[str] = []

            for path_idx in range(n_paths):
                state = PortfolioState(balance=params.initial_balance, weights=weights)
                balances = np.zeros(total_periods)
                spending = np.zeros(total_periods)
                returns = np.zeros(total_periods)

                # Determine how many historical accumulation periods are available for this path
                hist_accum_periods = min(pre_retire_periods, available_periods)

                # If any historical accumulation, pick a random contiguous block (bootstrap if needed)
                if hist_accum_periods > 0:
                    max_start = max(1, available_periods - hist_accum_periods + 1)
                    start = int(rng.integers(0, max_start))
                    hist_block = asset_returns[start : start + hist_accum_periods]

                else:
                    hist_block = np.empty((0, n_assets))

                # Generate Monte Carlo returns for the remainder (accum extension + retirement)
                remaining_periods = total_periods - hist_accum_periods
                z = rng.standard_normal(size=(remaining_periods, n_assets))
                mc_log_returns = z @ L.T + log_means
                mc_arith_returns = np.expm1(mc_log_returns)

                # Apply crypto handling to Monte Carlo returns
                mc_arith_returns = self._cap_crypto_normal_returns(
                    mc_arith_returns, tickers, params.frequency
                )
                mc_arith_returns = self._inject_extreme_volatility_events(
                    mc_arith_returns, tickers, params.frequency, rng
                )

                # Build full sequence of portfolio returns
                full_asset_returns = (
                    np.vstack([hist_block, mc_arith_returns])
                    if hist_block.shape[0] > 0
                    else mc_arith_returns
                )

                # Step through accumulation (no spending) and retirement (spending)
                spend_pp = spend_pp_nominal_year1
                for i in range(total_periods):
                    in_accumulation = i < pre_retire_periods
                    if in_accumulation:
                        if use_wage_based:
                            # Use cached annual wage for this year
                            years_into_accumulation = i / ppy
                            year = params.current_year + int(years_into_accumulation)
                            annual_wage = annual_wages_by_year.get(year)
                            if annual_wage is None:
                                # Fallback to most recent available wage
                                annual_wage = (
                                    list(annual_wages_by_year.values())[-1]
                                    if annual_wages_by_year
                                    else 0.0
                                )
                            annual_contrib = annual_wage * params.savings_rate
                            contrib = per_period_amount(
                                annual_contrib, params.frequency
                            )
                        else:
                            contrib = contrib_pp
                        spend = 0.0
                    else:
                        contrib = 0.0
                        # Handle dynamic vs fixed spending
                        if use_dynamic_withdrawal:
                            periods_into_retirement = i - pre_retire_periods
                            years_into_retirement = periods_into_retirement / ppy
                            annual_withdrawal = (
                                self.portfolio_service.calculate_dynamic_withdrawal(
                                    params.withdrawal_params,
                                    initial_category_spending,
                                    years_into_retirement,
                                    avg_inflation_rate,
                                    params.frequency,
                                    avg_category_inflation_rates,
                                )
                            )
                            spend_pp = (
                                self.portfolio_service.calculate_period_withdrawal(
                                    annual_withdrawal,
                                    params.frequency,
                                    params.pacing,
                                    i,
                                )
                            )
                        else:
                            # Fixed spending; apply inflation every period
                            if i == pre_retire_periods:
                                spend_pp = spend_pp_nominal_year1
                            else:
                                spend_pp *= 1.0 + inflation_pp
                        spend = spend_pp

                    # Get per-asset returns for this period
                    if n_assets == 1:
                        # full_asset_returns[i] is shape (1,) when 2D, or scalar when 1D
                        period_return_val = (
                            full_asset_returns[i, 0]
                            if full_asset_returns.ndim == 2
                            else full_asset_returns[i]
                        )
                        period_returns_array = np.array([float(period_return_val)])
                        portfolio_return = float(period_return_val) * float(weights[0])
                    else:
                        period_returns_array = full_asset_returns[i].astype(float)
                        portfolio_return = float(np.dot(period_returns_array, weights))

                    # step portfolio with per-asset returns
                    state, rebal_msg = self.portfolio_service.step_portfolio(
                        state=state,
                        contrib=contrib,
                        spend=spend,
                        inflation_rate_annual=avg_inflation_rate,
                        period_return=period_returns_array,
                        period_index=None,
                        prev_index=None,
                        freq=params.frequency,
                        period_number=i,
                    )
                    if rebal_msg and rebal_msg not in rebalancing_events:
                        rebalancing_events.append(rebal_msg)

                    balances[i] = state.balance
                    spending[i] = spend
                    returns[i] = portfolio_return

                balances_over_time.append(balances)
                spending_over_time.append(spending)
                returns_over_time.append(returns)
                terminal_balances.append(state.balance)
                # Success = balance stayed positive throughout retirement (not just at end)
                # A balance of 0 or negative means the portfolio ran out of money
                retire_start_period = pre_retire_periods
                if retire_start_period < total_periods:
                    retirement_balances = balances[retire_start_period:]
                    path_success = np.all(retirement_balances > 0) and state.balance > 0
                else:
                    path_success = state.balance > 0
                success_count += 1 if path_success else 0

            balances_over_time_arr = np.vstack(balances_over_time)
            spending_over_time_arr = np.vstack(spending_over_time)
            returns_over_time_arr = np.vstack(returns_over_time)
            median_path = np.median(balances_over_time_arr, axis=0)
            p10_path = np.percentile(balances_over_time_arr, 10, axis=0)
            p90_path = np.percentile(balances_over_time_arr, 90, axis=0)
            median_spending = np.median(spending_over_time_arr, axis=0)
            # Use mean returns (not median) - period-by-period median compounds incorrectly
            median_returns = np.mean(returns_over_time_arr, axis=0)
            success_rate = success_count / float(len(balances_over_time))

            # sample up to 100
            rng2 = np.random.default_rng(seed + 1 if seed is not None else None)
            n_samples = min(100, len(balances_over_time))
            sample_indices = rng2.choice(
                len(balances_over_time), n_samples, replace=False
            )
            sample_paths = np.array([balances_over_time[i] for i in sample_indices])

            return SimulationResult(
                success_rate=success_rate,
                terminal_balances=np.array(terminal_balances),
                median_path=median_path,
                p10_path=p10_path,
                p90_path=p90_path,
                periods_per_year=ppy,
                horizon_periods=total_periods,
                requested_periods=total_periods,
                data_limited=(available_periods < pre_retire_periods),
                available_years=available_periods / ppy,
                sample_paths=sample_paths,
                spending_over_time=median_spending,
                returns_over_time=median_returns,
                rebalancing_events=rebalancing_events if rebalancing_events else None,
            )

        except Exception as e:
            raise SimulationError(f"Simulation failed: {str(e)}")
