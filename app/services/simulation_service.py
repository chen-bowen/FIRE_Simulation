"""Simulation service for historical and Monte Carlo simulations.

This module contains the core simulation engines:
- Historical simulation: Rolling window backtests using actual market data
- Monte Carlo simulation: Statistical modeling with calibrated parameters
- Hybrid simulation: Combines both approaches for comprehensive analysis

Key features:
- Rolling window backtests with bootstrap sampling fallback
- Calibrated Monte Carlo with Cholesky decomposition
- Portfolio rebalancing and inflation adjustments
- Success rate calculations and percentile analysis
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.schemas import (
    PortfolioState,
    SimulationError,
    SimulationParams,
    SimulationResult,
)
from app.utils import periods_per_year

from .data_service import DataService
from .portfolio_service import PortfolioService


class SimulationService:
    """Service for running retirement simulations.

    This class provides three main simulation methods:
    1. Historical: Uses rolling windows of actual market data
    2. Monte Carlo: Uses statistical modeling with calibrated parameters
    3. Hybrid: Combines both approaches (historical for accumulation + MC for retirement)
    """

    def __init__(self):
        # Initialize portfolio service for portfolio math operations
        self.portfolio_service = PortfolioService()
        # Initialize data service for CPI and other data
        self.data_service = DataService()

    # ------------------------- Helpers ------------------------- #
    def _calc_periods_and_ppy(self, params: SimulationParams) -> Tuple[int, int]:
        ppy = periods_per_year(params.frequency)
        total_periods = int((params.pre_retire_years + params.retire_years) * ppy)
        return total_periods, ppy

    def _safe_cholesky(self, cov: np.ndarray) -> np.ndarray:
        # add tiny jitter for numerical stability
        jitter = 1e-12
        return np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))

    # --------------------- Historical Simulation --------------------- #
    def run_historical_simulation(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        params: SimulationParams,
        max_starts: int = 1000,
        min_bootstrap_starts: int = 100,
    ) -> SimulationResult:
        """Run historical rolling-window simulation.

        Args:
            returns_df: Historical returns data (rows = periods, cols = assets)
            weights: Portfolio weights (array length == n_assets)
            params: Simulation parameters

        Returns:
            SimulationResult
        """
        try:
            total_periods, ppy = self._calc_periods_and_ppy(params)

            use_dynamic_withdrawal = params.withdrawal_params is not None
            initial_category_spending: Optional[Dict[str, float]] = None
            cpi_inflation_rates: Optional[pd.Series] = None
            category_cpi_data: Dict[str, pd.Series] = {}

            if use_dynamic_withdrawal:
                initial_category_spending = (
                    self.portfolio_service.calculate_initial_category_spending(
                        params.withdrawal_params
                    )
                )
                # Try to load CPI inflation rates; fall back to annual inflation rate
                try:
                    cpi_inflation_rates = self.data_service.calculate_inflation_rates()
                except Exception:
                    cpi_inflation_rates = None

                # Load category-specific CPI inflation rates if CPI adjustment is enabled
                if params.withdrawal_params.use_cpi_adjustment:
                    for category_name in initial_category_spending.keys():
                        try:
                            category_rates = (
                                self.data_service.calculate_category_inflation_rates(
                                    category_name
                                )
                            )
                            if category_rates is not None:
                                category_cpi_data[category_name] = category_rates
                        except Exception:
                            # If category-specific data unavailable, will fall back to general rate
                            pass

            # Determine total annual spending
            total_annual_spend = (
                sum(initial_category_spending.values())
                if use_dynamic_withdrawal
                else params.annual_spend
            )

            contrib_pp, spend_pp_nominal_year1 = (
                self.portfolio_service.calculate_period_amounts(
                    params.annual_contrib, total_annual_spend, params.frequency
                )
            )
            inflation_pp = self.portfolio_service.calculate_inflation_factor(
                params.inflation_rate_annual, params.frequency
            )

            # Convert returns to numpy
            asset_returns = returns_df.to_numpy()
            n_periods_available = asset_returns.shape[0]
            # Handle single asset case - ensure 2D array
            if asset_returns.ndim == 1:
                asset_returns = asset_returns.reshape(-1, 1)
            n_assets = asset_returns.shape[1]

            # Use available data length, adjusting simulation parameters if needed
            available_periods = n_periods_available
            actual_periods = min(total_periods, available_periods)

            if actual_periods < total_periods:
                available_years = available_periods / ppy
                requested_years = total_periods / ppy
                print(
                    f"Warning: Only {available_years:.1f} years of data available, but {requested_years:.1f} years requested. Using {available_years:.1f} years."
                )

            # Rolling window starts
            max_possible_starts = max(0, available_periods - actual_periods + 1)
            starts = (
                min(max_starts, max_possible_starts) if max_possible_starts > 0 else 0
            )

            use_bootstrap = starts < min_bootstrap_starts
            if use_bootstrap:
                # increase starts by bootstrap strategy
                starts = min(500, max_starts)
                print(
                    f"Warning: Limited historical data. Using bootstrap sampling with {starts} starts."
                )

            balances_over_time: List[np.ndarray] = []
            spending_over_time: List[np.ndarray] = []
            returns_over_time: List[np.ndarray] = []
            terminal_balances: List[float] = []
            success_count = 0

            rng = np.random.default_rng()

            for s in range(starts):
                if use_bootstrap:
                    # sample a random start with replacement (weighted toward recent data)
                    weights_idx = (
                        np.arange(1, max_possible_starts + 1)
                        if max_possible_starts > 0
                        else np.array([1])
                    )
                    weights_idx = weights_idx / weights_idx.sum()
                    start = (
                        int(rng.choice(max_possible_starts, p=weights_idx))
                        if max_possible_starts > 0
                        else 0
                    )
                else:
                    # evenly spaced rolling windows
                    step = max(1, (max_possible_starts) // starts) if starts > 0 else 1
                    start = s * step
                    if start + actual_periods > available_periods:
                        start = available_periods - actual_periods

                end = start + actual_periods
                window_returns = asset_returns[start:end]
                dates = (
                    returns_df.index[start:end]
                    if returns_df.index is not None
                    else [None] * actual_periods
                )

                state = PortfolioState(balance=params.initial_balance, weights=weights)
                balances = np.zeros(actual_periods)
                spending = np.zeros(actual_periods)
                returns = np.zeros(actual_periods)
                prev_date = None

                # pre-calc adjusted pre-retire periods in case of shortened data
                adjusted_pre_retire_periods = (
                    int(
                        params.pre_retire_years * ppy * (actual_periods / total_periods)
                    )
                    if total_periods > 0
                    else int(params.pre_retire_years * ppy)
                )

                for i in range(actual_periods):
                    date_i = dates[i] if len(dates) > i else None

                    in_accumulation = i < adjusted_pre_retire_periods

                    # inflation rate for the current year (if CPI available)
                    year_inflation_rate = params.inflation_rate_annual
                    category_inflation_rates: Optional[Dict[str, float]] = None

                    if cpi_inflation_rates is not None and isinstance(
                        date_i, (pd.Timestamp, pd.DatetimeIndex)
                    ):
                        year = date_i.year
                        if year in cpi_inflation_rates.index:
                            year_inflation_rate = float(cpi_inflation_rates.loc[year])

                        # Get category-specific inflation rates for this year if available
                        if (
                            use_dynamic_withdrawal
                            and params.withdrawal_params.use_cpi_adjustment
                        ):
                            category_inflation_rates = {}
                            for (
                                category_name,
                                category_rates,
                            ) in category_cpi_data.items():
                                if year in category_rates.index:
                                    category_inflation_rates[category_name] = float(
                                        category_rates.loc[year]
                                    )
                            # If no category rates found, use None to fall back to general rate
                            if not category_inflation_rates:
                                category_inflation_rates = None

                    # compute contribution and spending for this period
                    if in_accumulation:
                        contrib = contrib_pp
                        spend = 0.0
                    else:
                        # dynamic vs fixed spending
                        if use_dynamic_withdrawal:
                            years_into_retirement = (
                                i - adjusted_pre_retire_periods
                            ) / ppy
                            annual_withdrawal = (
                                self.portfolio_service.calculate_dynamic_withdrawal(
                                    params.withdrawal_params,
                                    initial_category_spending,
                                    years_into_retirement,
                                    year_inflation_rate,
                                    params.frequency,
                                    category_inflation_rates,
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
                            # fixed spending; apply inflation every period
                            if i == adjusted_pre_retire_periods:
                                spend_pp = spend_pp_nominal_year1
                            else:
                                spend_pp *= 1.0 + inflation_pp

                        contrib = 0.0
                        spend = spend_pp

                    # determine portfolio return for this period (portfolio-level)
                    period_asset_returns = window_returns[i]
                    # Handle both single and multiple assets
                    if n_assets == 1:
                        portfolio_return = (
                            float(
                                period_asset_returns[0]
                                if isinstance(period_asset_returns, np.ndarray)
                                else period_asset_returns
                            )
                            * weights[0]
                        )
                    else:
                        portfolio_return = float(np.dot(period_asset_returns, weights))

                    # step portfolio
                    state = self.portfolio_service.step_portfolio(
                        state=state,
                        contrib=contrib,
                        spend=spend,
                        inflation_rate_annual=year_inflation_rate,
                        period_return=portfolio_return,
                        period_index=date_i,
                        prev_index=prev_date,
                        freq=params.frequency,
                    )

                    balances[i] = state.balance
                    spending[i] = spend
                    returns[i] = portfolio_return
                    prev_date = date_i

                balances_over_time.append(balances)
                spending_over_time.append(spending)
                returns_over_time.append(returns)
                terminal_balances.append(state.balance)
                success_count += 1 if state.balance > 0 else 0

            if len(balances_over_time) == 0:
                raise SimulationError(
                    "No historical paths could be generated from the provided data."
                )

            balances_over_time_arr = np.vstack(balances_over_time)
            spending_over_time_arr = np.vstack(spending_over_time)
            returns_over_time_arr = np.vstack(returns_over_time)
            median_path = np.median(balances_over_time_arr, axis=0)
            p10_path = np.percentile(balances_over_time_arr, 10, axis=0)
            p90_path = np.percentile(balances_over_time_arr, 90, axis=0)
            median_spending = np.median(spending_over_time_arr, axis=0)
            median_returns = np.median(returns_over_time_arr, axis=0)

            success_rate = success_count / float(len(balances_over_time))

            # sample paths for visualization
            n_samples = min(100, len(balances_over_time))
            sample_indices = rng.choice(
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
                horizon_periods=actual_periods,
                requested_periods=total_periods,
                data_limited=(available_periods < total_periods),
                available_years=available_periods / ppy,
                sample_paths=sample_paths,
                spending_over_time=median_spending,
                returns_over_time=median_returns,
            )

        except Exception as e:
            raise SimulationError(f"Historical simulation failed: {str(e)}")

    # --------------------- Monte Carlo Simulation --------------------- #
    def run_monte_carlo_simulation(
        self,
        means: np.ndarray,
        cov: np.ndarray,
        weights: np.ndarray,
        params: SimulationParams,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """Run Monte Carlo simulation.

        Args:
            means: Mean returns (per period, arithmetic)
            cov: Covariance matrix (of returns in the same space as means)
            weights: Portfolio weights
            params: Simulation parameters
            n_paths: Number of simulation paths
            seed: RNG seed
        """
        try:
            rng = np.random.default_rng(seed)
            total_periods, ppy = self._calc_periods_and_ppy(params)

            use_dynamic_withdrawal = params.withdrawal_params is not None
            initial_category_spending: Optional[Dict[str, float]] = None
            avg_category_inflation_rates: Optional[Dict[str, float]] = None

            if use_dynamic_withdrawal:
                initial_category_spending = (
                    self.portfolio_service.calculate_initial_category_spending(
                        params.withdrawal_params
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

            contrib_pp, spend_pp_nominal_year1 = (
                self.portfolio_service.calculate_period_amounts(
                    params.annual_contrib, total_annual_spend, params.frequency
                )
            )
            avg_inflation_rate = params.inflation_rate_annual
            try:
                avg_inflation_rate = self.data_service.get_average_inflation_rate()
            except Exception:
                # fall back to provided inflation
                avg_inflation_rate = params.inflation_rate_annual

            inflation_pp = self.portfolio_service.calculate_inflation_factor(
                avg_inflation_rate, params.frequency
            )

            # Cholesky for correlated asset returns
            L = self._safe_cholesky(cov)

            n_assets = means.shape[0]
            terminal_balances = np.zeros(n_paths)
            all_paths = np.zeros((n_paths, total_periods))
            all_spending = np.zeros((n_paths, total_periods))
            all_returns = np.zeros((n_paths, total_periods))

            for p in range(n_paths):
                state = PortfolioState(balance=params.initial_balance, weights=weights)
                spend_pp = spend_pp_nominal_year1

                # Generate correlated normal shocks and convert to arithmetic returns
                z = rng.standard_normal(size=(total_periods, n_assets))
                correlated = z @ L.T
                asset_returns = (
                    correlated + means
                )  # assumes means are per-period additive
                portfolio_returns = asset_returns @ weights

                for i in range(total_periods):
                    in_accumulation = i < int(params.pre_retire_years * ppy)

                    if in_accumulation:
                        contrib = contrib_pp
                        spend = 0.0
                    else:
                        if use_dynamic_withdrawal:
                            periods_into_retirement = i - int(
                                params.pre_retire_years * ppy
                            )
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
                            if i == int(params.pre_retire_years * ppy):
                                spend_pp = spend_pp_nominal_year1
                            else:
                                spend_pp *= 1.0 + inflation_pp

                        contrib = 0.0
                        spend = spend_pp

                    if (
                        params.frequency == "daily"
                        and params.pacing == "monthly-boundary"
                    ):
                        # adjust monthly-boundary frequencies to per-period fractions
                        is_month_boundary = (
                            (i % (ppy // 12)) == 0 if ppy >= 12 else True
                        )
                        contrib = (
                            contrib if in_accumulation and is_month_boundary else 0.0
                        ) * (ppy / 12)
                        spend = (
                            spend
                            if (not in_accumulation and is_month_boundary)
                            else 0.0
                        ) * (ppy / 12)

                    # step portfolio
                    portfolio_return = float(portfolio_returns[i])
                    state = self.portfolio_service.step_portfolio(
                        state=state,
                        contrib=contrib,
                        spend=spend,
                        inflation_rate_annual=avg_inflation_rate,
                        period_return=portfolio_return,
                        period_index=None,
                        prev_index=None,
                        freq=params.frequency,
                    )

                    all_paths[p, i] = state.balance
                    all_spending[p, i] = spend
                    all_returns[p, i] = portfolio_return

                terminal_balances[p] = state.balance

            median_path = np.median(all_paths, axis=0)
            p10_path = np.percentile(all_paths, 10, axis=0)
            p90_path = np.percentile(all_paths, 90, axis=0)
            median_spending = np.median(all_spending, axis=0)
            median_returns = np.median(all_returns, axis=0)
            success_rate = float(np.mean(terminal_balances > 0))

            # sample up to 100 paths
            rng2 = np.random.default_rng(seed + 1 if seed is not None else None)
            n_samples = min(100, n_paths)
            sample_indices = rng2.choice(n_paths, n_samples, replace=False)
            sample_paths = all_paths[sample_indices]

            return SimulationResult(
                success_rate=success_rate,
                terminal_balances=terminal_balances,
                median_path=median_path,
                p10_path=p10_path,
                p90_path=p90_path,
                periods_per_year=ppy,
                horizon_periods=total_periods,
                sample_paths=sample_paths,
                spending_over_time=median_spending,
                returns_over_time=median_returns,
            )

        except Exception as e:
            raise SimulationError(f"Monte Carlo simulation failed: {str(e)}")

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

            # If enough historical data to generate rolling starts, use run_historical_simulation
            if available_periods >= total_periods:
                return self.run_historical_simulation(returns_df, weights, params)

            # Otherwise, generate many hybrid paths: for each path, randomly pick historical accumulation (if any)
            rng = np.random.default_rng(seed)

            # Prepare Monte Carlo parameters from historical log-returns
            log_returns = np.log1p(asset_returns)
            log_means = np.mean(log_returns, axis=0)
            # Handle single asset case for covariance
            if n_assets == 1:
                cov = np.array([[np.var(log_returns.flatten())]])
            else:
                cov = np.cov(log_returns.T)
            L = self._safe_cholesky(cov)

            use_dynamic_withdrawal = params.withdrawal_params is not None
            initial_category_spending: Optional[Dict[str, float]] = None
            avg_category_inflation_rates: Optional[Dict[str, float]] = None

            if use_dynamic_withdrawal:
                initial_category_spending = (
                    self.portfolio_service.calculate_initial_category_spending(
                        params.withdrawal_params
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

            contrib_pp, spend_pp_nominal_year1 = (
                self.portfolio_service.calculate_period_amounts(
                    params.annual_contrib, total_annual_spend, params.frequency
                )
            )

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

                    # Handle single asset case
                    if n_assets == 1:
                        # full_asset_returns[i] is shape (1,) when 2D, or scalar when 1D
                        period_return_val = (
                            full_asset_returns[i, 0]
                            if full_asset_returns.ndim == 2
                            else full_asset_returns[i]
                        )
                        portfolio_return = float(period_return_val) * float(weights[0])
                    else:
                        portfolio_return = float(np.dot(full_asset_returns[i], weights))

                    state = self.portfolio_service.step_portfolio(
                        state=state,
                        contrib=contrib,
                        spend=spend,
                        inflation_rate_annual=avg_inflation_rate,
                        period_return=portfolio_return,
                        period_index=None,
                        prev_index=None,
                        freq=params.frequency,
                    )

                    balances[i] = state.balance
                    spending[i] = spend
                    returns[i] = portfolio_return

                balances_over_time.append(balances)
                spending_over_time.append(spending)
                returns_over_time.append(returns)
                terminal_balances.append(state.balance)
                success_count += 1 if state.balance > 0 else 0

            balances_over_time_arr = np.vstack(balances_over_time)
            spending_over_time_arr = np.vstack(spending_over_time)
            returns_over_time_arr = np.vstack(returns_over_time)
            median_path = np.median(balances_over_time_arr, axis=0)
            p10_path = np.percentile(balances_over_time_arr, 10, axis=0)
            p90_path = np.percentile(balances_over_time_arr, 90, axis=0)
            median_spending = np.median(spending_over_time_arr, axis=0)
            median_returns = np.median(returns_over_time_arr, axis=0)
            success_rate = success_count / float(len(balances_over_time))

            print(f"Simulation: Generated {len(balances_over_time)} paths (hybrid)")

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
            )

        except Exception as e:
            raise SimulationError(f"Simulation failed: {str(e)}")
