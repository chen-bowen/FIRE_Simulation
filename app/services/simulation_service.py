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

from typing import Dict, List, Optional

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
    3. Hybrid: Combines both approaches (50% historical + 50% Monte Carlo)

    All methods handle:
    - Portfolio rebalancing and contributions/withdrawals
    - Inflation adjustments
    - Success rate calculations
    - Percentile analysis of outcomes
    """

    def __init__(self):
        # Initialize portfolio service for portfolio math operations
        self.portfolio_service = PortfolioService()
        # Initialize data service for CPI data
        self.data_service = DataService()

    def run_historical_simulation(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        params: SimulationParams,
    ) -> SimulationResult:
        """
        Run historical rolling-window simulation.

        Args:
            returns_df: Historical returns data
            weights: Portfolio weights
            params: Simulation parameters

        Returns:
            Simulation results
        """
        try:
            ppy = periods_per_year(params.frequency)
            total_periods = (params.pre_retire_years + params.retire_years) * ppy

            # Handle dynamic withdrawal vs fixed spending
            use_dynamic_withdrawal = params.withdrawal_params is not None
            initial_category_spending: Optional[Dict[str, float]] = None
            cpi_inflation_rates: Optional[pd.Series] = None
            category_inflation_rates: Optional[Dict[str, float]] = None

            if use_dynamic_withdrawal:
                # Calculate initial category spending
                initial_category_spending = (
                    self.portfolio_service.calculate_initial_category_spending(
                        params.withdrawal_params
                    )
                )
                # Load CPI inflation rates for historical simulation
                if params.withdrawal_params.use_cpi_adjustment:
                    try:
                        cpi_inflation_rates = (
                            self.data_service.calculate_inflation_rates()
                        )
                        # Load category-specific inflation rates
                        category_inflation_rates = {}
                        for category_name in initial_category_spending.keys():
                            category_rate = (
                                self.data_service.get_average_category_inflation_rate(
                                    category_name
                                )
                            )
                            if category_rate is not None:
                                category_inflation_rates[category_name] = category_rate
                            else:
                                # Fallback to overall CPI average if category data unavailable
                                avg_cpi_rate = (
                                    self.data_service.get_average_inflation_rate()
                                )
                                category_inflation_rates[category_name] = avg_cpi_rate
                    except Exception as e:
                        print(
                            f"Warning: Could not load CPI data: {e}. Using default inflation."
                        )
                        cpi_inflation_rates = None
                        category_inflation_rates = None
                # Fallback to fixed spending for calculation
                total_annual_spend = sum(initial_category_spending.values())
            else:
                total_annual_spend = params.annual_spend

            # Calculate per-period amounts
            contrib_pp, spend_pp_nominal_year1 = (
                self.portfolio_service.calculate_period_amounts(
                    params.annual_contrib, total_annual_spend, params.frequency
                )
            )
            inflation_pp = self.portfolio_service.calculate_inflation_factor(
                params.inflation_rate_annual, params.frequency
            )

            # Build portfolio returns
            asset_returns = returns_df.to_numpy()

            # Debug: Check data shapes
            print(f"Debug - returns_df shape: {returns_df.shape}")
            print(f"Debug - returns_df columns: {list(returns_df.columns)}")
            print(f"Debug - asset_returns shape: {asset_returns.shape}")
            print(f"Debug - weights shape: {weights.shape}")
            print(f"Debug - weights: {weights}")

            # Use available data length, adjusting simulation parameters if needed
            available_periods = len(asset_returns) - 1
            actual_periods = min(total_periods, available_periods)

            # Adjust simulation parameters to fit available data
            if actual_periods < total_periods:
                available_years = available_periods / ppy
                requested_years = total_periods / ppy
                print(
                    f"Warning: Only {available_years:.1f} years of data available, "
                    f"but {requested_years:.1f} years requested. Using {available_years:.1f} years."
                )

            # Run rolling windows
            balances_over_time: List[np.ndarray] = []
            terminal_balances: List[float] = []
            success_count = 0

            # Ensure we have enough data for multiple rolling windows
            max_starts = min(1000, len(asset_returns) - actual_periods)

            # If we don't have enough data for rolling windows, use bootstrap sampling
            if max_starts < 100:
                print(
                    f"Warning: Limited historical data. Using bootstrap sampling to generate {min(500, max_starts * 5)} scenarios."
                )
                max_starts = min(500, max_starts * 5)
                use_bootstrap = True
            else:
                use_bootstrap = False

            for start in range(max_starts):
                if use_bootstrap:
                    # Bootstrap sampling: randomly select starting points with replacement
                    # Use different strategies for better variation
                    if start % 3 == 0:
                        # Random start point
                        bootstrap_start = np.random.randint(
                            0, len(asset_returns) - actual_periods + 1
                        )
                    elif start % 3 == 1:
                        # Staggered sampling (every nth point)
                        step = max(1, len(asset_returns) // 50)
                        bootstrap_start = (start * step) % (
                            len(asset_returns) - actual_periods + 1
                        )
                    else:
                        # Weighted sampling (prefer more recent data)
                        sampling_weights = np.arange(
                            1, len(asset_returns) - actual_periods + 2
                        )
                        sampling_weights = sampling_weights / sampling_weights.sum()
                        bootstrap_start = np.random.choice(
                            len(asset_returns) - actual_periods + 1, p=sampling_weights
                        )

                    end = bootstrap_start + actual_periods
                    window_asset_returns = asset_returns[bootstrap_start:end]
                    dates = returns_df.index[bootstrap_start:end]
                else:
                    # Standard rolling window with step size for more variation
                    step = max(1, (len(asset_returns) - actual_periods) // max_starts)
                    adjusted_start = start * step
                    end = adjusted_start + actual_periods
                    window_asset_returns = asset_returns[adjusted_start:end]
                    dates = returns_df.index[adjusted_start:end]

                state = PortfolioState(balance=params.initial_balance, weights=weights)
                balances = np.zeros(actual_periods)
                spend_pp = spend_pp_nominal_year1
                prev_date = None

                for i in range(actual_periods):
                    date_i = dates[i]

                    # Determine phase - scale down proportionally if data is limited
                    if actual_periods < total_periods:
                        scale_factor = actual_periods / total_periods
                        adjusted_pre_retire_periods = int(
                            params.pre_retire_years * ppy * scale_factor
                        )
                    else:
                        adjusted_pre_retire_periods = params.pre_retire_years * ppy

                    in_accumulation = i < adjusted_pre_retire_periods

                    # Calculate spending for this period
                    if not in_accumulation and use_dynamic_withdrawal:
                        # Calculate years into retirement
                        periods_into_retirement = i - adjusted_pre_retire_periods
                        years_into_retirement = periods_into_retirement / ppy

                        # Get inflation rate for this year (use CPI if available)
                        if (
                            cpi_inflation_rates is not None
                            and date_i.year in cpi_inflation_rates.index
                        ):
                            # Use actual historical CPI inflation rate for this year
                            year_inflation_rate = float(
                                cpi_inflation_rates.loc[date_i.year]
                            )
                        else:
                            # Fallback to default inflation rate
                            year_inflation_rate = params.inflation_rate_annual

                        # Get category-specific inflation rates for this year if available
                        year_category_rates: Optional[Dict[str, float]] = None
                        if category_inflation_rates is not None:
                            year_category_rates = {}
                            for category_name in initial_category_spending.keys():
                                category_year_rate = self.data_service.get_category_inflation_rate_for_year(
                                    category_name, date_i.year
                                )
                                if category_year_rate is not None:
                                    year_category_rates[category_name] = (
                                        category_year_rate
                                    )
                                elif category_name in category_inflation_rates:
                                    # Fallback to average category rate
                                    year_category_rates[category_name] = (
                                        category_inflation_rates[category_name]
                                    )
                                else:
                                    # Final fallback to overall rate
                                    year_category_rates[category_name] = (
                                        year_inflation_rate
                                    )

                        # Calculate dynamic withdrawal for this year
                        annual_withdrawal = (
                            self.portfolio_service.calculate_dynamic_withdrawal(
                                params.withdrawal_params,
                                initial_category_spending,
                                years_into_retirement,
                                year_inflation_rate,
                                params.frequency,
                                year_category_rates,
                            )
                        )

                        # Convert to per-period amount
                        spend_pp = self.portfolio_service.calculate_period_withdrawal(
                            annual_withdrawal, params.frequency, params.pacing, i
                        )

                    # Handle different pacing modes
                    if (
                        params.frequency == "daily"
                        and params.pacing == "monthly-boundary"
                    ):
                        is_month_boundary = (i == 0) or (
                            dates[i - 1].month != date_i.month
                        )
                        contrib = (
                            contrib_pp if in_accumulation and is_month_boundary else 0.0
                        ) * (ppy / 12)
                        spend = (
                            0.0
                            if in_accumulation
                            else (spend_pp if is_month_boundary else 0.0)
                        ) * (ppy / 12)
                    else:
                        contrib = contrib_pp if in_accumulation else 0.0
                        spend = 0.0 if in_accumulation else spend_pp

                    # Step portfolio
                    # Debug: Check shapes before calling step_portfolio
                    if i == 0:  # Only debug first iteration
                        print(
                            f"Debug - window_asset_returns[i] shape: {window_asset_returns[i].shape}"
                        )
                        print(f"Debug - weights shape: {weights.shape}")
                        print(
                            f"Debug - window_asset_returns[i]: {window_asset_returns[i]}"
                        )
                        print(f"Debug - weights: {weights}")

                    state = self.portfolio_service.step_portfolio(
                        state=state,
                        contrib=contrib,
                        spend=spend,
                        inflation_rate_annual=params.inflation_rate_annual,
                        period_return=window_asset_returns[
                            i
                        ],  # Individual asset returns for this period
                        period_index=date_i,
                        prev_index=prev_date,
                        freq=params.frequency,
                    )
                    balances[i] = state.balance

                    # Inflate spending (only for fixed spending mode)
                    if not in_accumulation and not use_dynamic_withdrawal:
                        if (
                            params.frequency == "daily"
                            and params.pacing == "monthly-boundary"
                        ):
                            is_month_boundary = (i == 0) or (
                                dates[i - 1].month != date_i.month
                            )
                            if is_month_boundary:
                                spend_pp *= (1.0 + inflation_pp) ** (ppy / 12)
                        else:
                            spend_pp *= 1.0 + inflation_pp

                    prev_date = date_i

                balances_over_time.append(balances)
                terminal_balances.append(state.balance)
                success_count += 1 if state.balance > 0 else 0

            # Calculate results
            balances_over_time_arr = np.vstack(balances_over_time)
            median_path = np.median(balances_over_time_arr, axis=0)
            p10_path = np.percentile(balances_over_time_arr, 10, axis=0)
            p90_path = np.percentile(balances_over_time_arr, 90, axis=0)

            success_rate = success_count / float(max_starts)

            # Store sample paths for visualization (up to 100 for better variation)
            sample_paths = None
            if len(balances_over_time) > 0:
                n_samples = min(100, len(balances_over_time))
                sample_indices = np.random.choice(
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
                data_limited=actual_periods < total_periods,
                available_years=actual_periods / ppy,
                sample_paths=sample_paths,
            )

        except Exception as e:
            raise SimulationError(f"Historical simulation failed: {str(e)}")

    def run_monte_carlo_simulation(
        self,
        means: np.ndarray,
        cov: np.ndarray,
        weights: np.ndarray,
        params: SimulationParams,
        n_paths: int,
        seed: int,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            means: Mean returns
            cov: Covariance matrix
            weights: Portfolio weights
            params: Simulation parameters
            n_paths: Number of simulation paths
            seed: Random seed

        Returns:
            Simulation results
        """
        try:
            rng = np.random.default_rng(seed)
            ppy = periods_per_year(params.frequency)
            total_periods = (params.pre_retire_years + params.retire_years) * ppy

            # Handle dynamic withdrawal vs fixed spending
            use_dynamic_withdrawal = params.withdrawal_params is not None
            initial_category_spending: Optional[Dict[str, float]] = None
            avg_inflation_rate = params.inflation_rate_annual
            category_inflation_rates: Optional[Dict[str, float]] = None

            if use_dynamic_withdrawal:
                # Calculate initial category spending
                initial_category_spending = (
                    self.portfolio_service.calculate_initial_category_spending(
                        params.withdrawal_params
                    )
                )
                # Get average CPI inflation rate for Monte Carlo
                if params.withdrawal_params.use_cpi_adjustment:
                    try:
                        avg_inflation_rate = (
                            self.data_service.get_average_inflation_rate()
                        )
                        # Load category-specific average inflation rates
                        category_inflation_rates = {}
                        for category_name in initial_category_spending.keys():
                            category_rate = (
                                self.data_service.get_average_category_inflation_rate(
                                    category_name
                                )
                            )
                            if category_rate is not None:
                                category_inflation_rates[category_name] = category_rate
                            else:
                                # Fallback to overall CPI average if category data unavailable
                                category_inflation_rates[category_name] = (
                                    avg_inflation_rate
                                )
                    except Exception as e:
                        print(
                            f"Warning: Could not load CPI data: {e}. Using default inflation."
                        )
                        avg_inflation_rate = params.inflation_rate_annual
                        category_inflation_rates = None
                # Fallback to fixed spending for calculation
                total_annual_spend = sum(initial_category_spending.values())
            else:
                total_annual_spend = params.annual_spend

            # Calculate per-period amounts
            contrib_pp, spend_pp_nominal_year1 = (
                self.portfolio_service.calculate_period_amounts(
                    params.annual_contrib, total_annual_spend, params.frequency
                )
            )
            inflation_pp = self.portfolio_service.calculate_inflation_factor(
                avg_inflation_rate, params.frequency
            )

            # Cholesky decomposition for correlation
            L = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))

            terminal_balances = np.zeros(n_paths)
            all_paths = np.zeros((n_paths, total_periods))

            for p in range(n_paths):
                state = PortfolioState(balance=params.initial_balance, weights=weights)
                spend_pp = spend_pp_nominal_year1

                # Generate correlated normal returns
                z = rng.standard_normal(size=(total_periods, means.shape[0]))
                correlated = z @ L.T
                asset_returns = correlated + means
                portfolio_returns = asset_returns @ weights

                # Debug: Print first few returns to verify randomness
                if p == 0:
                    print(
                        f"Monte Carlo path {p}: First 5 returns = {portfolio_returns[:5]}"
                    )
                    print(f"Random seed used: {seed}")
                    print("Monte Carlo simulation type: RANDOM GENERATED")

                for i in range(total_periods):
                    in_accumulation = i < params.pre_retire_years * ppy

                    # Calculate spending for this period
                    if not in_accumulation and use_dynamic_withdrawal:
                        # Calculate years into retirement
                        periods_into_retirement = i - params.pre_retire_years * ppy
                        years_into_retirement = periods_into_retirement / ppy

                        # Calculate dynamic withdrawal using category-specific inflation rates
                        annual_withdrawal = (
                            self.portfolio_service.calculate_dynamic_withdrawal(
                                params.withdrawal_params,
                                initial_category_spending,
                                years_into_retirement,
                                avg_inflation_rate,
                                params.frequency,
                                category_inflation_rates,
                            )
                        )

                        # Convert to per-period amount
                        spend_pp = self.portfolio_service.calculate_period_withdrawal(
                            annual_withdrawal, params.frequency, params.pacing, i
                        )

                    # Handle different pacing modes
                    if (
                        params.frequency == "daily"
                        and params.pacing == "monthly-boundary"
                    ):
                        is_month_boundary = i % (ppy // 12) == 0
                        contrib = (
                            contrib_pp if in_accumulation and is_month_boundary else 0.0
                        ) * (ppy / 12)
                        spend = (
                            0.0
                            if in_accumulation
                            else (spend_pp if is_month_boundary else 0.0)
                        ) * (ppy / 12)
                    else:
                        contrib = contrib_pp if in_accumulation else 0.0
                        spend = 0.0 if in_accumulation else spend_pp

                    # Step portfolio
                    state = PortfolioState(
                        balance=max(
                            0.0,
                            (state.balance + contrib - spend)
                            * (1.0 + portfolio_returns[i]),
                        ),
                        weights=state.weights,
                    )
                    all_paths[p, i] = state.balance

                    # Inflate spending (only for fixed spending mode)
                    if not in_accumulation and not use_dynamic_withdrawal:
                        if (
                            params.frequency == "daily"
                            and params.pacing == "monthly-boundary"
                        ):
                            if i % (ppy // 12) == 0:
                                spend_pp *= (1.0 + inflation_pp) ** (ppy / 12)
                        else:
                            spend_pp *= 1.0 + inflation_pp

                terminal_balances[p] = state.balance

            # Calculate results
            median_path = np.median(all_paths, axis=0)
            p10_path = np.percentile(all_paths, 10, axis=0)
            p90_path = np.percentile(all_paths, 90, axis=0)

            success_rate = float(np.mean(terminal_balances > 0))

            # Store sample paths for visualization (up to 100 for better variation)
            sample_paths = None
            if n_paths > 0:
                n_samples = min(100, n_paths)
                sample_indices = np.random.choice(n_paths, n_samples, replace=False)
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
            )

        except Exception as e:
            raise SimulationError(f"Monte Carlo simulation failed: {str(e)}")

    def run_simulation(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        params: SimulationParams,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run simulation: historical backtest for accumulation, Monte Carlo for retirement.

        This method:
        1. Uses actual historical market data for the accumulation phase (pre-retirement)
        2. Projects forward using Monte Carlo for the retirement phase (post-retirement)
        3. Combines both phases into complete simulation paths
        """
        try:
            ppy = periods_per_year(params.frequency)
            pre_retire_periods = params.pre_retire_years * ppy
            retire_periods = params.retire_years * ppy
            total_periods = pre_retire_periods + retire_periods

            # Handle dynamic withdrawal vs fixed spending
            use_dynamic_withdrawal = params.withdrawal_params is not None
            initial_category_spending: Optional[Dict[str, float]] = None
            avg_inflation_rate = params.inflation_rate_annual
            category_inflation_rates: Optional[Dict[str, float]] = None

            if use_dynamic_withdrawal:
                # Calculate initial category spending
                initial_category_spending = (
                    self.portfolio_service.calculate_initial_category_spending(
                        params.withdrawal_params
                    )
                )
                # Get average CPI inflation rate for Monte Carlo retirement phase
                if params.withdrawal_params.use_cpi_adjustment:
                    try:
                        avg_inflation_rate = (
                            self.data_service.get_average_inflation_rate()
                        )
                        # Load category-specific average inflation rates
                        category_inflation_rates = {}
                        for category_name in initial_category_spending.keys():
                            category_rate = (
                                self.data_service.get_average_category_inflation_rate(
                                    category_name
                                )
                            )
                            if category_rate is not None:
                                category_inflation_rates[category_name] = category_rate
                            else:
                                # Fallback to overall CPI average if category data unavailable
                                category_inflation_rates[category_name] = (
                                    avg_inflation_rate
                                )
                    except Exception as e:
                        print(
                            f"Warning: Could not load CPI data: {e}. Using default inflation."
                        )
                        avg_inflation_rate = params.inflation_rate_annual
                        category_inflation_rates = None
                # Fallback to fixed spending for calculation
                total_annual_spend = sum(initial_category_spending.values())
            else:
                total_annual_spend = params.annual_spend

            # Calculate per-period amounts
            contrib_pp, spend_pp_nominal_year1 = (
                self.portfolio_service.calculate_period_amounts(
                    params.annual_contrib, total_annual_spend, params.frequency
                )
            )
            inflation_pp = self.portfolio_service.calculate_inflation_factor(
                avg_inflation_rate, params.frequency
            )

            # Get historical asset returns
            ticker_list = list(returns_df.columns)
            asset_returns = returns_df.to_numpy()
            available_periods = len(asset_returns) - 1

            # Calculate statistics for all assets (for Monte Carlo when needed)
            # Crypto and non-crypto are treated the same: use historical when available, MC when not
            # IMPORTANT: For Monte Carlo, we need to use geometric mean (log-space) for proper compounding
            # Arithmetic mean overestimates long-term returns due to volatility drag
            # Convert to log returns, calculate mean, then convert back
            log_returns = np.log1p(
                asset_returns
            )  # log(1 + r) ≈ r for small r, but handles large returns correctly
            log_means = np.mean(log_returns, axis=0)
            # Ensure result is always an array (even for single asset)
            if log_means.ndim == 0:
                log_means = np.array([log_means])

            # Calculate geometric mean (this is what we should use for long-term projections)
            # Geometric mean = exp(mean(log(1+r))) - 1, which accounts for volatility drag
            geometric_means = np.expm1(log_means)  # exp(x) - 1, inverse of log1p
            # Ensure result is always an array (even for single asset)
            if geometric_means.ndim == 0:
                geometric_means = np.array([geometric_means])

            # For Monte Carlo, we'll work in log-space to preserve proper distribution
            # The mean in log-space is log_means (already calculated)
            # Covariance should be calculated on log returns for proper correlation structure
            combined_cov = np.cov(log_returns.T)
            combined_cov_full = combined_cov + 1e-12 * np.eye(combined_cov.shape[0])
            L = np.linalg.cholesky(combined_cov_full)

            # Calculate realistic bounds based on actual historical data
            # Use percentiles to avoid extreme outliers from skewing the bounds
            # Ensure result is always an array (even for single asset)
            min_returns = np.percentile(asset_returns, 0.1, axis=0)  # 0.1th percentile
            max_returns = np.percentile(
                asset_returns, 99.9, axis=0
            )  # 99.9th percentile
            # Convert to array if scalar (happens when only 1 asset)
            if min_returns.ndim == 0:
                min_returns = np.array([min_returns])
            if max_returns.ndim == 0:
                max_returns = np.array([max_returns])
            # Ensure bounds are realistic (much tighter than before)
            ppy = periods_per_year(params.frequency)
            if ppy == 252:  # Daily
                # Daily: Realistic extreme bounds (Black Monday was -22%, flash crashes can hit -30%)
                min_bound, max_bound = -0.3, 0.3  # -30% to +30% per day (very extreme)
            else:  # Monthly
                # Monthly: Realistic extreme bounds
                min_bound, max_bound = (
                    -0.5,
                    0.5,
                )  # -50% to +50% per month (very extreme)
            # Use the tighter of historical percentiles or fixed bounds
            min_returns = np.maximum(min_returns, min_bound)
            max_returns = np.minimum(max_returns, max_bound)

            # Diagnostic: Print annualized returns to verify they're reasonable
            annualized_geometric = (1 + geometric_means) ** ppy - 1
            print(
                f"Diagnostic - Annualized geometric mean returns: {annualized_geometric}"
            )
            print(
                f"Diagnostic - Portfolio weighted annualized return: {np.dot(weights, annualized_geometric):.2%}"
            )
            print(
                f"Diagnostic - Return bounds (min, max): {min_returns}, {max_returns}"
            )

            # Check if we have enough historical data for accumulation phase
            # This applies to all assets (crypto and non-crypto treated the same)
            if available_periods < pre_retire_periods:
                print(
                    f"Warning: Only {available_periods / ppy:.1f} years of historical data available, "
                    f"but {params.pre_retire_years} years needed for accumulation. "
                    f"Using available data and extending with Monte Carlo."
                )
                actual_pre_retire_periods = available_periods
            else:
                actual_pre_retire_periods = pre_retire_periods

            # Determine number of historical start points
            if actual_pre_retire_periods > 0:
                max_historical_starts = min(
                    n_paths, available_periods - actual_pre_retire_periods + 1
                )
                if max_historical_starts < 1:
                    max_historical_starts = 1
                    use_bootstrap = True
                else:
                    use_bootstrap = False
            else:
                # No historical data: use Monte Carlo for everything
                max_historical_starts = 1
                use_bootstrap = False

            balances_over_time: List[np.ndarray] = []
            terminal_balances: List[float] = []
            success_count = 0

            # Initialize random number generator for Monte Carlo portion
            if seed is not None:
                rng = np.random.default_rng(seed)
            else:
                rng = np.random.default_rng()

            # Generate simulation paths
            for path_idx in range(n_paths):
                # Select historical start point for accumulation phase
                historical_start = 0
                historical_end = 0
                if actual_pre_retire_periods > 0:
                    if use_bootstrap:
                        # Bootstrap sampling: randomly select starting points
                        if path_idx % 3 == 0:
                            historical_start = rng.integers(
                                0,
                                max(
                                    1, available_periods - actual_pre_retire_periods + 1
                                ),
                            )
                        elif path_idx % 3 == 1:
                            # Staggered sampling
                            step = max(1, available_periods // 50)
                            historical_start = (path_idx * step) % max(
                                1, available_periods - actual_pre_retire_periods + 1
                            )
                        else:
                            # Weighted sampling (prefer more recent data)
                            weights_arr = np.arange(
                                1, available_periods - actual_pre_retire_periods + 2
                            )
                            weights_arr = weights_arr / weights_arr.sum()
                            historical_start = rng.choice(
                                available_periods - actual_pre_retire_periods + 1,
                                p=weights_arr,
                            )
                    else:
                        # Rolling window with step size
                        step = max(
                            1,
                            (available_periods - actual_pre_retire_periods)
                            // max_historical_starts,
                        )
                        historical_start = (path_idx % max_historical_starts) * step

                    historical_end = historical_start + actual_pre_retire_periods

                # Extract historical returns for accumulation phase (includes all assets: crypto and non-crypto)
                historical_returns = None
                historical_dates = None
                if actual_pre_retire_periods > 0:
                    historical_returns = asset_returns[historical_start:historical_end]
                    historical_dates = returns_df.index[historical_start:historical_end]

                # Calculate how many periods need Monte Carlo extension
                periods_needing_mc = total_periods - actual_pre_retire_periods

                # Generate Monte Carlo returns only for periods where data is not available
                # This includes: extension of accumulation phase + entire retirement phase
                # Generate in log-space, then convert back to arithmetic returns
                # IMPORTANT: Bound returns to realistic values (-99% to +500% per period to handle extreme cases)
                mc_extension_returns = None
                if periods_needing_mc > 0:
                    z_extension = rng.standard_normal(
                        size=(periods_needing_mc, len(ticker_list))
                    )
                    correlated_log_returns = z_extension @ L.T
                    # Add log means and convert back to arithmetic space
                    log_returns_mc = correlated_log_returns + log_means
                    mc_extension_returns = np.expm1(
                        log_returns_mc
                    )  # Convert log returns to arithmetic returns
                    # Bound returns to realistic values based on historical data
                    for j in range(len(ticker_list)):
                        mc_extension_returns[:, j] = np.clip(
                            mc_extension_returns[:, j], min_returns[j], max_returns[j]
                        )

                # Initialize portfolio state
                state = PortfolioState(balance=params.initial_balance, weights=weights)
                balances = np.zeros(total_periods)
                spend_pp = spend_pp_nominal_year1
                prev_date = None

                # PHASE 1: Accumulation phase (pre-retirement)
                for i in range(pre_retire_periods):
                    # Use historical data when available, Monte Carlo when not
                    if i < actual_pre_retire_periods and historical_returns is not None:
                        # Use historical data (includes all assets: crypto and non-crypto)
                        period_returns = historical_returns[i]
                    else:
                        # Use Monte Carlo for extension period (when historical data runs out)
                        mc_idx = i - actual_pre_retire_periods
                        if (
                            mc_extension_returns is not None
                            and mc_idx >= 0
                            and mc_idx < len(mc_extension_returns)
                        ):
                            period_returns = mc_extension_returns[mc_idx]
                        else:
                            # Fallback: generate on-the-fly if needed (in log-space)
                            z = rng.standard_normal(size=len(ticker_list))
                            log_returns_period = (z @ L.T) + log_means
                            period_returns = np.expm1(
                                log_returns_period
                            )  # Convert to arithmetic returns
                            # Bound returns to realistic values based on historical data
                            period_returns = np.clip(
                                period_returns, min_returns, max_returns
                            )

                    # Calculate portfolio return
                    portfolio_return = np.dot(weights, period_returns)

                    date_i = (
                        historical_dates[i]
                        if (historical_dates is not None and i < len(historical_dates))
                        else None
                    )

                    # Handle different pacing modes
                    if (
                        params.frequency == "daily"
                        and params.pacing == "monthly-boundary"
                    ):
                        if date_i is not None:
                            is_month_boundary = (i == 0) or (
                                historical_dates[i - 1].month != date_i.month
                            )
                        else:
                            is_month_boundary = (i % (ppy // 12)) == 0
                        contrib = contrib_pp if is_month_boundary else 0.0
                        contrib *= ppy / 12
                    else:
                        contrib = contrib_pp

                    # Step portfolio with combined returns
                    state = self.portfolio_service.step_portfolio(
                        state=state,
                        contrib=contrib,
                        spend=0.0,  # No spending during accumulation
                        inflation_rate_annual=params.inflation_rate_annual,
                        period_return=portfolio_return,  # Use portfolio return
                        period_index=date_i,
                        prev_index=prev_date,
                        freq=params.frequency,
                    )
                    balances[i] = state.balance
                    prev_date = date_i

                # PHASE 2: Monte Carlo retirement phase (post-retirement)
                for i in range(retire_periods):
                    period_idx = pre_retire_periods + i
                    years_into_retirement = i / ppy

                    # Use Monte Carlo for retirement phase (all assets: crypto and non-crypto)
                    # Index into MC extension: retirement starts after accumulation extension
                    mc_idx = (pre_retire_periods - actual_pre_retire_periods) + i
                    if (
                        mc_extension_returns is not None
                        and mc_idx >= 0
                        and mc_idx < len(mc_extension_returns)
                    ):
                        period_returns = mc_extension_returns[mc_idx]
                    else:
                        # Fallback: generate on-the-fly if needed (in log-space)
                        z = rng.standard_normal(size=len(ticker_list))
                        log_returns_period = (z @ L.T) + log_means
                        period_returns = np.expm1(
                            log_returns_period
                        )  # Convert to arithmetic returns
                        # Bound returns to realistic values based on historical data
                        period_returns = np.clip(
                            period_returns, min_returns, max_returns
                        )

                    # Calculate portfolio return
                    portfolio_return = np.dot(weights, period_returns)

                    # Calculate spending for this period
                    if use_dynamic_withdrawal:
                        # Calculate dynamic withdrawal using category-specific inflation rates
                        annual_withdrawal = (
                            self.portfolio_service.calculate_dynamic_withdrawal(
                                params.withdrawal_params,
                                initial_category_spending,
                                years_into_retirement,
                                avg_inflation_rate,
                                params.frequency,
                                category_inflation_rates,
                            )
                        )

                        # Convert to per-period amount
                        spend_pp = self.portfolio_service.calculate_period_withdrawal(
                            annual_withdrawal,
                            params.frequency,
                            params.pacing,
                            period_idx,
                        )
                    else:
                        # Fixed spending with inflation adjustment
                        if i == 0:
                            spend_pp = spend_pp_nominal_year1
                        else:
                            if (
                                params.frequency == "daily"
                                and params.pacing == "monthly-boundary"
                            ):
                                if i % (ppy // 12) == 0:
                                    spend_pp *= (1.0 + inflation_pp) ** (ppy / 12)
                            else:
                                spend_pp *= 1.0 + inflation_pp

                    # Handle different pacing modes
                    if (
                        params.frequency == "daily"
                        and params.pacing == "monthly-boundary"
                    ):
                        is_month_boundary = (i % (ppy // 12)) == 0
                        spend = spend_pp if is_month_boundary else 0.0
                        spend *= ppy / 12
                    else:
                        spend = spend_pp

                    # Step portfolio with combined Monte Carlo returns
                    state = self.portfolio_service.step_portfolio(
                        state=state,
                        contrib=0.0,  # No contributions during retirement
                        spend=spend,
                        inflation_rate_annual=avg_inflation_rate,
                        period_return=portfolio_return,  # Use portfolio return
                        period_index=None,  # No date for Monte Carlo
                        prev_index=None,
                        freq=params.frequency,
                    )
                    balances[period_idx] = state.balance

                balances_over_time.append(balances)
                terminal_balances.append(state.balance)
                success_count += 1 if state.balance > 0 else 0

            # Calculate results
            balances_over_time_arr = np.vstack(balances_over_time)
            median_path = np.median(balances_over_time_arr, axis=0)
            p10_path = np.percentile(balances_over_time_arr, 10, axis=0)
            p90_path = np.percentile(balances_over_time_arr, 90, axis=0)

            success_rate = success_count / float(len(balances_over_time))

            # Debug information
            print(f"Simulation: Generated {len(balances_over_time)} paths")
            print(
                f"Terminal balance range: ${np.min(terminal_balances):,.0f} to ${np.max(terminal_balances):,.0f}"
            )
            print(f"Success rate: {success_rate:.1%}")
            print(
                f"Sampling method: {'Bootstrap' if use_bootstrap else 'Rolling Windows'}"
            )

            # Store sample paths for visualization (up to 100 for better variation)
            sample_paths = None
            if len(balances_over_time) > 0:
                n_samples = min(100, len(balances_over_time))
                sample_indices = rng.choice(
                    len(balances_over_time), size=n_samples, replace=False
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
                data_limited=available_periods < pre_retire_periods,
                available_years=available_periods / ppy,
                sample_paths=sample_paths,
            )

        except Exception as e:
            raise SimulationError(f"Simulation failed: {str(e)}")
