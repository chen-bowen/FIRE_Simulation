"""Simulation service for historical and Monte Carlo simulations."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from app.schemas import SimulationParams, SimulationResult, SimulationError, PortfolioState
from app.utils import periods_per_year
from .portfolio_service import PortfolioService


class SimulationService:
    """Service for running retirement simulations."""

    def __init__(self):
        self.portfolio_service = PortfolioService()

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

            # Calculate per-period amounts
            contrib_pp, spend_pp_nominal_year1 = self.portfolio_service.calculate_period_amounts(
                params.annual_contrib, params.annual_spend, params.frequency
            )
            inflation_pp = self.portfolio_service.calculate_inflation_factor(params.inflation_rate_annual, params.frequency)

            # Build portfolio returns
            asset_returns = returns_df.to_numpy()

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
                print(f"Warning: Limited historical data. Using bootstrap sampling to generate {min(500, max_starts * 5)} scenarios.")
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
                        bootstrap_start = np.random.randint(0, len(asset_returns) - actual_periods + 1)
                    elif start % 3 == 1:
                        # Staggered sampling (every nth point)
                        step = max(1, len(asset_returns) // 50)
                        bootstrap_start = (start * step) % (len(asset_returns) - actual_periods + 1)
                    else:
                        # Weighted sampling (prefer more recent data)
                        weights = np.arange(1, len(asset_returns) - actual_periods + 2)
                        weights = weights / weights.sum()
                        bootstrap_start = np.random.choice(len(asset_returns) - actual_periods + 1, p=weights)

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
                        adjusted_pre_retire_periods = int(params.pre_retire_years * ppy * scale_factor)
                    else:
                        adjusted_pre_retire_periods = params.pre_retire_years * ppy

                    in_accumulation = i < adjusted_pre_retire_periods

                    # Handle different pacing modes
                    if params.frequency == "daily" and params.pacing == "monthly-boundary":
                        is_month_boundary = (i == 0) or (dates[i - 1].month != date_i.month)
                        contrib = (contrib_pp if in_accumulation and is_month_boundary else 0.0) * (ppy / 12)
                        spend = (0.0 if in_accumulation else (spend_pp if is_month_boundary else 0.0)) * (ppy / 12)
                    else:
                        contrib = contrib_pp if in_accumulation else 0.0
                        spend = 0.0 if in_accumulation else spend_pp

                    # Step portfolio
                    state = self.portfolio_service.step_portfolio(
                        state=state,
                        contrib=contrib,
                        spend=spend,
                        inflation_rate_annual=params.inflation_rate_annual,
                        period_return=window_asset_returns[i],  # Individual asset returns for this period
                        period_index=date_i,
                        prev_index=prev_date,
                        freq=params.frequency,
                    )
                    balances[i] = state.balance

                    # Inflate spending
                    if not in_accumulation:
                        if params.frequency == "daily" and params.pacing == "monthly-boundary":
                            is_month_boundary = (i == 0) or (dates[i - 1].month != date_i.month)
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
                sample_indices = np.random.choice(len(balances_over_time), n_samples, replace=False)
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

            # Calculate per-period amounts
            contrib_pp, spend_pp_nominal_year1 = self.portfolio_service.calculate_period_amounts(
                params.annual_contrib, params.annual_spend, params.frequency
            )
            inflation_pp = self.portfolio_service.calculate_inflation_factor(params.inflation_rate_annual, params.frequency)

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
                    print(f"Monte Carlo path {p}: First 5 returns = {portfolio_returns[:5]}")
                    print(f"Random seed used: {seed}")
                    print(f"Monte Carlo simulation type: RANDOM GENERATED")

                for i in range(total_periods):
                    in_accumulation = i < params.pre_retire_years * ppy

                    # Handle different pacing modes
                    if params.frequency == "daily" and params.pacing == "monthly-boundary":
                        is_month_boundary = i % (ppy // 12) == 0
                        contrib = (contrib_pp if in_accumulation and is_month_boundary else 0.0) * (ppy / 12)
                        spend = (0.0 if in_accumulation else (spend_pp if is_month_boundary else 0.0)) * (ppy / 12)
                    else:
                        contrib = contrib_pp if in_accumulation else 0.0
                        spend = 0.0 if in_accumulation else spend_pp

                    # Step portfolio
                    state = PortfolioState(
                        balance=max(0.0, (state.balance + contrib - spend) * (1.0 + portfolio_returns[i])),
                        weights=state.weights,
                    )
                    all_paths[p, i] = state.balance

                    # Inflate spending
                    if not in_accumulation:
                        if params.frequency == "daily" and params.pacing == "monthly-boundary":
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

    def run_hybrid_simulation(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        params: SimulationParams,
        n_paths: int = 1000,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """Run hybrid simulation combining historical and Monte Carlo approaches."""
        try:
            ppy = periods_per_year(params.frequency)
            total_periods = (params.pre_retire_years + params.retire_years) * ppy

            # Calculate per-period amounts
            contrib_pp, spend_pp_nominal_year1 = self.portfolio_service.calculate_period_amounts(
                params.annual_contrib, params.annual_spend, params.frequency
            )
            inflation_pp = self.portfolio_service.calculate_inflation_factor(params.inflation_rate_annual, params.frequency)

            # Use available data length, adjusting simulation parameters if needed
            available_periods = len(returns_df) - 1
            actual_periods = min(total_periods, available_periods)

            # Adjust simulation parameters to fit available data
            if actual_periods < total_periods:
                available_years = available_periods / ppy
                requested_years = total_periods / ppy
                print(
                    f"Warning: Only {available_years:.1f} years of data available, "
                    f"but {requested_years:.1f} years requested. Using {available_years:.1f} years."
                )

            # Get historical asset returns
            asset_returns = returns_df.to_numpy()

            # Calculate historical moments for calibration
            means = np.mean(asset_returns, axis=0)
            cov = np.cov(asset_returns.T)

            # Run hybrid simulation: 50% historical, 50% Monte Carlo
            n_historical = n_paths // 2
            n_mc = n_paths - n_historical

            balances_over_time: List[np.ndarray] = []
            terminal_balances: List[float] = []
            success_count = 0

            # Historical portion - determine sampling strategy
            max_starts = min(n_historical, len(asset_returns) - actual_periods)

            # If we don't have enough data for rolling windows, use bootstrap sampling
            if max_starts < 50:
                print(
                    f"Warning: Limited historical data for hybrid. Using bootstrap sampling to generate {min(200, max_starts * 3)} scenarios."
                )
                max_starts = min(200, max_starts * 3)
                use_bootstrap = True
            else:
                use_bootstrap = False
            for start in range(max_starts):
                if use_bootstrap:
                    # Bootstrap sampling: randomly select starting points with replacement
                    # Use different strategies for better variation
                    if start % 3 == 0:
                        # Random start point
                        bootstrap_start = np.random.randint(0, len(asset_returns) - actual_periods + 1)
                    elif start % 3 == 1:
                        # Staggered sampling (every nth point)
                        step = max(1, len(asset_returns) // 50)
                        bootstrap_start = (start * step) % (len(asset_returns) - actual_periods + 1)
                    else:
                        # Weighted sampling (prefer more recent data)
                        weights = np.arange(1, len(asset_returns) - actual_periods + 2)
                        weights = weights / weights.sum()
                        bootstrap_start = np.random.choice(len(asset_returns) - actual_periods + 1, p=weights)

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

                    # Determine phase
                    if actual_periods < total_periods:
                        scale_factor = actual_periods / total_periods
                        adjusted_pre_retire_periods = int(params.pre_retire_years * ppy * scale_factor)
                    else:
                        adjusted_pre_retire_periods = params.pre_retire_years * ppy

                    in_accumulation = i < adjusted_pre_retire_periods

                    # Handle different pacing modes
                    if params.frequency == "daily" and params.pacing == "monthly-boundary":
                        is_month_boundary = (i == 0) or (dates[i - 1].month != date_i.month)
                        contrib = (contrib_pp if in_accumulation and is_month_boundary else 0.0) * (ppy / 12)
                        spend = (0.0 if in_accumulation else (spend_pp if is_month_boundary else 0.0)) * (ppy / 12)
                    else:
                        contrib = contrib_pp if in_accumulation else 0.0
                        spend = 0.0 if in_accumulation else spend_pp

                    # Step portfolio
                    state = self.portfolio_service.step_portfolio(
                        state=state,
                        contrib=contrib,
                        spend=spend,
                        inflation_rate_annual=params.inflation_rate_annual,
                        period_return=window_asset_returns[i],
                        period_index=date_i,
                        prev_index=prev_date,
                        freq=params.frequency,
                    )
                    balances[i] = state.balance

                    # Inflate spending
                    if not in_accumulation:
                        if params.frequency == "daily" and params.pacing == "monthly-boundary":
                            is_month_boundary = (i == 0) or (dates[i - 1].month != date_i.month)
                            if is_month_boundary:
                                spend_pp *= (1.0 + inflation_pp) ** (ppy / 12)
                        else:
                            spend_pp *= 1.0 + inflation_pp

                    prev_date = date_i

                balances_over_time.append(balances)
                terminal_balances.append(state.balance)
                success_count += 1 if state.balance > 0 else 0

            # Monte Carlo portion
            if seed is not None:
                np.random.seed(seed)

            for _ in range(n_mc):
                # Generate random returns using historical moments
                random_returns = np.random.multivariate_normal(means, cov, actual_periods)

                state = PortfolioState(balance=params.initial_balance, weights=weights)
                balances = np.zeros(actual_periods)
                spend_pp = spend_pp_nominal_year1

                for i in range(actual_periods):
                    # Determine phase
                    if actual_periods < total_periods:
                        scale_factor = actual_periods / total_periods
                        adjusted_pre_retire_periods = int(params.pre_retire_years * ppy * scale_factor)
                    else:
                        adjusted_pre_retire_periods = params.pre_retire_years * ppy

                    in_accumulation = i < adjusted_pre_retire_periods

                    # Handle different pacing modes
                    if params.frequency == "daily" and params.pacing == "monthly-boundary":
                        is_month_boundary = (i % (ppy // 12)) == 0
                        contrib = (contrib_pp if in_accumulation and is_month_boundary else 0.0) * (ppy / 12)
                        spend = (0.0 if in_accumulation else (spend_pp if is_month_boundary else 0.0)) * (ppy / 12)
                    else:
                        contrib = contrib_pp if in_accumulation else 0.0
                        spend = 0.0 if in_accumulation else spend_pp

                    # Step portfolio
                    state = self.portfolio_service.step_portfolio(
                        state=state,
                        contrib=contrib,
                        spend=spend,
                        inflation_rate_annual=params.inflation_rate_annual,
                        period_return=random_returns[i],
                        period_index=None,
                        prev_index=None,
                        freq=params.frequency,
                    )
                    balances[i] = state.balance

                    # Inflate spending
                    if not in_accumulation:
                        if params.frequency == "daily" and params.pacing == "monthly-boundary":
                            is_month_boundary = (i % (ppy // 12)) == 0
                            if is_month_boundary:
                                spend_pp *= (1.0 + inflation_pp) ** (ppy / 12)
                        else:
                            spend_pp *= 1.0 + inflation_pp

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
            print(f"Historical simulation: Generated {len(balances_over_time)} paths")
            print(f"Terminal balance range: ${np.min(terminal_balances):,.0f} to ${np.max(terminal_balances):,.0f}")
            print(f"Success rate: {success_rate:.1%}")
            print(f"Sampling method: {'Bootstrap' if use_bootstrap else 'Rolling Windows'}")

            # Debug: Show first path returns for comparison
            if len(balances_over_time) > 0:
                first_path = balances_over_time[0]
                print(f"Historical first path: First 5 balances = {first_path[:5]}")
                print(f"Historical simulation type: REAL MARKET DATA")

            # Store sample paths for visualization (up to 100 for better variation)
            sample_paths = None
            if len(balances_over_time) > 0:
                n_samples = min(100, len(balances_over_time))
                sample_indices = np.random.choice(len(balances_over_time), n_samples, replace=False)
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
            raise SimulationError(f"Hybrid simulation failed: {str(e)}")
