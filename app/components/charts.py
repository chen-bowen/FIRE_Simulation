"""Chart components for visualization.

This module handles all charting and visualization for the retirement planner:
- Portfolio path charts with percentile bands
- Interactive portfolio progress charts showing accumulation and retirement phases
- Comparison charts between different simulation methods
- Interactive Plotly charts with hover information

Key features:
- Adaptive styling for different simulation types
- Phase transition markers (accumulation to retirement)
- Sample path visualization with appropriate density
- Professional financial chart styling
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from app.schemas import SimulationParams, SimulationResult
from app.services.data_service import DataService


class ChartComponent:
    """Charts for the retirement planner.

    This class provides all visualization functionality:
    - Portfolio path charts with percentile bands and sample paths
    - Interactive portfolio progress charts showing accumulation and retirement phases
    - Comparison charts between simulation methods
    - Adaptive styling based on simulation type (Historical vs Monte Carlo)
    """

    def __init__(self):
        self.data_service = DataService()

    def _is_crypto_ticker(self, ticker: str) -> bool:
        """Check if a ticker is a crypto asset."""
        return self.data_service.is_crypto_ticker(ticker)

    def plot_simulation_paths(
        self, result: SimulationResult, title: str, current_age: int = None
    ) -> None:
        """Plot simulation paths with percentile bands and sample paths.

        Args:
            result: Simulation result data
            title: Chart title
            current_age: Current age for age-based x-axis labels (optional)
        """
        time_periods = np.arange(result.horizon_periods) / result.periods_per_year
        # Use age if provided, otherwise use years
        if current_age is not None:
            x = current_age + time_periods
            xaxis_title = "Age"
        else:
            x = time_periods
            xaxis_title = "Years"

        fig = go.Figure()

        # Add shaded confidence band (P10 to P90)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.p90_path,
                name="P90",
                line=dict(color="rgba(158, 202, 225, 0.0)", width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.p10_path,
                name="80% Confidence Band",
                line=dict(color="rgba(158, 202, 225, 0.0)", width=0),
                fill="tonexty",
                fillcolor="rgba(158, 202, 225, 0.2)",
                showlegend=True,
            )
        )

        # Add percentile lines with better styling
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.p90_path,
                name="P90 (90th Percentile)",
                line=dict(color="#3182bd", width=1.5, dash="dash"),
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.median_path,
                name="Median",
                line=dict(color="#08519c", width=3),
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.p10_path,
                name="P10 (10th Percentile)",
                line=dict(color="#3182bd", width=1.5, dash="dash"),
                showlegend=True,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            showlegend=True,
        )

        # Format x-axis to show integer values with adaptive tick interval
        if current_age is not None:
            max_age = current_age + (result.horizon_periods / result.periods_per_year)
            age_range = max_age - current_age
            # Adaptive tick interval: 5 years for long ranges, 2-3 for shorter
            if age_range > 30:
                dtick = 10
            elif age_range > 15:
                dtick = 5
            else:
                dtick = 2
            fig.update_xaxes(tickmode="linear", dtick=dtick)

        st.plotly_chart(fig, use_container_width=True)

    def plot_interactive_portfolio_chart(
        self,
        result: SimulationResult,
        title: str = "Portfolio Quantiles",
        current_age: int = None,
        current_year: int = None,
        initial_balance: float = None,
    ) -> None:
        """Plot interactive portfolio chart with controls for metric type and chart type.

        Args:
            result: Simulation result data
            title: Chart title
            current_age: Current age for age-based x-axis labels (optional)
            current_year: Current year for calendar year x-axis labels (optional)
        """
        from datetime import datetime

        if current_year is None:
            current_year = datetime.now().year

        # Use a container to group controls and chart together
        # This helps prevent scroll jumping when radio buttons change
        with st.container():
            # Interactive controls
            col1, col2 = st.columns(2)

            with col1:
                metric_type = st.radio(
                    "Metric Type",
                    ["Portfolio", "Spending"],
                    key="portfolio_metric_type",
                )

            with col2:
                chart_type = st.radio(
                    "Chart Type",
                    ["Min/Max/Mean", "Spending vs Returns"],
                    key="portfolio_chart_type",
                )

            # Add a small spacer to maintain visual consistency
            st.markdown("<br>", unsafe_allow_html=True)

            # Render appropriate chart based on selections
            if chart_type == "Min/Max/Mean":
                if metric_type == "Portfolio":
                    self._plot_portfolio_quantiles(
                        result, title, current_age, current_year
                    )
                else:  # Spending
                    self._plot_spending_quantiles(
                        result, "Spending Quantiles", current_age, current_year
                    )
            else:  # Spending vs Returns
                self._plot_spending_vs_returns(result, current_year, initial_balance)

            # Add note about today's dollars (styled like examples)
            st.markdown(
                '<div style="background-color: #e8d5ff; padding: 10px; border-radius: 5px; margin-top: 10px;">'
                '<strong>Note:</strong> All values on this page are in "Today\'s Dollars"'
                "</div>",
                unsafe_allow_html=True,
            )

    def _plot_portfolio_quantiles(
        self,
        result: SimulationResult,
        title: str,
        current_age: int = None,
        current_year: int = None,
    ) -> None:
        """Plot portfolio quantiles with shaded bands (Min/Max/Mean view).

        Args:
            result: Simulation result data
            title: Chart title
            current_age: Current age for age-based x-axis labels (optional)
            current_year: Current year for calendar year x-axis labels (optional)
        """
        from datetime import datetime

        if current_year is None:
            current_year = datetime.now().year

        time_periods = np.arange(result.horizon_periods) / result.periods_per_year

        # Determine x-axis: prefer calendar years if available, otherwise age or years
        if current_year is not None:
            x = current_year + time_periods
            xaxis_title = "Year"
        elif current_age is not None:
            x = current_age + time_periods
            xaxis_title = "Age"
        else:
            x = time_periods
            xaxis_title = "Years"

        fig = go.Figure()

        # Calculate additional percentiles for richer visualization
        if (
            result.sample_paths is not None
            and len(result.sample_paths) > 0
            and result.sample_paths.shape[1] == len(result.median_path)
        ):
            # Use sample paths to calculate more percentiles
            p25 = np.percentile(result.sample_paths, 25, axis=0)
            p75 = np.percentile(result.sample_paths, 75, axis=0)
            p5 = np.percentile(result.sample_paths, 5, axis=0)
            p95 = np.percentile(result.sample_paths, 95, axis=0)
        else:
            # Fallback to interpolated percentiles based on P10, median, P90
            p25 = result.p10_path + (result.median_path - result.p10_path) * 0.75
            p75 = result.median_path + (result.p90_path - result.median_path) * 0.75
            p5 = result.p10_path - (result.median_path - result.p10_path) * 0.5
            p95 = result.p90_path + (result.p90_path - result.median_path)

        # Add shaded quantile bands with green color scheme
        # Outer band: P5 to P95
        fig.add_trace(
            go.Scatter(
                x=x,
                y=p95,
                name="P95",
                line=dict(color="rgba(26, 188, 156, 0.0)", width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=p5,
                name="90% Range",
                line=dict(color="rgba(26, 188, 156, 0.0)", width=0),
                fill="tonexty",
                fillcolor="rgba(26, 188, 156, 0.15)",
                showlegend=False,
            )
        )

        # Middle band: P10 to P90
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.p90_path,
                name="P90",
                line=dict(color="rgba(26, 188, 156, 0.0)", width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.p10_path,
                name="80% Range",
                line=dict(color="rgba(26, 188, 156, 0.0)", width=0),
                fill="tonexty",
                fillcolor="rgba(26, 188, 156, 0.25)",
                showlegend=False,
            )
        )

        # Inner band: P25 to P75
        fig.add_trace(
            go.Scatter(
                x=x,
                y=p75,
                name="P75",
                line=dict(color="rgba(26, 188, 156, 0.0)", width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=p25,
                name="50% Range",
                line=dict(color="rgba(26, 188, 156, 0.0)", width=0),
                fill="tonexty",
                fillcolor="rgba(26, 188, 156, 0.35)",
                showlegend=False,
            )
        )

        # Add median line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.median_path,
                name="Median",
                line=dict(color="#1abc9c", width=2.5),
                showlegend=False,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            showlegend=False,
        )

        # Format x-axis
        if current_year is not None:
            max_year = current_year + (result.horizon_periods / result.periods_per_year)
            year_range = max_year - current_year
            if year_range > 30:
                dtick = 10
            elif year_range > 15:
                dtick = 5
            else:
                dtick = 2
            fig.update_xaxes(tickmode="linear", dtick=dtick)
        elif current_age is not None:
            max_age = current_age + (result.horizon_periods / result.periods_per_year)
            age_range = max_age - current_age
            if age_range > 30:
                dtick = 10
            elif age_range > 15:
                dtick = 5
            else:
                dtick = 2
            fig.update_xaxes(tickmode="linear", dtick=dtick)

        st.plotly_chart(fig, use_container_width=True)

    def _plot_spending_quantiles(
        self,
        result: SimulationResult,
        title: str,
        current_age: int = None,
        current_year: int = None,
    ) -> None:
        """Plot spending quantiles (placeholder - requires spending data).

        Args:
            result: Simulation result data
            title: Chart title
            current_age: Current age for age-based x-axis labels (optional)
            current_year: Current year for calendar year x-axis labels (optional)
        """
        if result.spending_over_time is None:
            st.warning("Spending data not available for this visualization.")
            return

        from datetime import datetime

        if current_year is None:
            current_year = datetime.now().year

        time_periods = np.arange(result.horizon_periods) / result.periods_per_year

        if current_year is not None:
            x = current_year + time_periods
            xaxis_title = "Year"
        elif current_age is not None:
            x = current_age + time_periods
            xaxis_title = "Age"
        else:
            x = time_periods
            xaxis_title = "Years"

        fig = go.Figure()

        # Convert period spending to annual spending
        ppy = result.periods_per_year
        annual_spending = result.spending_over_time * ppy

        fig.add_trace(
            go.Scatter(
                x=x,
                y=annual_spending,
                name="Median Spending",
                line=dict(color="#1abc9c", width=2.5),
                showlegend=True,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title="Annual Spending ($)",
            hovermode="x unified",
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)

    def _plot_spending_vs_returns(
        self,
        result: SimulationResult,
        current_year: int = None,
        initial_balance: float = None,
    ) -> None:
        """Plot spending vs returns bar chart.

        Args:
            result: Simulation result data
            current_year: Current year for calendar year x-axis labels
        """
        from datetime import datetime

        if current_year is None:
            current_year = datetime.now().year

        if result.spending_over_time is None or result.returns_over_time is None:
            st.warning(
                "Spending and returns data not available for this visualization."
            )
            return

        ppy = result.periods_per_year
        total_years = max(1, int(math.ceil(result.horizon_periods / ppy)))

        # Aggregate periods into years
        years = []
        annual_spending = []
        annual_returns = []

        portfolio_depleted = False

        for year_idx in range(total_years):
            start_period = year_idx * ppy
            end_period = min((year_idx + 1) * ppy, result.horizon_periods)

            # Sum spending for the year (spending_over_time is per-period)
            year_spending = np.sum(result.spending_over_time[start_period:end_period])

            # Convert percentage returns into dollar growth on the median portfolio path
            year_return_dollars = 0.0

            if not portfolio_depleted:
                year_returns = result.returns_over_time[start_period:end_period]
                year_balances = result.median_path[start_period:end_period]

                for period_idx, period_return in enumerate(year_returns):
                    if np.isnan(period_return):
                        continue
                    if period_idx >= len(year_balances):
                        continue

                    balance_after_return = year_balances[period_idx]
                    if np.isnan(balance_after_return):
                        continue

                    if balance_after_return <= 0:
                        # Once the portfolio is depleted, stop counting returns for this and future years
                        portfolio_depleted = True
                        year_return_dollars = 0.0
                        break

                    denom = 1.0 + period_return
                    if np.isclose(denom, 0.0):
                        # Skip pathological cases where return is effectively -100%
                        # to avoid division by zero; values remain uncapped otherwise.
                        continue

                    # Balance before returns is after contributions/withdrawals but before growth
                    balance_before_return = balance_after_return / denom
                    if balance_before_return <= 0:
                        portfolio_depleted = True
                        year_return_dollars = 0.0
                        break

                    period_return_dollars = balance_before_return * period_return
                    year_return_dollars += period_return_dollars

            years.append(current_year + year_idx)
            annual_spending.append(-year_spending)  # Negative for spending
            annual_returns.append(year_return_dollars)  # Positive for returns

        fig = go.Figure()

        # Add spending bars (negative, pink)
        fig.add_trace(
            go.Bar(
                x=years,
                y=annual_spending,
                name="Spending",
                marker_color="#e74c3c",  # Pink/red for negative
                hovertemplate="%{x}<br>Spending: $%{y:,.0f}<extra></extra>",
            )
        )

        # Add returns bars (positive, teal)
        fig.add_trace(
            go.Bar(
                x=years,
                y=annual_returns,
                name="Returns",
                marker_color="#1abc9c",  # Teal for positive
                hovertemplate="%{x}<br>Returns: $%{y:,.0f}<extra></extra>",
            )
        )

        # Calculate y-axis range
        all_values = annual_spending + annual_returns
        y_min = min(all_values) * 1.1
        y_max = max(all_values) * 1.1

        fig.update_layout(
            title="Average Spending vs Returns",
            xaxis_title="Year",
            yaxis_title="Amount ($)",
            barmode="group",
            hovermode="x unified",
            showlegend=True,
            yaxis=dict(range=[y_min, y_max]),
            xaxis=dict(tickmode="linear", dtick=2),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Returns reflect investment growth on the median portfolio after each year's "
            "contributions and withdrawals."
        )

    def plot_terminal_wealth_histogram(
        self,
        result: SimulationResult,
        pre_retire_years: int = None,
        current_age: int = None,
        portfolio_weights: dict = None,
        user_name: str = None,
        returns_df: pd.DataFrame = None,
        initial_balance: float = None,
        asset_class_mapping: dict = None,
    ) -> None:
        """Plot interactive year-by-year portfolio allocation explorer.

        This replaces the terminal wealth histogram with an interactive slider-based visualization
        that shows portfolio value, allocation breakdown, and phase for each year.

        Args:
            result: Simulation result data
            pre_retire_years: Number of years in accumulation phase
            current_age: Current age for age-based labels (optional)
            portfolio_weights: Dictionary mapping asset class names to percentages
            user_name: User name for display (optional)
        """
        # Get portfolio weights from session state if not provided
        if portfolio_weights is None:
            if "portfolio_weights" in st.session_state:
                portfolio_weights = {
                    k: v
                    for k, v in st.session_state["portfolio_weights"].items()
                    if v > 0
                }
            else:
                st.warning(
                    "Portfolio weights not available. Cannot display allocation breakdown."
                )
                return

        if not portfolio_weights:
            st.warning("No portfolio allocation configured.")
            return

        # Calculate total years
        total_years = result.horizon_periods / result.periods_per_year

        # Get year indices (convert years to period indices)
        # Each year corresponds to periods_per_year periods
        ppy = result.periods_per_year

        # Create slider for year selection
        max_year = int(total_years)
        from datetime import datetime

        year_label = "Select Simulation Year"
        if current_age is not None:
            current_year = datetime.now().year
            year_label += f" ({current_year})"
            if user_name:
                year_label += f" - {user_name}"
            age_label = f" (Age: {current_age})"
        else:
            age_label = ""

        # Use a stable key to maintain slider state across reruns
        slider_key = "portfolio_year_slider"

        selected_year = st.slider(
            year_label + age_label,
            min_value=1,
            max_value=max_year,
            value=st.session_state.get(slider_key, 1),
            step=1,
            key=slider_key,
        )

        # Calculate period index for selected year
        # Year 1 = periods 0 to ppy-1, Year 2 = periods ppy to 2*ppy-1, etc.
        period_idx = min((selected_year - 1) * ppy, len(result.median_path) - 1)

        # Get portfolio values for selected year
        portfolio_value_p90 = result.p90_path[period_idx]
        portfolio_value_p10 = result.p10_path[period_idx]
        portfolio_value_median = result.median_path[period_idx]

        # Use median as primary value
        portfolio_value = portfolio_value_median

        # Determine phase
        if pre_retire_years is not None:
            is_accumulation = selected_year <= pre_retire_years
            phase = "Accumulation Phase" if is_accumulation else "Retirement Phase"
            phase_color = "green" if is_accumulation else "red"
        else:
            phase = "Unknown"
            phase_color = "gray"

        # Calculate dollar amounts per asset class accounting for annual rebalancing
        asset_classes = list(portfolio_weights.keys())
        weights_array = np.array([portfolio_weights[ac] for ac in asset_classes])
        # Normalize weights to sum to 100%
        weights_array = weights_array / weights_array.sum() * 100.0

        dollar_amounts = {}
        actual_percentages = {}

        if (
            returns_df is not None
            and initial_balance is not None
            and asset_class_mapping is not None
        ):
            # Find the last rebalancing point
            # Rebalancing happens at the END of each year, so:
            # - For year N, rebalancing happens at period (N * ppy - 1) or start of year N+1
            # - We want to show allocation at the selected year, accounting for drift since last rebalance
            ppy = result.periods_per_year

            # Calculate which year period_idx falls into
            # period_idx = (selected_year - 1) * ppy is at the START of selected_year
            # Rebalancing happens at the END of each year, so last rebalance was at end of (selected_year - 1)
            # For year 1, there's no previous rebalance, so use initial allocation
            if selected_year == 1:
                # Year 1: use initial balance and calculate from start
                rebalance_period = 0
                rebalanced_portfolio_value = initial_balance
            else:
                # Last rebalancing was at the end of previous year (start of current year)
                rebalance_period = (selected_year - 1) * ppy
                if rebalance_period > 0 and rebalance_period < len(result.median_path):
                    rebalanced_portfolio_value = result.median_path[rebalance_period]
                else:
                    # Fallback: estimate from initial balance
                    rebalanced_portfolio_value = initial_balance

            # Calculate how many periods since last rebalancing
            periods_since_rebalance = period_idx - rebalance_period + 1
            periods_since_rebalance = max(
                1, min(periods_since_rebalance, ppy)
            )  # Clamp to current year

            # Calculate dollar amounts at last rebalancing (target allocation)
            rebalanced_amounts = {}
            for ac, weight_pct in zip(asset_classes, weights_array):
                rebalanced_amounts[ac] = rebalanced_portfolio_value * (
                    weight_pct / 100.0
                )

            # Calculate returns since last rebalancing for each asset class
            asset_returns_since_rebalance = {}
            available_periods = len(returns_df) if returns_df is not None else 0

            for ac in asset_classes:
                if ac in asset_class_mapping:
                    # Get tickers for this asset class
                    tickers = asset_class_mapping[ac]
                    # Find matching tickers in returns_df
                    matching_tickers = (
                        [t for t in tickers if t in returns_df.columns]
                        if returns_df is not None
                        else []
                    )
                    if matching_tickers:
                        # Use first matching ticker
                        ticker = matching_tickers[0]
                        if ticker in returns_df.columns:
                            # Calculate returns from last rebalancing point to selected period
                            start_period = rebalance_period
                            end_period = min(period_idx + 1, available_periods)

                            if (
                                end_period > start_period
                                and start_period < available_periods
                            ):
                                # Get returns for the current year (since last rebalancing)
                                year_returns = (
                                    returns_df[ticker]
                                    .iloc[start_period:end_period]
                                    .values
                                )
                                if len(year_returns) > 0:
                                    # Calculate cumulative return for this year
                                    cumulative_return = np.prod(1 + year_returns) - 1

                                    # If we need to extend beyond available data, use average
                                    if end_period < period_idx + 1:
                                        avg_return_per_period = (
                                            np.mean(year_returns)
                                            if len(year_returns) > 0
                                            else 0.0
                                        )
                                        additional_periods = (
                                            period_idx + 1
                                        ) - end_period
                                        cumulative_return = (1 + cumulative_return) * (
                                            (1 + avg_return_per_period)
                                            ** additional_periods
                                        ) - 1

                                    asset_returns_since_rebalance[ac] = (
                                        cumulative_return
                                    )
                                else:
                                    asset_returns_since_rebalance[ac] = 0.0
                            else:
                                asset_returns_since_rebalance[ac] = 0.0
                        else:
                            asset_returns_since_rebalance[ac] = 0.0
                    else:
                        asset_returns_since_rebalance[ac] = 0.0
                else:
                    asset_returns_since_rebalance[ac] = 0.0

            # Calculate actual dollar amounts: start from rebalanced amounts, apply returns since rebalancing
            # Crypto gets the residual (total - sum of others) to avoid unrealistic deflation of other assets
            non_crypto_value = 0.0
            crypto_asset_class = None

            for ac in asset_classes:
                # Check if this is crypto
                is_crypto = False
                if ac == "Crypto":
                    is_crypto = True
                    crypto_asset_class = ac
                elif ac in asset_class_mapping:
                    tickers = asset_class_mapping[ac]
                    matching_tickers = (
                        [t for t in tickers if t in returns_df.columns]
                        if returns_df is not None
                        else []
                    )
                    if matching_tickers:
                        ticker = matching_tickers[0]
                        if self._is_crypto_ticker(ticker):
                            is_crypto = True
                            crypto_asset_class = ac

                if not is_crypto:
                    # Start from rebalanced amount, apply returns since rebalancing
                    rebalanced_amt = rebalanced_amounts.get(ac, 0.0)
                    year_return = asset_returns_since_rebalance.get(ac, 0.0)
                    dollar_amounts[ac] = rebalanced_amt * (1 + year_return)
                    non_crypto_value += dollar_amounts[ac]
                else:
                    # Will calculate crypto as residual
                    dollar_amounts[ac] = 0.0

            # Calculate crypto as residual: total portfolio value - sum of non-crypto assets
            if crypto_asset_class:
                crypto_value = max(0.0, portfolio_value - non_crypto_value)
                dollar_amounts[crypto_asset_class] = crypto_value

            # Normalize dollar amounts to sum to portfolio value (accounts for contributions/withdrawals)
            total_drifted_value = sum(dollar_amounts.values())
            if total_drifted_value > 0 and portfolio_value > 0:
                # Scale all dollar amounts proportionally to match actual portfolio value
                scale_factor = portfolio_value / total_drifted_value
                for ac in asset_classes:
                    dollar_amounts[ac] = dollar_amounts[ac] * scale_factor

            # Calculate actual percentages
            if portfolio_value > 0:
                for ac in asset_classes:
                    actual_percentages[ac] = (
                        dollar_amounts[ac] / portfolio_value
                    ) * 100.0
            else:
                # Fallback to original weights if calculation fails
                for ac, weight_pct in zip(asset_classes, weights_array):
                    actual_percentages[ac] = weight_pct
                    dollar_amounts[ac] = portfolio_value * (weight_pct / 100.0)
        else:
            # Fallback: use fixed allocation if returns data not available
            for ac, weight_pct in zip(asset_classes, weights_array):
                dollar_amounts[ac] = portfolio_value * (weight_pct / 100.0)
                actual_percentages[ac] = weight_pct

        # Display title with year info
        if current_age is not None:
            current_year = datetime.now().year
            display_year = current_year + selected_year - 1
            display_age = current_age + selected_year - 1
            st.subheader(f"Allocation for Year {display_year} (Age {display_age})")
        else:
            st.subheader(f"Allocation for Year {selected_year}")

        # Create two columns: pie chart and info
        col1, col2 = st.columns([2, 1])

        with col1:
            # Create pie chart with dollar amounts
            labels = list(dollar_amounts.keys())
            values = list(dollar_amounts.values())
            # Use actual percentages if calculated, otherwise use original weights
            if actual_percentages:
                percentages = [f"{actual_percentages[ac]:.1f}%" for ac in labels]
            else:
                percentages = [f"{w:.1f}%" for w in weights_array]

            # Use color scheme from sidebar if available
            color_map = {
                "US Stocks": "#2ecc71",
                "International Stocks": "#3498db",
                "Bonds": "#9b59b6",
                "Cash": "#e74c3c",
                "Crypto": "#f39c12",
                "Real Estate": "#1abc9c",
                "Commodities": "#e67e22",
            }
            colors = [color_map.get(label, "#95a5a6") for label in labels]

            # Create hover text with both dollar amount and percentage
            hover_text = [
                f"{label}<br>${amt:,.0f}<br>{pct}"
                for label, amt, pct in zip(labels, values, percentages)
            ]

            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.4,  # Donut chart
                        textinfo="label+percent",
                        textposition="auto",  # Auto-position labels inside/outside based on space
                        hovertemplate="%{hovertext}<extra></extra>",
                        hovertext=hover_text,
                        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                    )
                ]
            )

            # Add total value in center
            fig.add_annotation(
                text=f"<b>${portfolio_value:,.0f}</b><br>Total Value<br>(Year {selected_year})",
                x=0.5,
                y=0.5,
                font_size=16,
                showarrow=False,
            )

            fig.update_layout(
                title="",
                showlegend=False,
                height=450,
                margin=dict(l=60, r=60, t=30, b=50),
                legend=dict(visible=False),
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Display portfolio info
            st.markdown("### Portfolio Details")
            st.metric("Total Value", f"${portfolio_value:,.0f}")

            # Show P10-P90 range
            st.caption(
                f"Range: ${portfolio_value_p10:,.0f} (P10) - ${portfolio_value_p90:,.0f} (P90)"
            )

            st.markdown("#### Allocation")
            for ac, amt in dollar_amounts.items():
                # Use actual percentage if calculated, otherwise use original weight
                if actual_percentages and ac in actual_percentages:
                    pct = actual_percentages[ac]
                else:
                    pct = portfolio_weights[ac]
                st.write(f"**{ac}**: ${amt:,.0f} ({pct:.1f}%)")

            st.markdown("#### Phase")
            st.markdown(
                f'<span style="color: {phase_color}; font-weight: bold;">{phase}</span>',
                unsafe_allow_html=True,
            )

        # Show percentage below chart
        st.caption(f"Total Allocation: {sum(weights_array):.1f}%")

    def plot_comparison_chart(
        self,
        historical_result: SimulationResult,
        mc_result: SimulationResult,
        current_age: int = None,
    ) -> None:
        """Plot comparison between historical and Monte Carlo results.

        Args:
            historical_result: Historical simulation result
            mc_result: Monte Carlo simulation result
            current_age: Current age for age-based x-axis labels (optional)
        """
        years_hist = (
            np.arange(historical_result.horizon_periods)
            / historical_result.periods_per_year
        )
        years_mc = np.arange(mc_result.horizon_periods) / mc_result.periods_per_year

        # Use age if provided, otherwise use years
        if current_age is not None:
            x_hist = current_age + years_hist
            x_mc = current_age + years_mc
            xaxis_title = "Age"
        else:
            x_hist = years_hist
            x_mc = years_mc
            xaxis_title = "Years"

        fig = go.Figure()

        # Historical paths
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=historical_result.median_path,
                name="Historical Median",
                line=dict(color="#3182bd", dash="solid"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=historical_result.p90_path,
                name="Historical P90",
                line=dict(color="#9ecae1", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=historical_result.p10_path,
                name="Historical P10",
                line=dict(color="#9ecae1", dash="dash"),
            )
        )

        # Monte Carlo paths
        fig.add_trace(
            go.Scatter(
                x=x_mc,
                y=mc_result.median_path,
                name="Monte Carlo Median",
                line=dict(color="#31a354", dash="solid"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_mc,
                y=mc_result.p90_path,
                name="Monte Carlo P90",
                line=dict(color="#c7e9c0", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_mc,
                y=mc_result.p10_path,
                name="Monte Carlo P10",
                line=dict(color="#c7e9c0", dash="dash"),
            )
        )

        fig.update_layout(
            title="Historical vs Monte Carlo Comparison",
            xaxis_title=xaxis_title,
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            showlegend=True,
        )

        # Format x-axis to show integer values with adaptive tick interval
        if current_age is not None:
            max_age_hist = current_age + (
                historical_result.horizon_periods / historical_result.periods_per_year
            )
            max_age_mc = current_age + (
                mc_result.horizon_periods / mc_result.periods_per_year
            )
            max_age = max(max_age_hist, max_age_mc)
            age_range = max_age - current_age
            # Adaptive tick interval: 5 years for long ranges, 2-3 for shorter
            if age_range > 30:
                dtick = 10
            elif age_range > 15:
                dtick = 5
            else:
                dtick = 2
            fig.update_xaxes(tickmode="linear", dtick=dtick)

        st.plotly_chart(fig, use_container_width=True)

    def plot_savings_and_returns_breakdown(
        self,
        params: SimulationParams,
        result: SimulationResult,
        current_age: int,
        current_year: int,
    ) -> None:
        """
        Plot savings contributions vs portfolio growth, showing investment returns.

        Args:
            params: Simulation parameters (must have wage-based savings enabled)
            result: Simulation result data
            current_age: Current age
            current_year: Current year
        """
        if not params.use_wage_based_savings or not params.education_level:
            return

        ppy = result.periods_per_year
        pre_retire_periods = int(params.pre_retire_years * ppy)

        # Calculate data for accumulation phase only
        years = []
        annual_contributions = []
        cumulative_contributions = []
        portfolio_balances = []
        cumulative_investment_returns = []  # Cumulative investment returns
        annual_investment_returns = []  # Annual investment return amount

        cumulative_contrib = params.initial_balance  # Start with initial balance
        prev_portfolio_balance = params.initial_balance

        # Add initial data point
        years.append(current_year)
        annual_contributions.append(0.0)  # No contribution yet
        cumulative_contributions.append(params.initial_balance)
        portfolio_balances.append(params.initial_balance)
        cumulative_investment_returns.append(0.0)
        annual_investment_returns.append(0.0)

        for i in range(min(pre_retire_periods, len(result.median_path))):
            years_into_accumulation = i / ppy
            year = current_year + int(years_into_accumulation)

            # Only add data points once per year (at the end of each year)
            if i % ppy == 0 and i > 0:  # End of a year (skip i=0 as it's already added)
                # Get wage for this age (use the year we're ending)
                age = current_age + int(years_into_accumulation)
                weekly_wage = self.data_service.get_wage_for_age(
                    params.education_level, current_age, current_year, age
                )
                if weekly_wage is None:
                    weekly_wage = self.data_service.get_income_for_education_level(
                        params.education_level
                    )
                    if weekly_wage is None:
                        continue
                    growth_rate = self.data_service.calculate_wage_growth_rate(
                        params.education_level
                    )
                    if growth_rate:
                        weekly_wage = weekly_wage * (
                            (1.0 + growth_rate) ** years_into_accumulation
                        )

                annual_wage = self.data_service.get_annual_wage(weekly_wage)
                annual_contrib = annual_wage * params.savings_rate

                # Get portfolio balance at END of this year
                portfolio_balance = result.median_path[i]

                # Calculate annual investment return:
                # Portfolio grew from prev_portfolio_balance to portfolio_balance
                # During this year, we contributed annual_contrib
                # So: return = portfolio_balance - prev_portfolio_balance - annual_contrib
                annual_return = (
                    portfolio_balance - prev_portfolio_balance - annual_contrib
                )

                # Add this year's contribution to cumulative
                cumulative_contrib += annual_contrib

                # Cumulative investment return = portfolio balance - cumulative contributions
                cumulative_return = portfolio_balance - cumulative_contrib

                years.append(year)
                annual_contributions.append(annual_contrib)
                cumulative_contributions.append(cumulative_contrib)
                portfolio_balances.append(portfolio_balance)
                cumulative_investment_returns.append(cumulative_return)
                annual_investment_returns.append(annual_return)

                prev_portfolio_balance = portfolio_balance

        if not years or len(years) < 2:
            st.warning(
                "Insufficient data to display savings and returns chart. Please ensure wage-based savings is enabled and you have at least one year of accumulation."
            )
            return

        # Create figure with single y-axis
        fig = go.Figure()

        # Portfolio balance line
        fig.add_trace(
            go.Scatter(
                x=years,
                y=portfolio_balances,
                name="Portfolio Balance",
                line=dict(color="#e74c3c", width=3),
                hovertemplate="Year: %{x}<br>Portfolio: $%{y:,.0f}<extra></extra>",
            )
        )

        # Cumulative contributions line
        fig.add_trace(
            go.Scatter(
                x=years,
                y=cumulative_contributions,
                name="Cumulative Contributions",
                line=dict(color="#31a354", width=2, dash="dash"),
                hovertemplate="Year: %{x}<br>Contributions: $%{y:,.0f}<extra></extra>",
                fill=None,
            )
        )

        # Investment returns (filled area between contributions and portfolio)
        fig.add_trace(
            go.Scatter(
                x=years,
                y=portfolio_balances,
                name="Cumulative Investment Returns",
                fill="tonexty",
                fillcolor="rgba(52, 152, 219, 0.3)",
                line=dict(color="rgba(52, 152, 219, 0.0)", width=0),
                hovertemplate="Year: %{x}<br>Cumulative Returns: $%{customdata:,.0f}<extra></extra>",
                customdata=cumulative_investment_returns,
                showlegend=True,
            )
        )

        # Show annual contributions and returns as grouped bars (not stacked, to compare them)
        # Use a secondary y-axis for annual values since they're much smaller than cumulative
        fig.add_trace(
            go.Bar(
                x=years,
                y=annual_contributions,
                name="Annual Contribution",
                marker_color="#2ecc71",
                opacity=0.7,
                yaxis="y2",
                hovertemplate="Year: %{x}<br>Contribution: $%{y:,.0f}<extra></extra>",
            )
        )

        # Annual investment returns
        fig.add_trace(
            go.Bar(
                x=years,
                y=annual_investment_returns,
                name="Annual Investment Return",
                marker_color="#3498db",
                opacity=0.7,
                yaxis="y2",
                hovertemplate="Year: %{x}<br>Return: $%{y:,.0f}<extra></extra>",
            )
        )

        # Add retirement transition marker
        retire_year = current_year + params.pre_retire_years
        if retire_year <= years[-1]:
            fig.add_vline(
                x=retire_year,
                line_dash="dot",
                line_color="red",
                annotation_text="Retirement",
                annotation_position="top",
            )

        # Update layout with dual y-axes (needed because cumulative values are much larger than annual)
        fig.update_layout(
            title="Savings Contributions & Investment Returns (Accumulation Phase Only)",
            xaxis_title="Year",
            yaxis=dict(
                title="Portfolio Balance & Cumulative Contributions ($)",
                side="left",
                tickformat="$,.0f",
            ),
            yaxis2=dict(
                title="Annual Contribution & Return ($)",
                side="right",
                overlaying="y",
                tickformat="$,.0f",
            ),
            hovermode="x unified",
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            height=500,
            barmode="group",  # Group bars to compare contributions vs returns
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add explanation
        st.markdown("---")
        st.markdown("### 💡 Understanding This Chart")

        # Calculate terminal wealth for display
        terminal_wealth_m = np.median(result.terminal_balances) / 1e6

        # Determine simulation type
        if result.data_limited:
            simulation_type = (
                "Monte Carlo simulation (statistical modeling based on historical data)"
            )
            if result.available_years:
                simulation_type += f" - {result.available_years:.1f} years of historical data available"
        else:
            simulation_type = (
                "Historical simulation (rolling windows of actual market data)"
            )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
            **Portfolio Balance** (red line): Your portfolio value during accumulation phase (up to retirement)

            **Cumulative Contributions** (green dashed): Total amount you've saved over time (wage × savings rate)

            **Cumulative Investment Returns** (blue area): Total investment gains = Portfolio Balance - Cumulative Contributions

            **Returns Source:** {simulation_type}

            **Note:** This chart shows accumulation only. Terminal wealth (${terminal_wealth_m:.2f}M) is the balance at the end of the full simulation (after retirement).
            """
            )
        with col2:
            st.markdown(
                """
            **Annual Contribution** (green bars): How much you save each year based on your wage and savings rate

            **Annual Investment Return** (blue bars): Investment gains/losses for that year

            As your portfolio grows, annual returns typically exceed annual contributions due to compounding.
            """
            )
