"""Results display component."""

import traceback
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from app.schemas import SimulationResult
from app.utils import format_currency, format_percentage


class ResultsComponent:
    """Component for displaying simulation results."""

    def __init__(self):
        pass

    def display_metrics(self, result: SimulationResult, title: str) -> None:
        """Display key metrics for a simulation result."""
        st.subheader(title)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Success Rate",
                format_percentage(result.success_rate),
                help="Percentage of scenarios where portfolio didn't deplete",
            )

        with col2:
            median_terminal = np.median(result.terminal_balances)
            # Format with appropriate units for better readability
            abs_terminal = abs(median_terminal)
            sign = "-" if median_terminal < 0 else ""
            if abs_terminal >= 1e9:
                # Billions: show 1-2 decimal places
                wealth_display = f"{sign}${abs_terminal/1e9:.2f}B"
            elif abs_terminal >= 1e6:
                # Millions: show 1-2 decimal places
                wealth_display = f"{sign}${abs_terminal/1e6:.2f}M"
            elif abs_terminal >= 1e3:
                # Thousands: show 1 decimal place
                wealth_display = f"{sign}${abs_terminal/1e3:.1f}K"
            else:
                # Less than 1000: show whole number
                wealth_display = f"{sign}${abs_terminal:,.0f}"
            st.metric(
                "Median Terminal Wealth",
                wealth_display,
                help="Median portfolio value at the end of the simulation",
            )

        with col3:
            st.metric(
                "Simulation Period",
                f"{result.horizon_periods / result.periods_per_year:.1f} years",
                help="Total simulation period",
            )

        with col4:
            if result.pre_retire_avg_spending is not None and result.pre_retire_avg_spending > 0:
                st.metric(
                    "Avg Pre-Retirement Spending",
                    format_currency(result.pre_retire_avg_spending),
                    help="Average annual spending during accumulation phase",
                )
            elif result.earliest_retirement_ages is not None and len(result.earliest_retirement_ages) > 0:
                median_earliest = np.median(result.earliest_retirement_ages)
                st.metric(
                    "Earliest Retirement",
                    f"Age {median_earliest:.0f}",
                    help="Median earliest possible retirement age (25x expenses rule)",
                )
            else:
                st.metric("", "")  # Empty column for alignment

    def display_rebalancing_events(self, result: SimulationResult) -> None:
        """Display rebalancing events if any occurred."""
        if result.rebalancing_events and len(result.rebalancing_events) > 0:
            st.markdown("---")
            st.subheader("📊 Portfolio Rebalancing Events")
            for event in result.rebalancing_events:
                st.info(f"📊 {event}")
            st.caption(
                "Portfolio allocations are rebalanced annually to maintain target weights. "
                "This resets all asset allocations back to your target percentages."
            )

    def display_summary_table(
        self,
        inputs: dict,
        historical_result: SimulationResult,
        mc_result: SimulationResult,
        hybrid_result: SimulationResult = None,
    ) -> None:
        """Display summary table of inputs and results."""
        st.subheader("Summary")

        # Create summary data
        parameters = [
            "Current Age",
            "Retirement Age",
            "Plan Until Age",
            "Initial Balance",
            "Annual Contribution",
            "Annual Spending",
            "Inflation Rate",
            "Frequency",
            "Historical Success Rate",
            "Monte Carlo Success Rate",
        ]

        values = [
            f"{inputs['current_age']}",
            f"{inputs['retire_age']}",
            f"{inputs['plan_until_age']}",
            format_currency(inputs["initial_balance"]),
            format_currency(inputs["annual_contrib"]),
            format_currency(inputs["annual_spend"]),
            format_percentage(inputs["inflation"]),
            inputs["frequency"].title(),
            format_percentage(historical_result.success_rate),
            format_percentage(mc_result.success_rate),
        ]

        # Add hybrid results if available
        if hybrid_result is not None:
            parameters.extend(
                [
                    "Hybrid Success Rate",
                    "Historical Median Terminal",
                    "Monte Carlo Median Terminal",
                    "Hybrid Median Terminal",
                ]
            )
            values.extend(
                [
                    format_percentage(hybrid_result.success_rate),
                    format_currency(np.median(historical_result.terminal_balances)),
                    format_currency(np.median(mc_result.terminal_balances)),
                    format_currency(np.median(hybrid_result.terminal_balances)),
                ]
            )
        else:
            parameters.extend(
                [
                    "Historical Median Terminal",
                    "Monte Carlo Median Terminal",
                ]
            )
            values.extend(
                [
                    format_currency(np.median(historical_result.terminal_balances)),
                    format_currency(np.median(mc_result.terminal_balances)),
                ]
            )

        summary_data = {
            "Parameter": parameters,
            "Value": values,
        }

        df = pd.DataFrame(summary_data)
        st.dataframe(df, use_container_width=True)

    def display_statistics(self, result: SimulationResult) -> None:
        """Display detailed statistics."""
        with st.expander("Detailed Statistics"):
            stats_data = {
                "Statistic": [
                    "Mean Terminal Wealth",
                    "Median Terminal Wealth",
                    "25th Percentile",
                    "75th Percentile",
                    "90th Percentile",
                    "95th Percentile",
                    "Worst Case",
                    "Best Case",
                ],
                "Value": [
                    format_currency(np.mean(result.terminal_balances)),
                    format_currency(np.median(result.terminal_balances)),
                    format_currency(np.percentile(result.terminal_balances, 25)),
                    format_currency(np.percentile(result.terminal_balances, 75)),
                    format_currency(np.percentile(result.terminal_balances, 90)),
                    format_currency(np.percentile(result.terminal_balances, 95)),
                    format_currency(np.min(result.terminal_balances)),
                    format_currency(np.max(result.terminal_balances)),
                ],
            }

            df = pd.DataFrame(stats_data)
            st.dataframe(df, use_container_width=True)

    def display_runtime(self) -> None:
        """Display simulation runtime if available.

        Shows runtime in seconds (if < 60s), minutes:seconds (if < 60m), or hours:minutes:seconds format.
        """
        if "simulation_runtime" in st.session_state:
            runtime = st.session_state["simulation_runtime"]
            if runtime < 60:
                runtime_text = f"⏱️ Simulation completed in {runtime:.2f} seconds"
            elif runtime < 3600:
                minutes = int(runtime // 60)
                seconds = runtime % 60
                runtime_text = f"⏱️ Simulation completed in {minutes}m {seconds:.1f}s"
            else:
                hours = int(runtime // 3600)
                minutes = int((runtime % 3600) // 60)
                seconds = runtime % 60
                runtime_text = f"⏱️ Simulation completed in {hours}h {minutes}m {seconds:.1f}s"
            st.caption(runtime_text)

    def display_validation_messages(self, simulation_result: SimulationResult, inputs: dict) -> None:
        """Display validation messages for retirement spending and early retirement metrics.

        Args:
            simulation_result: Simulation result object
            inputs: Dictionary of user inputs
        """
        # Display validation messages
        if simulation_result.pre_retire_avg_spending is not None and simulation_result.pre_retire_avg_spending > 0:
            retirement_spending = inputs.get("annual_spend") or 0.0
            if inputs.get("withdrawal_params"):
                retirement_spending = inputs["withdrawal_params"].total_annual_expense or 0.0
            if retirement_spending > 0:
                ratio = retirement_spending / simulation_result.pre_retire_avg_spending
                if 0.70 <= ratio <= 0.90:
                    st.success(f"✓ Retirement spending is {ratio*100:.0f}% of pre-retirement average (typical range: 70-90%)")
                elif 0.90 < ratio <= 1.00:
                    st.warning(f"⚠ Retirement spending is {ratio*100:.0f}% of pre-retirement average (high - consider reducing)")
                elif ratio > 1.00:
                    excess_pct = (ratio - 1.0) * 100
                    st.error(f"⚠ Retirement spending exceeds pre-retirement average by {excess_pct:.0f}% (unusual - may be unsustainable)")
                else:
                    st.info(
                        f"ℹ Retirement spending is {ratio*100:.0f}% of pre-retirement average (low - may indicate conservative planning)"
                    )

        # Display early retirement metrics
        if simulation_result.earliest_retirement_ages is not None and len(simulation_result.earliest_retirement_ages) > 0:
            median_earliest = np.median(simulation_result.earliest_retirement_ages)
            st.info(
                f"📅 **Earliest Possible Retirement:** age {median_earliest:.0f} \n\n"
                f"Based on 25x annual expenses rule (4% withdrawal rate)."
            )

    def display_all_results(
        self,
        simulation_result: SimulationResult,
        inputs: dict,
        charts,
        pre_retire_years: int,
        total_years: int,
        current_age: int,
        inputs_changed: bool,
        returns_df_cached: Optional = None,
        asset_class_mapping: Optional[dict] = None,
    ) -> None:
        """Orchestrate all result displays including tabs and charts.

        Args:
            simulation_result: Simulation result object
            inputs: Dictionary of user inputs
            charts: ChartComponent instance
            pre_retire_years: Years until retirement
            total_years: Total planning period in years
            current_age: Current age
            inputs_changed: Whether inputs have changed
            returns_df_cached: Cached returns dataframe
            asset_class_mapping: Asset class mapping from sidebar
        """
        # Display metrics
        self.display_metrics(simulation_result, "Simulation Results")

        # Display runtime if available
        self.display_runtime()

        # Display validation messages
        self.display_validation_messages(simulation_result, inputs)

        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Portfolio Performance", "Savings & Returns"])

        with tab1:
            # Plot interactive portfolio chart
            current_year = datetime.now().year
            # Get annual spending (wage-based calculation already done in simulation service)
            annual_spend = inputs.get("annual_spend") or 0.0
            if inputs.get("withdrawal_params"):
                annual_spend = inputs["withdrawal_params"].total_annual_expense or 0.0

            charts.plot_interactive_portfolio_chart(
                simulation_result,
                title="Portfolio Quantiles",
                current_age=inputs["current_age"],
                current_year=current_year,
                initial_balance=inputs["initial_balance"],
                pre_retire_years=pre_retire_years,
                annual_spend=annual_spend if annual_spend > 0 else None,
            )

        with tab2:
            # Plot savings contributions and returns breakdown
            # Only use stored params if they match current simulation (not stale)
            stored_params = None
            if simulation_result is not None and not inputs_changed:
                stored_params = st.session_state.get("simulation_params")
            if stored_params and stored_params.use_wage_based_savings and stored_params.education_level:
                current_year = stored_params.current_year or datetime.now().year
                try:
                    charts.plot_savings_and_returns_breakdown(
                        stored_params,
                        simulation_result,
                        stored_params.current_age or inputs["current_age"],
                        current_year,
                    )
                except Exception as e:
                    st.error(f"Error displaying savings and returns chart: {str(e)}")
                    st.code(traceback.format_exc())
            else:
                st.info('Enable "Detailed Plan" to see savings and returns breakdown.')

        # Plot interactive portfolio progress chart (replaces terminal wealth histogram)
        portfolio_weights_dict = None
        if "portfolio_weights" in st.session_state:
            portfolio_weights_dict = {k: v for k, v in st.session_state["portfolio_weights"].items() if v > 0}

        # Use stored values from current simulation result (only if result exists and matches)
        if simulation_result is not None and not inputs_changed:
            stored_pre_retire_years = st.session_state.get("simulation_pre_retire_years", pre_retire_years)
            stored_current_age = st.session_state.get("simulation_current_age", inputs["current_age"])
        else:
            # Use current inputs if no valid cached simulation
            stored_pre_retire_years = pre_retire_years
            stored_current_age = inputs["current_age"]

        charts.plot_terminal_wealth_histogram(
            simulation_result,
            pre_retire_years=stored_pre_retire_years,
            current_age=stored_current_age,
            portfolio_weights=portfolio_weights_dict,
            returns_df=returns_df_cached,
            initial_balance=inputs["initial_balance"],
            asset_class_mapping=asset_class_mapping,
        )
