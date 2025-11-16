"""Results display component."""

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

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Success Rate",
                format_percentage(result.success_rate),
                help="Percentage of scenarios where portfolio didn't deplete",
            )

        with col2:
            median_terminal = np.median(result.terminal_balances)
            # Format in millions for better readability
            if median_terminal >= 1e6:
                wealth_display = f"${median_terminal/1e6:.2f}M"
            elif median_terminal >= 1e3:
                wealth_display = f"${median_terminal/1e3:.1f}K"
            else:
                wealth_display = f"${median_terminal:.0f}"
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

    def display_data_warning(self, result: SimulationResult, total_years: int) -> None:
        """Display warning about data limitations."""
        # Removed - not needed since we simulate forward anyway using Monte Carlo
        pass

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
