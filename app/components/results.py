"""Results display component."""

import streamlit as st
import pandas as pd
import numpy as np
from app.schemas import SimulationResult
from app.utils import format_percentage, format_currency


class ResultsComponent:
    """Component for displaying simulation results."""

    def __init__(self):
        pass

    def display_metrics(self, result: SimulationResult, title: str) -> None:
        """Display key metrics for a simulation result."""
        st.subheader(title)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Success Rate", format_percentage(result.success_rate), help="Percentage of scenarios where portfolio didn't deplete")

        with col2:
            median_terminal = np.median(result.terminal_balances)
            st.metric(
                "Median Terminal Wealth", format_currency(median_terminal), help="Median portfolio value at the end of the simulation"
            )

        with col3:
            if result.data_limited:
                st.metric(
                    "Data Used", f"{result.available_years:.1f} years", help="Years of historical data used (may be less than requested)"
                )
            else:
                st.metric(
                    "Simulation Period", f"{result.horizon_periods / result.periods_per_year:.1f} years", help="Total simulation period"
                )

    def display_data_warning(self, result: SimulationResult, total_years: int) -> None:
        """Display warning about data limitations."""
        if result.data_limited:
            st.info(
                f"📊 Using {result.available_years:.1f} years of available data " f"(scaled down from {total_years:.0f} years requested)"
            )
            st.caption("Note: Retirement planning phases have been proportionally scaled to fit available data.")

    def display_summary_table(
        self, inputs: dict, historical_result: SimulationResult, mc_result: SimulationResult, hybrid_result: SimulationResult = None
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
