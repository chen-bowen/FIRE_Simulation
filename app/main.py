"""Main application for the retirement planner.

This module orchestrates the entire retirement planning application, including:
- UI components (sidebar, charts, results)
- Data services (market data fetching)
- Simulation services (historical, Monte Carlo, hybrid)
- Input validation and error handling
"""

import streamlit as st
import numpy as np
from datetime import date

# Import application components and services
from app.config import get_config
from app.schemas import SimulationParams, DataConfig
from app.services import DataService, SimulationService
from app.components import SidebarComponent, ChartComponent, ResultsComponent
from app.utils import validate_weights, align_weights_with_data, calculate_horizon_years, validate_age_inputs, validate_financial_inputs


def main():
    """Main application entry point.

    Sets up the Streamlit interface and orchestrates the retirement planning workflow:
    1. Initialize UI components and services
    2. Render sidebar inputs and validate them
    3. Calculate derived parameters (retirement horizons)
    4. Run simulations when user clicks "Run Simulation"
    5. Display results across multiple tabs
    """
    # Configure Streamlit page layout
    st.set_page_config(page_title="Retirement Planner", layout="wide", initial_sidebar_state="expanded")
    st.title("Retirement Planner: Historical + Monte Carlo")

    # Initialize UI components
    sidebar = SidebarComponent()  # Handles user input form
    charts = ChartComponent()  # Handles all visualizations
    results = ResultsComponent()  # Handles metrics and statistics display

    # Initialize business logic services
    data_service = DataService()  # Fetches market data from Yahoo Finance
    simulation_service = SimulationService()  # Runs historical, MC, and hybrid simulations

    # Render sidebar and collect user inputs
    inputs = sidebar.render()

    # Validate inputs
    try:
        sidebar.validate_inputs(inputs)
        validate_age_inputs(inputs["current_age"], inputs["retire_age"], inputs["plan_until_age"])
        validate_financial_inputs(inputs["initial_balance"], inputs["annual_contrib"], inputs["annual_spend"])
    except Exception as e:
        st.error(f"Input validation error: {str(e)}")
        st.stop()

    # Calculate derived parameters
    pre_retire_years, retire_years = calculate_horizon_years(inputs["current_age"], inputs["retire_age"], inputs["plan_until_age"])

    # Show warning for long horizons
    total_years = pre_retire_years + retire_years
    if total_years > 20:
        st.warning(f"⚠️ Planning {total_years} years ahead. Historical data may be limited to ~22 years for SPY/AGG.")

    # Run button
    if st.button("Run Simulation", type="primary"):
        try:
            # Fetch data
            with st.spinner("Fetching market data..."):
                returns_df = data_service.fetch_returns(inputs["tickers"], inputs["start"], inputs["end"], inputs["frequency"])

            # Align weights with available data
            weights = align_weights_with_data(inputs["weights"], list(returns_df.columns))

            # Create market data
            market_data = data_service.create_market_data(returns_df, weights, inputs["frequency"])

            # Create simulation parameters
            params = SimulationParams(
                initial_balance=inputs["initial_balance"],
                annual_contrib=inputs["annual_contrib"],
                annual_spend=inputs["annual_spend"],
                pre_retire_years=pre_retire_years,
                retire_years=retire_years,
                inflation_rate_annual=inputs["inflation"],
                frequency=inputs["frequency"],
                pacing=inputs["pacing"],
            )

            # Run simulations
            tab_hist, tab_mc, tab_hybrid, tab_compare = st.tabs(["Historical", "Monte Carlo", "Hybrid", "Comparison"])

            with tab_hist:
                st.subheader("Historical Backtest")

                with st.spinner("Running historical simulation..."):
                    historical_result = simulation_service.run_historical_simulation(returns_df, weights, params)

                # Display results
                results.display_metrics(historical_result, "Historical Results")
                results.display_data_warning(historical_result, total_years)

                # Plot paths
                charts.plot_simulation_paths(historical_result, "Historical Portfolio Paths")

                # Display statistics
                results.display_statistics(historical_result)

            with tab_mc:
                st.subheader("Monte Carlo Simulation")

                with st.spinner("Running Monte Carlo simulation..."):
                    mc_result = simulation_service.run_monte_carlo_simulation(
                        market_data.means, market_data.cov, weights, params, inputs["n_paths"], inputs["seed"]
                    )

                # Display results
                results.display_metrics(mc_result, "Monte Carlo Results")

                # Plot paths
                charts.plot_simulation_paths(mc_result, "Monte Carlo Portfolio Paths")

                # Plot terminal wealth histogram
                charts.plot_terminal_wealth_histogram(mc_result)

                # Display statistics
                results.display_statistics(mc_result)

                with tab_hybrid:
                    st.subheader("Hybrid Simulation (Historical + Monte Carlo)")

                    with st.spinner("Running hybrid simulation..."):
                        hybrid_result = simulation_service.run_hybrid_simulation(
                            returns_df, weights, params, inputs["n_paths"], inputs["seed"]
                        )

                    # Display results
                    results.display_metrics(hybrid_result, "Hybrid Results")
                    results.display_data_warning(hybrid_result, total_years)

                    # Plot paths
                    charts.plot_simulation_paths(hybrid_result, "Hybrid Portfolio Paths")

                    # Plot terminal wealth histogram
                    charts.plot_terminal_wealth_histogram(hybrid_result)

                    # Display statistics
                    results.display_statistics(hybrid_result)

                with tab_compare:
                    st.subheader("Historical vs Monte Carlo vs Hybrid Comparison")

                    # Plot comparison
                    charts.plot_comparison_chart(historical_result, mc_result)

                    # Display summary table
                    results.display_summary_table(inputs, historical_result, mc_result, hybrid_result)

        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
