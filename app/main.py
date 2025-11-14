"""Main application for the retirement planner.

This module orchestrates the entire retirement planning application, including:
- UI components (sidebar, charts, results)
- Data services (market data fetching)
- Simulation services (historical, Monte Carlo, hybrid)
- Input validation and error handling
"""

import streamlit as st

from app.components import ChartComponent, ResultsComponent, SidebarComponent

# Import application components and services
from app.schemas import SimulationParams
from app.services import DataService, SimulationService
from app.utils import (
    align_weights_with_data,
    calculate_horizon_years,
    validate_age_inputs,
    validate_financial_inputs,
)


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
    st.set_page_config(
        page_title="Retirement Planner", layout="wide", initial_sidebar_state="expanded"
    )
    st.title("Retirement Planner: Historical + Monte Carlo")

    # Initialize UI components
    sidebar = SidebarComponent()  # Handles user input form
    charts = ChartComponent()  # Handles all visualizations
    results = ResultsComponent()  # Handles metrics and statistics display

    # Initialize business logic services
    data_service = DataService()  # Fetches market data from Yahoo Finance
    simulation_service = (
        SimulationService()
    )  # Runs historical, MC, and hybrid simulations

    # Render sidebar and collect user inputs
    inputs = sidebar.render()

    # Validate inputs
    try:
        sidebar.validate_inputs(inputs)
        validate_age_inputs(
            inputs["current_age"], inputs["retire_age"], inputs["plan_until_age"]
        )
        validate_financial_inputs(
            inputs["initial_balance"], inputs["annual_contrib"], inputs["annual_spend"]
        )
    except Exception as e:
        st.error(f"Input validation error: {str(e)}")
        st.stop()

    # Calculate derived parameters
    pre_retire_years, retire_years = calculate_horizon_years(
        inputs["current_age"], inputs["retire_age"], inputs["plan_until_age"]
    )

    # Show warning for long horizons
    total_years = pre_retire_years + retire_years
    if total_years > 20:
        st.warning(
            f"⚠️ Planning {total_years} years ahead. Historical data may be limited to ~22 years for SPY/AGG."
        )

    # Run button
    if st.button("Run Simulation", type="primary"):
        try:
            # Fetch data
            with st.spinner("Fetching market data..."):
                returns_df = data_service.fetch_returns(
                    inputs["tickers"],
                    inputs["start"],
                    inputs["end"],
                    inputs["frequency"],
                )

            # Align weights with available data
            weights = align_weights_with_data(
                inputs["weights"], list(returns_df.columns)
            )

            # Debug: Check alignment
            print(f"Debug - Original weights: {inputs['weights']}")
            print(f"Debug - Data columns: {list(returns_df.columns)}")
            print(f"Debug - Aligned weights: {weights}")
            print(f"Debug - Aligned weights shape: {weights.shape}")

            # Create simulation parameters
            params = SimulationParams(
                initial_balance=inputs["initial_balance"],
                annual_contrib=inputs["annual_contrib"],
                annual_spend=inputs["annual_spend"] or 0.0,  # Fallback if None
                pre_retire_years=pre_retire_years,
                retire_years=retire_years,
                inflation_rate_annual=inputs["inflation"],
                frequency=inputs["frequency"],
                pacing=inputs["pacing"],
                withdrawal_params=inputs.get("withdrawal_params"),  # May be None
            )

            # Debug: Verify weights alignment
            print(f"Debug - Final weights being passed to simulation: {weights}")
            print(f"Debug - Final weights shape: {weights.shape}")
            print(f"Debug - Data shape: {returns_df.shape}")

            # Run simulation (historical for pre-retirement, Monte Carlo for retirement)
            st.subheader("Simulation Results")
            st.markdown(
                "**Using historical data for accumulation phase and Monte Carlo for retirement phase**"
            )

            with st.spinner("Running simulation..."):
                simulation_result = simulation_service.run_simulation(
                    returns_df,
                    weights,
                    params,
                    inputs["n_paths"],
                    inputs["seed"],
                )

            # Display results
            results.display_metrics(simulation_result, "Simulation Results")
            results.display_data_warning(simulation_result, total_years)

            # Plot paths
            charts.plot_simulation_paths(simulation_result, "Portfolio Paths")

            # Plot terminal wealth histogram
            charts.plot_terminal_wealth_histogram(simulation_result)

            # Display statistics
            results.display_statistics(simulation_result)

        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
