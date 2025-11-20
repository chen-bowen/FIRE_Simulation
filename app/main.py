"""Main application for Financial Independence, Retire Early (FIRE).

This module orchestrates the entire FIRE planning application, including:
- UI components (sidebar, charts, results, summary)
- Data services (market data fetching)
- Simulation services (hybrid: historical accumulation + Monte Carlo retirement)
- Input validation and error handling
"""

import os
import sys

# Add project root to Python path before imports
# This ensures 'app' package can be found when Streamlit Cloud runs this file directly
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from app.components import (
    ChartComponent,
    ResultsComponent,
    SidebarComponent,
    SummaryComponent,
)
from app.services import DataService, SimulationController, SimulationService
from app.utils import (
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
    4. Display pre-simulation summary
    5. Run simulations when user clicks "Run Simulation"
    6. Display results across multiple tabs
    """
    # Configure Streamlit page layout
    st.set_page_config(page_title="Financial Independence, Retire Early", layout="wide", initial_sidebar_state="expanded")
    st.title("Financial Independence, Retire Early")

    # Initialize UI components
    sidebar = SidebarComponent()  # Handles user input form
    charts = ChartComponent()  # Handles all visualizations
    results = ResultsComponent()  # Handles metrics and statistics display
    summary = SummaryComponent()  # Handles pre-simulation summary display

    # Initialize business logic services
    data_service = DataService()  # Fetches market data from Yahoo Finance
    simulation_service = SimulationService()  # Runs hybrid simulation
    simulation_controller = SimulationController()  # Orchestrates simulation execution

    # Render sidebar and collect user inputs
    inputs = sidebar.render()

    # Validate inputs
    try:
        sidebar.validate_inputs(inputs)
        validate_age_inputs(inputs["current_age"], inputs["retire_age"], inputs["plan_until_age"])
        # Only validate annual_spend if not using dynamic withdrawal
        annual_spend = inputs["annual_spend"] if inputs["annual_spend"] is not None else 0.0
        validate_financial_inputs(inputs["initial_balance"], inputs["annual_contrib"], annual_spend)
    except Exception as e:
        st.error(f"Input validation error: {str(e)}")
        st.stop()

    # Calculate derived parameters
    pre_retire_years, retire_years = calculate_horizon_years(inputs["current_age"], inputs["retire_age"], inputs["plan_until_age"])
    total_years = pre_retire_years + retire_years

    # Display pre-simulation summary
    summary.render_summary_cards(inputs, pre_retire_years, retire_years, sidebar)
    summary.render_input_summary(inputs, pre_retire_years, total_years, data_service)

    # Handle input caching and simulation execution
    inputs_hash = simulation_controller.calculate_input_hash(inputs)
    inputs_changed = simulation_controller.detect_input_changes(inputs_hash)
    simulation_result, returns_df_cached = simulation_controller.get_cached_result(inputs_changed)

    # Run button - always runs fresh simulation
    if st.button("Run Simulation", type="primary"):
        simulation_result, returns_df_cached = simulation_controller.run_simulation(
            inputs,
            pre_retire_years,
            retire_years,
            data_service,
            simulation_service,
        )

    # Display results if we have a simulation result
    if simulation_result is not None:
        # Get asset class mapping from sidebar component
        asset_class_mapping = None
        if hasattr(sidebar, "ASSET_CLASSES"):
            asset_class_mapping = sidebar.ASSET_CLASSES

        results.display_all_results(
            simulation_result,
            inputs,
            charts,
            pre_retire_years,
            total_years,
            inputs["current_age"],
            inputs_changed,
            returns_df_cached,
            asset_class_mapping,
        )


if __name__ == "__main__":
    main()
