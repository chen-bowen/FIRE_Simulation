"""Main application for the retirement planner.

This module orchestrates the entire retirement planning application, including:
- UI components (sidebar, charts, results)
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

from datetime import datetime

import numpy as np
import streamlit as st

from app.components import (
    ChartComponent,
    ResultsComponent,
    SidebarComponent,
)
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
    st.set_page_config(page_title="Retirement Planner", layout="wide", initial_sidebar_state="expanded")
    st.title("Retirement Planner")

    # Initialize UI components
    sidebar = SidebarComponent()  # Handles user input form
    charts = ChartComponent()  # Handles all visualizations
    results = ResultsComponent()  # Handles metrics and statistics display

    # Initialize business logic services
    data_service = DataService()  # Fetches market data from Yahoo Finance
    simulation_service = SimulationService()  # Runs hybrid simulation (historical accumulation + Monte Carlo retirement)

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

    # Display pre-simulation summary cards
    st.markdown("---")
    st.subheader("📊 Simulation Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if pre_retire_years == 0:
            st.metric(
                "Years to Retirement",
                "Already Retired",
                help="You are already retired - simulation starts from retirement phase",
            )
        else:
            st.metric(
                "Years to Retirement",
                f"{pre_retire_years:,}",
                help="Years until retirement age",
            )

    with col2:
        st.metric(
            "Retirement Duration",
            f"{retire_years:,} years",
            help="Years from retirement to plan until age",
        )

    with col3:
        annual_spend_display = inputs.get("annual_spend") or 0.0
        if inputs.get("withdrawal_params"):
            annual_spend_display = inputs["withdrawal_params"].total_annual_expense or 0.0

        # Show spending label - wage-based if enabled
        spending_label = "Annual Spending"
        spending_help_text = "Annual retirement spending"
        if inputs.get("use_wage_based_spending"):
            replacement_ratio = inputs.get("replacement_ratio", 0.80)
            spending_label = f"Annual Spending (wage-based)"
            spending_help_text = f"Wage-based: {replacement_ratio*100:.0f}% of pre-retirement spending"

        st.metric(
            spending_label,
            f"${annual_spend_display:,.0f}",
            help=spending_help_text,
        )

    with col4:
        st.metric(
            "Portfolio",
            "",
            help="Portfolio allocation",
        )
        if "portfolio_weights" in st.session_state and st.session_state.portfolio_weights:
            portfolio_weights = {k: v for k, v in st.session_state.portfolio_weights.items() if v > 0}
            if portfolio_weights:
                fig = sidebar._create_portfolio_pie_chart(portfolio_weights, initial_balance=inputs["initial_balance"])
                # Remove title and adjust layout for better spacing
                fig.update_layout(
                    title="",  # Remove title since "Portfolio" metric already serves as title
                    height=250,  # Reduced height for more compact display
                    margin=dict(l=20, r=20, t=0, b=20),  # Minimal margins to bring chart closer to title
                    showlegend=False,  # Hide legend to prevent overlap with text labels
                )
                # Update center annotation font size for smaller chart
                if fig.layout.annotations:
                    fig.layout.annotations[0].font.size = 12
                st.plotly_chart(fig, use_container_width=True, key="summary_portfolio_chart")
            else:
                st.info("No portfolio allocation set")
        else:
            st.info("No portfolio allocation set")

    # Show input summary in expander
    with st.expander("📋 View Input Summary", expanded=False):
        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.markdown("**Timeline**")
            st.write(f"- Current Age: {inputs['current_age']:,}")
            st.write(f"- Retirement Age: {inputs['retire_age']:,}")
            st.write(f"- Plan Until Age: {inputs['plan_until_age']:,}")
            st.write(f"- Total Planning Period: {total_years:,} years")

            st.markdown("**Financial**")
            st.write(f"- Initial Balance: ${inputs['initial_balance']:,.0f}")

            # Show annual contribution - calculate from wage if wage-based savings is enabled
            if inputs.get("use_wage_based_savings") and inputs.get("education_level") and inputs.get("savings_rate"):
                # Calculate first-year contribution from wage
                weekly_wage = data_service.get_income_for_education_level(inputs["education_level"])
                if weekly_wage:
                    annual_wage = data_service.get_annual_wage(weekly_wage)
                    first_year_contrib = annual_wage * inputs["savings_rate"]
                    savings_pct = inputs["savings_rate"] * 100
                    st.write(f"- Annual Contribution: ${first_year_contrib:,.0f}")
                    st.write(f"  (wage-based, {savings_pct:.0f}% of ${annual_wage:,.0f})")
                else:
                    savings_pct = inputs["savings_rate"] * 100
                    st.write("- Annual Contribution: Wage-based")
                    st.write(f"  (education: {inputs['education_level']}, rate: {savings_pct:.0f}%)")
            else:
                st.write(f"- Annual Contribution: ${inputs['annual_contrib']:,.0f}")

            # Display annual spending
            annual_spend_display = inputs.get("annual_spend") or 0.0
            if inputs.get("withdrawal_params"):
                annual_spend_display = inputs["withdrawal_params"].total_annual_expense or 0.0
                st.write(f"- Annual Spending: ${annual_spend_display:,.0f}")
                if inputs.get("use_wage_based_spending"):
                    replacement_ratio = inputs.get("replacement_ratio", 0.80)
                    st.write(f"  (wage-based, {replacement_ratio*100:.0f}% of pre-retirement spending)")
                st.write(f"- CPI Adjustment: {'Yes' if inputs['withdrawal_params'].use_cpi_adjustment else 'No'}")
            else:
                st.write(f"- Annual Spending: ${annual_spend_display:,.0f}")
                if inputs.get("use_wage_based_spending"):
                    replacement_ratio = inputs.get("replacement_ratio", 0.80)
                    st.write(f"  (wage-based, {replacement_ratio*100:.0f}% of pre-retirement spending)")
                st.write(f"- Inflation Rate: {inputs['inflation']*100:.2f}%")

        with summary_col2:
            st.markdown("**Portfolio**")
            if "portfolio_weights" in st.session_state:
                for asset, weight in st.session_state["portfolio_weights"].items():
                    if weight > 0:
                        st.write(f"- {asset}: {weight:.1f}%")
            else:
                st.write("- Not configured")

            st.markdown("**Simulation Settings**")
            st.write(f"- Frequency: {inputs['frequency'].title()}")
            st.write(f"- Monte Carlo Paths: {inputs['n_paths']:,}")
            st.write(f"- Random Seed: {inputs['seed']:,}")

    st.markdown("---")

    # Create a hash of current inputs to detect changes
    import hashlib
    import json

    # Convert inputs to a hashable format (excluding withdrawal_params which may not be JSON serializable)
    inputs_for_hash = {k: v for k, v in inputs.items() if k != "withdrawal_params"}
    if "withdrawal_params" in inputs and inputs["withdrawal_params"] is not None:
        # Include withdrawal params summary
        wp = inputs["withdrawal_params"]
        inputs_for_hash["withdrawal_total"] = wp.total_annual_expense
        inputs_for_hash["withdrawal_cpi"] = wp.use_cpi_adjustment
        # Include category percentages to detect changes in spending distribution
        if wp.expense_categories:
            category_percentages = {
                cat.name: (cat.percentage if cat.percentage is not None else cat.current_amount) for cat in wp.expense_categories
            }
            inputs_for_hash["category_percentages"] = tuple(sorted(category_percentages.items()))
    inputs_hash = hashlib.md5(json.dumps(inputs_for_hash, sort_keys=True, default=str).encode()).hexdigest()

    # Check if inputs have changed - if so, clear ALL cached simulation data
    inputs_changed = False
    if "last_inputs_hash" in st.session_state:
        if st.session_state["last_inputs_hash"] != inputs_hash:
            inputs_changed = True
            # Inputs changed - clear all simulation-related session state
            simulation_keys = [
                "simulation_result",
                "simulation_params",
                "simulation_pre_retire_years",
                "simulation_current_age",
                "simulation_returns_df",
                "simulation_initial_balance",
                "portfolio_year_slider",
            ]
            for key in simulation_keys:
                if key in st.session_state:
                    del st.session_state[key]

    # Store current inputs hash
    st.session_state["last_inputs_hash"] = inputs_hash

    # Only use cached results if inputs haven't changed
    simulation_result = None
    returns_df_cached = None
    if not inputs_changed and "simulation_result" in st.session_state:
        simulation_result = st.session_state["simulation_result"]
        returns_df_cached = st.session_state.get("simulation_returns_df")

    # Run button - always runs fresh simulation
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
            weights = align_weights_with_data(inputs["weights"], list(returns_df.columns))

            # Get annual spending (wage-based calculation happens in simulation service)
            annual_spend_value = inputs["annual_spend"] or 0.0
            if inputs.get("withdrawal_params"):
                annual_spend_value = inputs["withdrawal_params"].total_annual_expense or 0.0

            # Create simulation parameters
            params = SimulationParams(
                initial_balance=inputs["initial_balance"],
                annual_contrib=inputs["annual_contrib"],
                annual_spend=annual_spend_value,
                pre_retire_years=pre_retire_years,
                retire_years=retire_years,
                inflation_rate_annual=inputs["inflation"],
                frequency=inputs["frequency"],
                pacing=inputs["pacing"],
                withdrawal_params=inputs.get("withdrawal_params"),  # May be None
                use_wage_based_savings=inputs.get("use_wage_based_savings", False),
                savings_rate=inputs.get("savings_rate"),
                savings_rate_profile=inputs.get("savings_rate_profile"),
                education_level=inputs.get("education_level"),
                current_age=inputs["current_age"],
                current_year=inputs.get("current_year"),
                use_wage_based_spending=inputs.get("use_wage_based_spending", False),
                replacement_ratio=inputs.get("replacement_ratio"),
                pre_retire_spending_tracked=inputs.get("pre_retire_spending_tracked", False),
            )

            # Run simulation
            st.subheader("Simulation Results")
            st.markdown("**Hybrid approach: historical data for accumulation phase, Monte Carlo projections for retirement phase**")

            with st.spinner("Running simulation..."):
                simulation_result = simulation_service.run_simulation(
                    returns_df,
                    weights,
                    params,
                    inputs["n_paths"],
                    inputs["seed"],
                )

            # Store simulation result and related data in session state
            # Clear any old simulation data first to ensure clean state
            simulation_keys = [
                "simulation_result",
                "simulation_params",
                "simulation_pre_retire_years",
                "simulation_current_age",
                "simulation_returns_df",
                "simulation_initial_balance",
            ]
            for key in simulation_keys:
                if key in st.session_state:
                    del st.session_state[key]

            # Store new simulation results
            st.session_state["simulation_result"] = simulation_result
            st.session_state["simulation_params"] = params
            st.session_state["simulation_pre_retire_years"] = pre_retire_years
            st.session_state["simulation_current_age"] = inputs["current_age"]
            st.session_state["simulation_returns_df"] = returns_df
            st.session_state["simulation_initial_balance"] = inputs["initial_balance"]
            returns_df_cached = returns_df

        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            st.exception(e)
            simulation_result = None

    # Display results if we have a simulation result
    if simulation_result is not None:
        # Display results
        results.display_metrics(simulation_result, "Simulation Results")
        results.display_data_warning(simulation_result, total_years)

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
            p10_earliest = np.percentile(simulation_result.earliest_retirement_ages, 10)
            p90_earliest = np.percentile(simulation_result.earliest_retirement_ages, 90)
            st.info(
                f"📅 **Earliest Possible Retirement:** age {median_earliest:.0f} \n\n"
                f"Based on 25x annual expenses rule (4% withdrawal rate)."
            )

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
                    import traceback

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

        # Get asset class mapping from sidebar component
        asset_class_mapping = None
        if hasattr(sidebar, "ASSET_CLASSES"):
            asset_class_mapping = sidebar.ASSET_CLASSES

        charts.plot_terminal_wealth_histogram(
            simulation_result,
            pre_retire_years=stored_pre_retire_years,
            current_age=stored_current_age,
            portfolio_weights=portfolio_weights_dict,
            returns_df=returns_df_cached,
            initial_balance=inputs["initial_balance"],
            asset_class_mapping=asset_class_mapping,
        )


if __name__ == "__main__":
    main()
