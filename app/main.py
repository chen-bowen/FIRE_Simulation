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
        # Only validate annual_spend if not using dynamic withdrawal
        annual_spend = (
            inputs["annual_spend"] if inputs["annual_spend"] is not None else 0.0
        )
        validate_financial_inputs(
            inputs["initial_balance"], inputs["annual_contrib"], annual_spend
        )
    except Exception as e:
        st.error(f"Input validation error: {str(e)}")
        st.stop()

    # Calculate derived parameters
    pre_retire_years, retire_years = calculate_horizon_years(
        inputs["current_age"], inputs["retire_age"], inputs["plan_until_age"]
    )
    total_years = pre_retire_years + retire_years

    # Display pre-simulation summary cards
    st.markdown("---")
    st.subheader("📊 Simulation Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Years to Retirement",
            f"{pre_retire_years}",
            help="Years until retirement age",
        )

    with col2:
        st.metric(
            "Retirement Duration",
            f"{retire_years} years",
            help="Years from retirement to plan until age",
        )

    with col3:
        annual_spend_display = inputs.get("annual_spend") or 0.0
        if inputs.get("withdrawal_params"):
            annual_spend_display = (
                inputs["withdrawal_params"].total_annual_expense or 0.0
            )
        st.metric(
            "Annual Spending",
            f"${annual_spend_display:,.0f}",
            help="Annual retirement spending",
        )

    with col4:
        st.metric(
            "Portfolio",
            "",
            help="Portfolio allocation",
        )
        if (
            "portfolio_weights" in st.session_state
            and st.session_state.portfolio_weights
        ):
            portfolio_weights = {
                k: v for k, v in st.session_state.portfolio_weights.items() if v > 0
            }
            if portfolio_weights:
                fig = sidebar._create_portfolio_pie_chart(portfolio_weights)
                # Add title with asset class count and adjust layout for better text visibility
                num_assets = len(portfolio_weights)
                asset_text = "asset class" if num_assets == 1 else "asset classes"
                fig.update_layout(
                    title=f"{num_assets} {asset_text}",
                    height=250,
                    margin=dict(
                        l=40, r=40, t=40, b=40
                    ),  # Increased margins for outside text
                    showlegend=False,  # Hide legend to prevent overlap with text labels
                )
                st.plotly_chart(
                    fig, use_container_width=True, key="summary_portfolio_chart"
                )
            else:
                st.info("No portfolio allocation set")
        else:
            st.info("No portfolio allocation set")

    # Show input summary in expander
    with st.expander("📋 View Input Summary", expanded=False):
        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:
            st.markdown("**Timeline**")
            st.write(f"- Current Age: {inputs['current_age']}")
            st.write(f"- Retirement Age: {inputs['retire_age']}")
            st.write(f"- Plan Until Age: {inputs['plan_until_age']}")
            st.write(f"- Total Planning Period: {total_years} years")

            st.markdown("**Financial**")
            st.write(f"- Initial Balance: ${inputs['initial_balance']:,.0f}")
            st.write(f"- Annual Contribution: ${inputs['annual_contrib']:,.0f}")
            if inputs.get("withdrawal_params"):
                st.write(
                    f"- Annual Spending: ${inputs['withdrawal_params'].total_annual_expense:,.0f}"
                )
                st.write(
                    f"- CPI Adjustment: {'Yes' if inputs['withdrawal_params'].use_cpi_adjustment else 'No'}"
                )
            else:
                st.write(f"- Annual Spending: ${inputs.get('annual_spend', 0):,.0f}")
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
            st.write(f"- Monte Carlo Paths: {inputs['n_paths']}")
            st.write(f"- Random Seed: {inputs['seed']}")

    st.markdown("---")

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
            charts.plot_simulation_paths(
                simulation_result, "Portfolio Paths", current_age=inputs["current_age"]
            )

            # Plot terminal wealth histogram
            charts.plot_terminal_wealth_histogram(simulation_result)

            # Display statistics
            results.display_statistics(simulation_result)

        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
