"""Summary component for displaying pre-simulation summary and input details."""

import streamlit as st

from app.services import DataService


class SummaryComponent:
    """Component for displaying simulation summary and input details."""

    def __init__(self):
        pass

    def render_summary_cards(
        self, inputs: dict, pre_retire_years: int, retire_years: int, sidebar_component
    ) -> None:
        """Display pre-simulation summary cards.

        Args:
            inputs: Dictionary of user inputs
            pre_retire_years: Years until retirement
            retire_years: Years in retirement
            sidebar_component: SidebarComponent instance for portfolio chart
        """
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
                annual_spend_display = (
                    inputs["withdrawal_params"].total_annual_expense or 0.0
                )

            # Show spending label - wage-based if enabled
            spending_label = "Annual Spending"
            spending_help_text = "Annual retirement spending"
            if inputs.get("use_wage_based_spending"):
                replacement_ratio = inputs.get("replacement_ratio", 0.80)
                spending_label = f"Annual Spending (wage-based)"
                spending_help_text = (
                    f"Wage-based: {replacement_ratio*100:.0f}% of pre-retirement spending"
                )

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
            if (
                "portfolio_weights" in st.session_state
                and st.session_state.portfolio_weights
            ):
                portfolio_weights = {
                    k: v for k, v in st.session_state.portfolio_weights.items() if v > 0
                }
                if portfolio_weights:
                    fig = sidebar_component.create_portfolio_pie_chart(
                        portfolio_weights, initial_balance=inputs["initial_balance"]
                    )
                    # Remove title and adjust layout for better spacing
                    fig.update_layout(
                        title="",  # Remove title since "Portfolio" metric already serves as title
                        height=250,  # Reduced height for more compact display
                        margin=dict(
                            l=20, r=20, t=0, b=20
                        ),  # Minimal margins to bring chart closer to title
                        showlegend=False,  # Hide legend to prevent overlap with text labels
                    )
                    # Update center annotation font size for smaller chart
                    if fig.layout.annotations:
                        fig.layout.annotations[0].font.size = 12
                    st.plotly_chart(
                        fig, use_container_width=True, key="summary_portfolio_chart"
                    )
                else:
                    st.info("No portfolio allocation set")
            else:
                st.info("No portfolio allocation set")

    def render_input_summary(
        self, inputs: dict, pre_retire_years: int, total_years: int, data_service: DataService
    ) -> None:
        """Display expandable input summary.

        Args:
            inputs: Dictionary of user inputs
            pre_retire_years: Years until retirement
            total_years: Total planning period in years
            data_service: DataService instance for wage calculations
        """
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
                if (
                    inputs.get("use_wage_based_savings")
                    and inputs.get("education_level")
                    and inputs.get("savings_rate")
                ):
                    # Calculate first-year contribution from wage
                    weekly_wage = data_service.get_income_for_education_level(
                        inputs["education_level"]
                    )
                    if weekly_wage:
                        annual_wage = data_service.get_annual_wage(weekly_wage)
                        first_year_contrib = annual_wage * inputs["savings_rate"]
                        savings_pct = inputs["savings_rate"] * 100
                        st.write(f"- Annual Contribution: ${first_year_contrib:,.0f}")
                        st.write(
                            f"  (wage-based, {savings_pct:.0f}% of ${annual_wage:,.0f})"
                        )
                    else:
                        savings_pct = inputs["savings_rate"] * 100
                        st.write("- Annual Contribution: Wage-based")
                        st.write(
                            f"  (education: {inputs['education_level']}, rate: {savings_pct:.0f}%)"
                        )
                else:
                    st.write(f"- Annual Contribution: ${inputs['annual_contrib']:,.0f}")

                # Display annual spending
                annual_spend_display = inputs.get("annual_spend") or 0.0
                if inputs.get("withdrawal_params"):
                    annual_spend_display = (
                        inputs["withdrawal_params"].total_annual_expense or 0.0
                    )
                    st.write(f"- Annual Spending: ${annual_spend_display:,.0f}")
                    if inputs.get("use_wage_based_spending"):
                        replacement_ratio = inputs.get("replacement_ratio", 0.80)
                        st.write(
                            f"  (wage-based, {replacement_ratio*100:.0f}% of pre-retirement spending)"
                        )
                    st.write(
                        f"- CPI Adjustment: {'Yes' if inputs['withdrawal_params'].use_cpi_adjustment else 'No'}"
                    )
                else:
                    st.write(f"- Annual Spending: ${annual_spend_display:,.0f}")
                    if inputs.get("use_wage_based_spending"):
                        replacement_ratio = inputs.get("replacement_ratio", 0.80)
                        st.write(
                            f"  (wage-based, {replacement_ratio*100:.0f}% of pre-retirement spending)"
                        )
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

