"""Sidebar component for user inputs."""

from typing import Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.config import get_config
from app.schemas import ExpenseCategory, WithdrawalParams


class SidebarComponent:
    """Component for handling sidebar inputs."""

    # Expense category definitions with examples
    EXPENSE_CATEGORIES = {
        "Food and beverages": "Breakfast cereal, milk, coffee, chicken, wine, full service meals, snacks",
        "Housing": "Rent of primary residence, owners' equivalent rent, utilities, bedroom furniture",
        "Apparel": "Men's shirts and sweaters, women's dresses, baby clothes, shoes, jewelry",
        "Transportation": "New vehicles, airline fares, gasoline, motor vehicle insurance",
        "Medical care": "Prescription drugs, medical equipment and supplies, physicians' services, eyeglasses and eye care, hospital services",
        "Recreation": "Televisions, toys, pets and pet products, sports equipment, park and museum admissions",
        "Education and communication": "College tuition, postage, telephone services, computer software and accessories",
        "Other goods and services": "Tobacco and smoking products, haircuts and other personal services, funeral expenses",
    }

    EDUCATION_LEVELS = [
        "Less than high school",
        "High school",
        "Some college",
        "Bachelor's degree",
        "Master's degree",
        "Professional degree",
        "Doctorate",
    ]

    def __init__(self):
        self.config = get_config()

    def render(self) -> dict:
        """Render sidebar and return input values."""
        st.sidebar.header("Inputs")

        # Data settings
        freq = st.sidebar.selectbox(
            "Frequency",
            options=["monthly", "daily"],
            index=0 if self.config.default_frequency == "monthly" else 1,
        )

        pacing = "pro-rata"
        if freq == "daily":
            pacing = st.sidebar.selectbox(
                "Flow pacing (daily only)",
                options=["pro-rata", "monthly-boundary"],
                index=0,
                help="Apply contributions/spending pro-rata each trading day or only on monthly boundaries.",
            )

        # Date range
        start = st.sidebar.text_input(
            "Data start (YYYY-MM-DD)", value=self.config.default_start_date
        )
        end = st.sidebar.text_input(
            "Data end (YYYY-MM-DD)", value=self.config.default_end_date
        )

        # Tickers and weights
        tickers_input = st.sidebar.text_input(
            "Tickers (comma)",
            value=",".join(self.config.default_tickers),
            help="Use ^GSPC,^TNX for longer history (1950s+) or SPY,AGG for ETF data (2003+)",
        )
        weights_input = st.sidebar.text_input(
            "Weights for tickers (comma)",
            value=",".join(map(str, self.config.default_weights)),
        )

        # Financial inputs
        initial_balance = st.sidebar.number_input(
            "Current savings ($)",
            value=self.config.default_initial_balance,
            step=10000.0,
        )
        annual_contrib = st.sidebar.number_input(
            "Annual savings before retirement ($)",
            value=self.config.default_annual_contrib,
            step=1000.0,
        )
        # Dynamic withdrawal section
        st.sidebar.markdown("---")
        st.sidebar.subheader("Retirement Spending")

        use_dynamic_withdrawal = st.sidebar.checkbox(
            "Use dynamic withdrawal (CPI-adjusted)",
            value=False,
            help="Adjust spending based on CPI inflation data instead of fixed annual amount",
        )

        annual_spend = None
        withdrawal_params = None

        if use_dynamic_withdrawal:
            # Expense input mode selection
            expense_mode = st.sidebar.radio(
                "Expense input mode",
                options=["Total + Percentages", "Dollar amounts per category"],
                help="Choose how to specify your expenses",
            )

            if expense_mode == "Total + Percentages":
                # Mode 1: Total annual expense + category percentages
                total_annual_expense = st.sidebar.number_input(
                    "Total annual retirement spending ($)",
                    value=self.config.default_annual_spend,
                    step=1000.0,
                    min_value=0.0,
                )

                st.sidebar.markdown("### Expense Categories")
                st.sidebar.markdown(
                    "**Adjust percentages using the pie chart below or sliders**"
                )

                # Initialize category percentages in session state
                if "category_percentages" not in st.session_state:
                    # Default: equal distribution
                    default_pct = 100.0 / len(self.EXPENSE_CATEGORIES)
                    st.session_state.category_percentages = {
                        cat: default_pct for cat in self.EXPENSE_CATEGORIES.keys()
                    }

                # Pie chart for interactive percentage adjustment
                percentages = st.session_state.category_percentages
                fig = self._create_pie_chart(percentages, total_annual_expense)
                st.sidebar.plotly_chart(fig, use_container_width=True)

                # Sliders for fine-tuning percentages
                st.sidebar.markdown("**Adjust percentages:**")
                new_percentages = {}
                for category in self.EXPENSE_CATEGORIES.keys():
                    new_percentages[category] = st.sidebar.slider(
                        f"{category} (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=percentages.get(category, 0.0),
                        step=0.1,
                        help=self.EXPENSE_CATEGORIES[category],
                    )

                # Normalize percentages to sum to 100
                total_pct = sum(new_percentages.values())
                if total_pct > 0:
                    normalized = {
                        k: (v / total_pct) * 100.0 for k, v in new_percentages.items()
                    }
                    st.session_state.category_percentages = normalized
                else:
                    st.session_state.category_percentages = new_percentages

                # Create expense categories
                expense_categories = [
                    ExpenseCategory(
                        name=cat, percentage=st.session_state.category_percentages[cat]
                    )
                    for cat in self.EXPENSE_CATEGORIES.keys()
                ]

                withdrawal_params = WithdrawalParams(
                    expense_categories=expense_categories,
                    total_annual_expense=total_annual_expense,
                )

            else:
                # Mode 2: Dollar amounts per category
                st.sidebar.markdown("### Expense Categories")
                st.sidebar.markdown("**Enter dollar amounts for each category:**")

                expense_categories = []
                for category, examples in self.EXPENSE_CATEGORIES.items():
                    amount = st.sidebar.number_input(
                        f"{category} ($/year)",
                        value=0.0,
                        step=100.0,
                        min_value=0.0,
                        help=examples,
                    )
                    if amount > 0:
                        expense_categories.append(
                            ExpenseCategory(name=category, current_amount=amount)
                        )

                if not expense_categories:
                    st.sidebar.warning(
                        "Please enter at least one expense category amount"
                    )

                if expense_categories:
                    withdrawal_params = WithdrawalParams(
                        expense_categories=expense_categories
                    )

            # Wage and education inputs
            st.sidebar.markdown("---")
            st.sidebar.subheader("Income & Education (for future wage growth)")

            current_wage = st.sidebar.number_input(
                "Current annual wage/salary ($)",
                value=0.0,
                step=1000.0,
                min_value=0.0,
                help="Your current annual income (optional, for future wage growth calculations)",
            )

            education_level = st.sidebar.selectbox(
                "Education level",
                options=[""] + self.EDUCATION_LEVELS,
                index=0,
                help="Your education level (optional, for future wage growth calculations)",
            )

            if withdrawal_params and (current_wage > 0 or education_level):
                withdrawal_params.current_wage = (
                    current_wage if current_wage > 0 else None
                )
                withdrawal_params.education_level = (
                    education_level if education_level else None
                )

            use_cpi = st.sidebar.checkbox(
                "Use CPI-based inflation adjustment",
                value=True,
                help="Adjust spending based on historical CPI inflation data",
            )
            if withdrawal_params:
                withdrawal_params.use_cpi_adjustment = use_cpi

        else:
            # Traditional fixed annual spending
            annual_spend = st.sidebar.number_input(
                "Annual retirement spending ($)",
                value=self.config.default_annual_spend,
                step=1000.0,
            )

        # Age inputs
        current_age = st.sidebar.number_input(
            "Current age", value=self.config.default_current_age, step=1
        )
        retire_age = st.sidebar.number_input(
            "Retirement age", value=self.config.default_retire_age, step=1
        )
        plan_until_age = st.sidebar.number_input(
            "Plan until age", value=self.config.default_plan_until_age, step=1
        )

        # Other inputs
        inflation = (
            st.sidebar.number_input(
                "Inflation (%/yr)", value=self.config.default_inflation * 100, step=0.1
            )
            / 100.0
        )
        n_paths = st.sidebar.number_input(
            "MC paths", value=self.config.default_mc_paths, min_value=100, step=100
        )
        seed = st.sidebar.number_input(
            "Random seed (change for different Monte Carlo results)",
            value=self.config.default_seed,
            step=1,
        )

        # Parse inputs
        ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        weights = np.array([float(x.strip()) for x in weights_input.split(",")])

        return {
            "frequency": freq,
            "pacing": pacing,
            "start": start,
            "end": end,
            "tickers": ticker_list,
            "weights": weights,
            "initial_balance": initial_balance,
            "annual_contrib": annual_contrib,
            "annual_spend": annual_spend,
            "withdrawal_params": withdrawal_params,
            "current_age": current_age,
            "retire_age": retire_age,
            "plan_until_age": plan_until_age,
            "inflation": inflation,
            "n_paths": n_paths,
            "seed": seed,
        }

    def validate_inputs(self, inputs: dict) -> None:
        """Validate user inputs."""
        # Basic validation
        if inputs["retire_age"] <= inputs["current_age"]:
            st.error("Retirement age must be greater than current age")
            st.stop()

        if inputs["plan_until_age"] <= inputs["retire_age"]:
            st.error("Plan until age must be greater than retirement age")
            st.stop()

        if len(inputs["tickers"]) == 0:
            st.error("At least one ticker must be provided")
            st.stop()

        if len(inputs["weights"]) != len(inputs["tickers"]):
            st.warning(
                "Number of weights doesn't match number of tickers. Will be adjusted automatically."
            )

    def _create_pie_chart(
        self, percentages: Dict[str, float], total_expense: float
    ) -> go.Figure:
        """Create an interactive pie chart for expense categories."""
        labels = list(percentages.keys())
        values = [percentages[label] for label in labels]

        # Calculate dollar amounts for display
        amounts = [(pct / 100.0) * total_expense for pct in values]

        # Create hover text with both percentage and dollar amount
        hover_text = [
            f"{label}<br>{pct:.1f}% = ${amt:,.0f}/year"
            for label, pct, amt in zip(labels, values, amounts)
        ]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    textinfo="label+percent",
                    textposition="outside",
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=hover_text,
                    pull=[0.05] * len(labels),  # Slight pull for better visibility
                )
            ]
        )

        fig.update_layout(
            title="Expense Category Distribution",
            showlegend=True,
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return fig
