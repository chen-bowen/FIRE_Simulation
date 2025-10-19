"""Sidebar component for user inputs."""

import streamlit as st
import numpy as np
from datetime import date
from ..config import get_config
from ..utils import validate_weights, align_weights_with_data


class SidebarComponent:
    """Component for handling sidebar inputs."""

    def __init__(self):
        self.config = get_config()

    def render(self) -> dict:
        """Render sidebar and return input values."""
        st.sidebar.header("Inputs")

        # Data settings
        freq = st.sidebar.selectbox("Frequency", options=["monthly", "daily"], index=0 if self.config.default_frequency == "monthly" else 1)

        pacing = "pro-rata"
        if freq == "daily":
            pacing = st.sidebar.selectbox(
                "Flow pacing (daily only)",
                options=["pro-rata", "monthly-boundary"],
                index=0,
                help="Apply contributions/spending pro-rata each trading day or only on monthly boundaries.",
            )

        # Date range
        start = st.sidebar.text_input("Data start (YYYY-MM-DD)", value=self.config.default_start_date)
        end = st.sidebar.text_input("Data end (YYYY-MM-DD)", value=self.config.default_end_date)

        # Tickers and weights
        tickers_input = st.sidebar.text_input(
            "Tickers (comma)",
            value=",".join(self.config.default_tickers),
            help="Use ^GSPC,^TNX for longer history (1950s+) or SPY,AGG for ETF data (2003+)",
        )
        weights_input = st.sidebar.text_input("Weights for tickers (comma)", value=",".join(map(str, self.config.default_weights)))

        # Financial inputs
        initial_balance = st.sidebar.number_input("Current savings ($)", value=self.config.default_initial_balance, step=10000.0)
        annual_contrib = st.sidebar.number_input(
            "Annual savings before retirement ($)", value=self.config.default_annual_contrib, step=1000.0
        )
        annual_spend = st.sidebar.number_input("Annual retirement spending ($)", value=self.config.default_annual_spend, step=1000.0)

        # Age inputs
        current_age = st.sidebar.number_input("Current age", value=self.config.default_current_age, step=1)
        retire_age = st.sidebar.number_input("Retirement age", value=self.config.default_retire_age, step=1)
        plan_until_age = st.sidebar.number_input("Plan until age", value=self.config.default_plan_until_age, step=1)

        # Other inputs
        inflation = st.sidebar.number_input("Inflation (%/yr)", value=self.config.default_inflation * 100, step=0.1) / 100.0
        n_paths = st.sidebar.number_input("MC paths", value=self.config.default_mc_paths, min_value=100, step=100)
        seed = st.sidebar.number_input("Random seed", value=self.config.default_seed, step=1)

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
            st.warning("Number of weights doesn't match number of tickers. Will be adjusted automatically.")
