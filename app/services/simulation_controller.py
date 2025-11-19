"""Simulation controller for orchestrating simulation execution and caching."""

import hashlib
import json
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from app.schemas import SimulationParams, SimulationResult
from app.utils import align_weights_with_data

from .data_service import DataService
from .simulation_service import SimulationService


class SimulationController:
    """Controller for managing simulation execution and caching."""

    def __init__(self):
        pass

    def calculate_input_hash(self, inputs: dict) -> str:
        """Calculate hash of inputs for change detection.

        Args:
            inputs: Dictionary of user inputs

        Returns:
            MD5 hash string of inputs
        """
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
                    cat.name: (
                        cat.percentage if cat.percentage is not None else cat.current_amount
                    )
                    for cat in wp.expense_categories
                }
                inputs_for_hash["category_percentages"] = tuple(
                    sorted(category_percentages.items())
                )
        inputs_hash = hashlib.md5(
            json.dumps(inputs_for_hash, sort_keys=True, default=str).encode()
        ).hexdigest()
        return inputs_hash

    def detect_input_changes(self, inputs_hash: str) -> bool:
        """Detect if inputs have changed and clear cache if needed.

        Args:
            inputs_hash: Current inputs hash

        Returns:
            True if inputs changed, False otherwise
        """
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
        return inputs_changed

    def get_cached_result(
        self, inputs_changed: bool
    ) -> Tuple[Optional[SimulationResult], Optional[pd.DataFrame]]:
        """Get cached simulation result if available and inputs haven't changed.

        Args:
            inputs_changed: Whether inputs have changed

        Returns:
            Tuple of (simulation_result, returns_df_cached) or (None, None)
        """
        simulation_result = None
        returns_df_cached = None
        if not inputs_changed and "simulation_result" in st.session_state:
            simulation_result = st.session_state["simulation_result"]
            returns_df_cached = st.session_state.get("simulation_returns_df")
        return simulation_result, returns_df_cached

    def run_simulation(
        self,
        inputs: dict,
        pre_retire_years: int,
        retire_years: int,
        data_service: DataService,
        simulation_service: SimulationService,
    ) -> Tuple[Optional[SimulationResult], Optional[pd.DataFrame]]:
        """Execute full simulation workflow.

        Args:
            inputs: Dictionary of user inputs
            pre_retire_years: Years until retirement
            retire_years: Years in retirement
            data_service: DataService instance
            simulation_service: SimulationService instance

        Returns:
            Tuple of (simulation_result, returns_df) or (None, None) on error
        """
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

            # Get annual spending (wage-based calculation happens in simulation service)
            annual_spend_value = inputs["annual_spend"] or 0.0
            if inputs.get("withdrawal_params"):
                annual_spend_value = (
                    inputs["withdrawal_params"].total_annual_expense or 0.0
                )

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
                pre_retire_spending_tracked=inputs.get(
                    "pre_retire_spending_tracked", False
                ),
            )

            # Run simulation
            st.subheader("Simulation Results")
            st.markdown(
                "**Hybrid approach: historical data for accumulation phase, Monte Carlo projections for retirement phase**"
            )

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

            return simulation_result, returns_df

        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            st.exception(e)
            return None, None

