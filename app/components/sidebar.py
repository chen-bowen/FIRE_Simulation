"""Sidebar component for user inputs."""

from typing import Dict, Tuple

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

    # Expense category presets based on typical US household spending
    EXPENSE_PRESETS = {
        "Typical US Household": {
            "Food and beverages": 13.0,
            "Housing": 33.0,
            "Apparel": 3.0,
            "Transportation": 16.0,
            "Medical care": 8.0,
            "Recreation": 5.0,
            "Education and communication": 6.0,
            "Other goods and services": 16.0,
        },
        "Conservative": {
            "Food and beverages": 15.0,
            "Housing": 30.0,
            "Apparel": 2.0,
            "Transportation": 15.0,
            "Medical care": 12.0,
            "Recreation": 3.0,
            "Education and communication": 4.0,
            "Other goods and services": 19.0,
        },
    }

    # Education-based expense category distributions
    # Higher education levels tend to spend more on education/healthcare and less on basic needs
    EDUCATION_EXPENSE_PRESETS = {
        "Less than high school": {
            "Food and beverages": 15.0,
            "Housing": 35.0,
            "Apparel": 4.0,
            "Transportation": 18.0,
            "Medical care": 6.0,
            "Recreation": 4.0,
            "Education and communication": 3.0,
            "Other goods and services": 15.0,
        },
        "High school": {
            "Food and beverages": 14.0,
            "Housing": 34.0,
            "Apparel": 3.5,
            "Transportation": 17.0,
            "Medical care": 7.0,
            "Recreation": 4.5,
            "Education and communication": 4.0,
            "Other goods and services": 15.0,
        },
        "Some college": {
            "Food and beverages": 13.5,
            "Housing": 33.5,
            "Apparel": 3.0,
            "Transportation": 16.5,
            "Medical care": 7.5,
            "Recreation": 5.0,
            "Education and communication": 5.0,
            "Other goods and services": 15.0,
        },
        "Bachelor's degree": {
            "Food and beverages": 12.0,
            "Housing": 32.0,
            "Apparel": 2.5,
            "Transportation": 15.0,
            "Medical care": 9.0,
            "Recreation": 6.0,
            "Education and communication": 7.0,
            "Other goods and services": 16.5,
        },
        "Master's degree": {
            "Food and beverages": 11.0,
            "Housing": 31.0,
            "Apparel": 2.0,
            "Transportation": 14.0,
            "Medical care": 10.0,
            "Recreation": 7.0,
            "Education and communication": 8.0,
            "Other goods and services": 17.0,
        },
        "Professional degree": {
            "Food and beverages": 10.0,
            "Housing": 30.0,
            "Apparel": 2.0,
            "Transportation": 13.0,
            "Medical care": 11.0,
            "Recreation": 8.0,
            "Education and communication": 9.0,
            "Other goods and services": 17.0,
        },
        "Doctorate": {
            "Food and beverages": 10.0,
            "Housing": 30.0,
            "Apparel": 2.0,
            "Transportation": 13.0,
            "Medical care": 11.0,
            "Recreation": 8.0,
            "Education and communication": 9.0,
            "Other goods and services": 17.0,
        },
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

    # Portfolio presets for quick allocation
    PORTFOLIO_PRESETS = {
        "Conservative (30/70)": {"US Stocks": 30.0, "Bonds": 70.0},
        "Moderate (60/40)": {"US Stocks": 60.0, "Bonds": 40.0},
        "Aggressive (90/10)": {"US Stocks": 90.0, "Bonds": 10.0},
    }

    # Asset class definitions with ticker symbols (prioritized by data availability)
    ASSET_CLASSES = {
        "US Stocks": [
            "^GSPC",
            "SPY",
            "VTI",
        ],  # Prefer ^GSPC for longer history, SPY/VTI for ETFs
        "International Stocks": [
            "^EFA",
            "VEA",
            "VXUS",
        ],  # Prefer ^EFA for longer history, VEA/VXUS for ETFs
        "Bonds": [
            "^TNX",
            "AGG",
            "BND",
        ],  # Prefer ^TNX for longer history, AGG/BND for ETFs
        "Cash": [
            "^IRX",
            "SHV",
            "BIL",
        ],  # Prefer ^IRX for longer history, SHV/BIL for ETFs
        "Crypto": [
            "BTC-USD",
            "ETH-USD",
        ],  # Bitcoin and Ethereum - limited historical data
        "Real Estate": ["VNQ", "IYR", "^GSPTSE"],  # REITs, prefer VNQ/IYR ETFs
        "Commodities": ["GLD", "DBC", "^GSPC"],  # Gold and commodities ETFs
    }

    def __init__(self):
        self.config = get_config()

    def render(self) -> dict:
        """Render sidebar and return input values."""
        st.sidebar.header("Inputs")

        # ===== ESSENTIAL INPUTS (Always Visible) =====

        # Age inputs
        st.sidebar.subheader("Age & Timeline")
        current_age = st.sidebar.number_input(
            "Current age", value=self.config.default_current_age, step=1
        )
        retire_age = st.sidebar.number_input(
            "Retirement age", value=self.config.default_retire_age, step=1
        )
        plan_until_age = st.sidebar.number_input(
            "Plan until age", value=self.config.default_plan_until_age, step=1
        )

        # Financial inputs
        st.sidebar.subheader("Savings & Spending")
        initial_balance = st.sidebar.number_input(
            "Current savings ($)",
            value=self.config.default_initial_balance,
            step=10000.0,
        )

        # Initialize wage-based savings toggle in session state
        if "use_wage_based_savings" not in st.session_state:
            st.session_state.use_wage_based_savings = False
        if "savings_rate" not in st.session_state:
            st.session_state.savings_rate = 0.15  # Default 15%

        # Initialize wage-based savings variables (will be set later)
        use_wage_based_savings = False
        savings_rate = None
        education_level = ""  # Will be set in dynamic withdrawal section

        # Annual contribution input (conditional on wage-based savings)
        # This will be set later if wage-based savings is not used
        annual_contrib = self.config.default_annual_contrib

        # Portfolio allocation
        st.sidebar.markdown("---")
        st.sidebar.subheader("Portfolio Allocation")

        # Initialize portfolio weights in session state
        if "portfolio_weights" not in st.session_state:
            # Default to Moderate 60/40
            st.session_state.portfolio_weights = self.PORTFOLIO_PRESETS[
                "Moderate (60/40)"
            ].copy()

        # Initialize mode selection
        if "portfolio_mode_selection" not in st.session_state:
            st.session_state.portfolio_mode_selection = "Use Preset"

        # Mode selection: Preset or Custom
        mode_selection = st.sidebar.radio(
            "Portfolio mode",
            options=["Use Preset", "Custom"],
            index=0 if st.session_state.portfolio_mode_selection == "Use Preset" else 1,
            help="Choose a preset allocation or customize your own",
        )
        st.session_state.portfolio_mode_selection = mode_selection

        is_custom_mode = mode_selection == "Custom"
        selected_preset = None
        preset_changed = False

        # Track previous preset to detect changes
        if "previous_portfolio_preset" not in st.session_state:
            st.session_state.previous_portfolio_preset = None

        if not is_custom_mode:
            # Show preset selection when "Use Preset" is selected
            preset_options = list(self.PORTFOLIO_PRESETS.keys())
            selected_preset = st.sidebar.selectbox(
                "Choose preset",
                options=preset_options,
                index=1,  # Default to "Moderate (60/40)"
                help="Select a predefined portfolio allocation",
            )

            # Track if preset changed to sync slider
            if selected_preset != st.session_state.previous_portfolio_preset:
                if selected_preset in self.PORTFOLIO_PRESETS:
                    st.session_state.portfolio_weights = self.PORTFOLIO_PRESETS[
                        selected_preset
                    ].copy()
                    preset_changed = True
                st.session_state.previous_portfolio_preset = selected_preset

            st.sidebar.caption(f"📋 Using preset: **{selected_preset}**")
        else:
            # Custom mode
            st.sidebar.info(
                "🎨 **Custom Mode**: Adjust asset classes and allocations below"
            )
            # Reset previous preset when switching to custom
            if st.session_state.previous_portfolio_preset is not None:
                st.session_state.previous_portfolio_preset = None

        # Multi-Asset mode
        # Checkboxes to enable/disable asset classes
        st.sidebar.markdown("**Select asset classes:**")
        enabled_assets = {}
        for asset_class in self.ASSET_CLASSES.keys():
            # Initialize checkbox state if not present
            checkbox_key = f"asset_checkbox_{asset_class}"
            if checkbox_key not in st.session_state:
                # Default to checked if it has a weight > 0, otherwise unchecked
                st.session_state[checkbox_key] = (
                    asset_class in st.session_state.portfolio_weights
                    and st.session_state.portfolio_weights[asset_class] > 0
                )

            is_enabled = st.sidebar.checkbox(
                asset_class,
                value=st.session_state[checkbox_key],
                key=checkbox_key,
                disabled=not is_custom_mode,  # Disable checkboxes if preset is selected
            )
            if is_enabled:
                # Get current weight or default to 0
                current_weight = st.session_state.portfolio_weights.get(
                    asset_class, 0.0
                )
                enabled_assets[asset_class] = current_weight
            else:
                # If unchecked, remove from portfolio weights
                if asset_class in st.session_state.portfolio_weights:
                    st.session_state.portfolio_weights[asset_class] = 0.0
                # Also clear the input state
                input_key = f"multi_asset_input_{asset_class}"
                if input_key in st.session_state:
                    st.session_state[input_key] = 0.0

        # If no assets enabled, set default distribution
        if not enabled_assets:
            # Default to US Stocks and Bonds if nothing is selected
            enabled_assets = {"US Stocks": 60.0, "Bonds": 40.0}
            st.session_state.portfolio_weights.update(enabled_assets)
            for asset_class in enabled_assets.keys():
                checkbox_key = f"asset_checkbox_{asset_class}"
                st.session_state[checkbox_key] = True
                input_key = f"multi_asset_input_{asset_class}"
                st.session_state[input_key] = enabled_assets[asset_class]

        # Handle newly enabled assets - give them a default weight if they have 0
        total_enabled_weight = sum(enabled_assets.values())
        newly_enabled = []
        for asset_class, weight in enabled_assets.items():
            if weight == 0.0:
                newly_enabled.append(asset_class)

        # If there are newly enabled assets, distribute remaining weight equally
        if newly_enabled and total_enabled_weight < 100.0:
            remaining_weight = 100.0 - total_enabled_weight
            per_new_asset = (
                remaining_weight / len(newly_enabled) if newly_enabled else 0.0
            )
            for asset_class in newly_enabled:
                enabled_assets[asset_class] = per_new_asset
                st.session_state.portfolio_weights[asset_class] = per_new_asset
                input_key = f"multi_asset_input_{asset_class}"
                st.session_state[input_key] = per_new_asset

        # Note: Crypto assets use Monte Carlo simulation for the full timeline based on available statistics

        # Interactive pie chart with sliders
        if enabled_assets:
            asset_list = list(enabled_assets.keys())

            # Initialize previous weights tracking
            if "prev_multi_asset_slider_values" not in st.session_state:
                st.session_state.prev_multi_asset_slider_values = {}

            # Initialize slider states from current portfolio weights (only on first load or preset change)
            if (
                "multi_asset_sliders_initialized" not in st.session_state
                or preset_changed
            ):
                for asset_class in asset_list:
                    slider_key = f"multi_asset_slider_{asset_class}"
                    st.session_state[slider_key] = enabled_assets.get(asset_class, 0.0)
                    st.session_state.prev_multi_asset_slider_values[
                        asset_class
                    ] = enabled_assets.get(asset_class, 0.0)
                st.session_state.multi_asset_sliders_initialized = True

            # Ensure all current assets have slider state initialized
            for asset_class in asset_list:
                slider_key = f"multi_asset_slider_{asset_class}"
                if slider_key not in st.session_state:
                    default_value = enabled_assets.get(asset_class, 0.0)
                    st.session_state[slider_key] = default_value
                    st.session_state.prev_multi_asset_slider_values[
                        asset_class
                    ] = default_value

            # Check if we have target values from proportional adjustment (from previous render)
            target_weights_key = "multi_asset_target_slider_values"
            if target_weights_key in st.session_state and not preset_changed:
                # Apply target values BEFORE creating widgets
                target_values = st.session_state[target_weights_key]
                for asset_class in asset_list:
                    slider_key = f"multi_asset_slider_{asset_class}"
                    if asset_class in target_values:
                        st.session_state[slider_key] = target_values[asset_class]
                # Clear target values after applying
                del st.session_state[target_weights_key]

            # Read all slider values (disabled if preset is selected, not Custom)
            slider_values = {}
            for asset_class in asset_list:
                slider_key = f"multi_asset_slider_{asset_class}"
                # Use .get() with default to avoid KeyError
                current_value = st.session_state.get(
                    slider_key, enabled_assets.get(asset_class, 0.0)
                )
                slider_value = st.sidebar.slider(
                    f"{asset_class} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=current_value,
                    step=0.5,
                    key=slider_key,
                    disabled=not is_custom_mode,  # Disable if preset is selected
                    help=(
                        f"Drag to adjust {asset_class}. Others adjust proportionally to maintain 100% total."
                        if is_custom_mode
                        else "Select 'Custom' preset to adjust allocations"
                    ),
                )
                slider_values[asset_class] = slider_value

            # Only do proportional adjustment if in custom mode
            if is_custom_mode:
                # Detect which slider changed the most (user interaction)
                max_change = 0
                changed_asset = None
                prev_values = st.session_state.prev_multi_asset_slider_values
                for asset_class in asset_list:
                    prev_value = prev_values.get(asset_class, 0.0)
                    change = abs(slider_values[asset_class] - prev_value)
                    if change > max_change:
                        max_change = change
                        changed_asset = asset_class

                # If one slider changed significantly, calculate proportional adjustments
                if changed_asset and max_change > 0.1:
                    old_value = prev_values.get(changed_asset, 0.0)
                    new_value = slider_values[changed_asset]
                    delta = new_value - old_value

                    # Get other assets
                    other_assets = {
                        k: v for k, v in slider_values.items() if k != changed_asset
                    }

                    if other_assets and abs(delta) > 0.01:
                        # Calculate total of other assets from previous values
                        other_total = sum(
                            prev_values.get(k, 0.0) for k in other_assets.keys()
                        )

                        if other_total > 0.01:
                            # Adjust other assets proportionally
                            target_values = slider_values.copy()
                            for asset in other_assets.keys():
                                prev_other_value = prev_values.get(asset, 0.0)
                                proportion = prev_other_value / other_total
                                adjustment = -delta * proportion
                                target_values[asset] = max(
                                    0.0, min(100.0, prev_other_value + adjustment)
                                )

                            # Store target values for next render
                            st.session_state[target_weights_key] = target_values
                            # Update previous values
                            st.session_state.prev_multi_asset_slider_values = (
                                target_values.copy()
                            )
                            # Trigger rerun to apply adjustments
                            st.rerun()
                        else:
                            # If other assets sum to 0, distribute equally
                            n_others = len(other_assets)
                            if n_others > 0:
                                target_values = slider_values.copy()
                                per_asset_adjustment = -delta / n_others
                                for asset in other_assets.keys():
                                    prev_other_value = prev_values.get(asset, 0.0)
                                    target_values[asset] = max(
                                        0.0,
                                        min(
                                            100.0,
                                            prev_other_value + per_asset_adjustment,
                                        ),
                                    )

                                st.session_state[target_weights_key] = target_values
                                st.session_state.prev_multi_asset_slider_values = (
                                    target_values.copy()
                                )
                                st.rerun()

                # Ensure total is exactly 100% (final normalization)
                total_pct = sum(slider_values.values())
                if abs(total_pct - 100.0) > 0.01 and total_pct > 0.01:
                    # Normalize to exactly 100%
                    normalization_factor = 100.0 / total_pct
                    normalized_values = {
                        k: v * normalization_factor for k, v in slider_values.items()
                    }
                    # Store normalized values for next render
                    st.session_state[target_weights_key] = normalized_values
                    st.session_state.prev_multi_asset_slider_values = (
                        normalized_values.copy()
                    )
                    st.rerun()

                # Update previous values for next comparison
                st.session_state.prev_multi_asset_slider_values = slider_values.copy()

            # Use slider values directly (they should sum to 100% now)
            st.session_state.portfolio_weights = slider_values

            # Show pie chart
            st.sidebar.markdown("**Portfolio Allocation:**")
            fig = self._create_portfolio_pie_chart(slider_values)
            # Remove chart title
            fig.update_layout(title="")
            st.sidebar.plotly_chart(
                fig, use_container_width=True, key="portfolio_pie_chart"
            )

            # Show total percentage
            total_display = sum(slider_values.values())
            if abs(total_display - 100.0) < 0.1:
                st.sidebar.caption("✅ Total: 100%")
            else:
                st.sidebar.caption(f"⚠️ Total: {total_display:.1f}% (adjusting...)")

            # Display summary
            allocation_text = ", ".join(
                [
                    f"{pct:.1f}% {asset}"
                    for asset, pct in st.session_state.portfolio_weights.items()
                    if pct > 0
                ]
            )
            st.sidebar.markdown(f"**Total:** {allocation_text}")
        else:
            st.sidebar.info("Please select at least one asset class")

        # Get tickers and weights from portfolio allocation
        # Initialize with defaults in case portfolio allocation fails
        portfolio_tickers = []
        portfolio_weights = np.array([])
        try:
            portfolio_tickers, portfolio_weights = self._get_tickers_for_assets(
                st.session_state.portfolio_weights
            )
        except Exception:
            # Fallback to defaults if there's an issue
            portfolio_tickers = list(self.config.default_tickers)
            portfolio_weights = np.array(self.config.default_weights)

        # Retirement spending
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
            # Dynamic withdrawal settings in expander
            with st.sidebar.expander("Dynamic Withdrawal Settings", expanded=True):
                # Total annual expense
                total_annual_expense = st.number_input(
                    "Total annual retirement spending ($)",
                    value=self.config.default_annual_spend,
                    step=1000.0,
                    min_value=0.0,
                )

                # Expense category preset selection
                preset_options = list(self.EXPENSE_PRESETS.keys()) + ["Custom"]

                # Track previous preset to detect changes
                if "previous_preset" not in st.session_state:
                    st.session_state.previous_preset = None

                selected_preset = st.selectbox(
                    "Expense category distribution",
                    options=preset_options,
                    index=0,  # Default to "Typical US Household"
                    help="Choose a preset distribution or customize your own",
                )

                # Track previous education level to detect changes
                if "previous_education_level" not in st.session_state:
                    st.session_state.previous_education_level = None

                # Initialize or update category percentages based on preset
                if "category_percentages" not in st.session_state:
                    if selected_preset in self.EXPENSE_PRESETS:
                        st.session_state.category_percentages = self.EXPENSE_PRESETS[
                            selected_preset
                        ].copy()
                    else:
                        # Default: equal distribution for Custom
                        default_pct = 100.0 / len(self.EXPENSE_CATEGORIES)
                        st.session_state.category_percentages = {
                            cat: default_pct for cat in self.EXPENSE_CATEGORIES.keys()
                        }
                    st.session_state.previous_preset = selected_preset
                elif selected_preset != st.session_state.previous_preset:
                    # Update if preset changed (but not if it's Custom - preserve custom values)
                    if selected_preset in self.EXPENSE_PRESETS:
                        st.session_state.category_percentages = self.EXPENSE_PRESETS[
                            selected_preset
                        ].copy()
                    st.session_state.previous_preset = selected_preset

                # Show pie chart in collapsible section
                with st.expander("View/Edit Category Breakdown", expanded=False):
                    percentages = st.session_state.category_percentages
                    fig = self._create_pie_chart(percentages, total_annual_expense)
                    st.plotly_chart(fig, use_container_width=True)

                    # Sliders for customization (only show if Custom or user wants to adjust)
                    if selected_preset == "Custom" or st.checkbox(
                        "Customize percentages", value=False
                    ):
                        st.markdown("**Adjust percentages:**")
                        new_percentages = {}
                        for category in self.EXPENSE_CATEGORIES.keys():
                            new_percentages[category] = st.slider(
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
                                k: (v / total_pct) * 100.0
                                for k, v in new_percentages.items()
                            }
                            st.session_state.category_percentages = normalized
                        else:
                            st.session_state.category_percentages = new_percentages

                # CPI adjustment toggle
                use_cpi = st.checkbox(
                    "Use CPI-based inflation adjustment",
                    value=True,
                    help="Adjust spending based on historical CPI inflation data",
                )

                # Optional: Wage and education (in sub-expander)
                current_wage = 0.0
                education_level = ""
                with st.expander("Income & Education (Optional)", expanded=False):
                    st.caption("For future wage growth calculations")
                    current_wage = st.number_input(
                        "Current annual wage/salary ($)",
                        value=0.0,
                        step=1000.0,
                        min_value=0.0,
                        help="Your current annual income (optional)",
                    )

                    # Get current education level from session state if available
                    current_education_index = 0
                    if "current_education_level" in st.session_state:
                        current_ed = st.session_state.current_education_level
                        if current_ed in self.EDUCATION_LEVELS:
                            current_education_index = (
                                self.EDUCATION_LEVELS.index(current_ed) + 1
                            )

                    education_level = st.selectbox(
                        "Education level",
                        options=[""] + self.EDUCATION_LEVELS,
                        index=current_education_index,
                        help="Your education level (optional). Selecting an education level will adjust expense category distributions.",
                    )

                    # Apply education-based expense preset when education level is selected/changed
                    if (
                        education_level
                        and education_level in self.EDUCATION_EXPENSE_PRESETS
                    ):
                        # Check if education level changed
                        if st.session_state.previous_education_level != education_level:
                            # Apply education-based preset
                            st.session_state.category_percentages = (
                                self.EDUCATION_EXPENSE_PRESETS[education_level].copy()
                            )
                            st.session_state.previous_education_level = education_level
                            # Show info message
                            st.info(
                                f"📚 Applied education-based spending adjustments for {education_level}"
                            )
                        elif (
                            st.session_state.previous_education_level is None
                            and education_level
                        ):
                            # First time selecting education level
                            st.session_state.category_percentages = (
                                self.EDUCATION_EXPENSE_PRESETS[education_level].copy()
                            )
                            st.session_state.previous_education_level = education_level
                            st.info(
                                f"📚 Applied education-based spending adjustments for {education_level}"
                            )
                    elif (
                        st.session_state.previous_education_level is not None
                        and not education_level
                    ):
                        # Education level was cleared, revert to previous preset
                        if st.session_state.previous_preset in self.EXPENSE_PRESETS:
                            st.session_state.category_percentages = (
                                self.EXPENSE_PRESETS[
                                    st.session_state.previous_preset
                                ].copy()
                            )
                        st.session_state.previous_education_level = None

                    # Store current education level in session state
                    st.session_state.current_education_level = (
                        education_level if education_level else None
                    )

                    # Wage-based savings option (only if education level is selected)
                    if education_level:
                        st.markdown("---")
                        st.markdown("**Wage-Based Savings**")
                        use_wage_based_savings = st.checkbox(
                            "Use wage-based savings",
                            value=st.session_state.use_wage_based_savings,
                            help="Calculate annual savings as a percentage of your wage, which grows over time based on your education level",
                        )
                        st.session_state.use_wage_based_savings = use_wage_based_savings

                        if use_wage_based_savings:
                            savings_rate = (
                                st.slider(
                                    "Savings rate (% of income)",
                                    min_value=5.0,
                                    max_value=50.0,
                                    value=st.session_state.savings_rate * 100.0,
                                    step=1.0,
                                    help="Percentage of your annual income to save each year",
                                )
                                / 100.0
                            )  # Convert to decimal
                            st.session_state.savings_rate = savings_rate

                            # Show estimated first year savings if we have wage data
                            if current_wage > 0:
                                first_year_savings = current_wage * savings_rate
                                savings_pct = savings_rate * 100
                                st.info(
                                    f"💵 **First year savings:** ${first_year_savings:,.0f}\n\n"
                                    f"({savings_pct:.0f}% of ${current_wage:,.0f} annual wage)"
                                )
                            else:
                                # Try to estimate from education level
                                from app.services import DataService

                                data_service = DataService()
                                if education_level:
                                    weekly_wage = (
                                        data_service.get_income_for_education_level(
                                            education_level
                                        )
                                    )
                                    if weekly_wage:
                                        annual_wage = data_service.get_annual_wage(
                                            weekly_wage
                                        )
                                        first_year_savings = annual_wage * savings_rate
                                        savings_pct = savings_rate * 100
                                        st.info(
                                            f"💵 **Estimated first year savings:** ${first_year_savings:,.0f}\n\n"
                                            f"({savings_pct:.0f}% of ${annual_wage:,.0f} estimated annual wage)"
                                        )
                        else:
                            savings_rate = None
                    else:
                        # Education level not selected, disable wage-based savings
                        use_wage_based_savings = False
                        savings_rate = None

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
                    use_cpi_adjustment=use_cpi,
                )

                # Set wage/education if provided
                if withdrawal_params and (current_wage > 0 or education_level):
                    withdrawal_params.current_wage = (
                        current_wage if current_wage > 0 else None
                    )
                    withdrawal_params.education_level = (
                        education_level if education_level else None
                    )

        else:
            # Traditional fixed annual spending
            annual_spend = st.sidebar.number_input(
                "Annual retirement spending ($)",
                value=self.config.default_annual_spend,
                step=1000.0,
            )

        # Show annual contribution input if NOT using wage-based savings
        if not use_wage_based_savings:
            annual_contrib = st.sidebar.number_input(
                "Annual savings before retirement ($)",
                value=self.config.default_annual_contrib,
                step=1000.0,
                help="Fixed annual contribution amount (ignored if wage-based savings is enabled)",
            )
        else:
            # Set a placeholder value, actual contributions will be calculated from wage
            annual_contrib = 0.0

        # ===== ADVANCED SETTINGS (In Expander) =====
        with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
            # Data settings
            freq = st.selectbox(
                "Frequency",
                options=["monthly", "daily"],
                index=0 if self.config.default_frequency == "monthly" else 1,
            )

            pacing = "pro-rata"
            if freq == "daily":
                pacing = st.selectbox(
                    "Flow pacing (daily only)",
                    options=["pro-rata", "monthly-boundary"],
                    index=0,
                    help="Apply contributions/spending pro-rata each trading day or only on monthly boundaries.",
                )

            # Date range
            start = st.text_input(
                "Data start (YYYY-MM-DD)", value=self.config.default_start_date
            )
            end = st.text_input(
                "Data end (YYYY-MM-DD)", value=self.config.default_end_date
            )

            # Manual ticker/weight override
            use_custom_tickers = st.checkbox(
                "Use custom tickers/weights",
                value=False,
                help="Override portfolio allocation with manual ticker/weight input",
            )

            if use_custom_tickers:
                st.warning(
                    "⚠️ Custom tickers/weights will override the portfolio allocation above."
                )
                tickers_input = st.text_input(
                    "Tickers (comma)",
                    value=",".join(self.config.default_tickers),
                    help="Use ^GSPC,^TNX for longer history (1950s+) or SPY,AGG for ETF data (2003+)",
                )
                weights_input = st.text_input(
                    "Weights for tickers (comma)",
                    value=",".join(map(str, self.config.default_weights)),
                )
            else:
                # Use portfolio allocation from main sidebar
                tickers_input = (
                    ",".join(portfolio_tickers)
                    if portfolio_tickers
                    else ",".join(self.config.default_tickers)
                )
                weights_input = (
                    ",".join(map(str, portfolio_weights))
                    if len(portfolio_weights) > 0
                    else ",".join(map(str, self.config.default_weights))
                )

            # Simulation settings
            st.markdown("**Simulation Settings**")
            n_paths = st.number_input(
                "MC paths", value=self.config.default_mc_paths, min_value=100, step=100
            )
            seed = st.number_input(
                "Random seed",
                value=self.config.default_seed,
                step=1,
                help="Change for different Monte Carlo results",
            )

            # Inflation (only if not using CPI)
            if not use_dynamic_withdrawal:
                inflation = (
                    st.number_input(
                        "Inflation (%/yr)",
                        value=self.config.default_inflation * 100,
                        step=0.1,
                    )
                    / 100.0
                )
            else:
                # Use default inflation for non-CPI calculations
                inflation = self.config.default_inflation

        # Parse inputs (from advanced settings)
        ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        weights = np.array([float(x.strip()) for x in weights_input.split(",")])

        # Get current year for wage projections
        from datetime import datetime

        current_year = datetime.now().year

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
            "use_wage_based_savings": use_wage_based_savings,
            "savings_rate": savings_rate,
            "education_level": education_level if education_level else None,
            "current_year": current_year,
        }

    def _get_tickers_for_assets(
        self, asset_weights: Dict[str, float]
    ) -> Tuple[list[str], np.ndarray]:
        """Map asset class names to ticker symbols and return tickers with weights.

        Args:
            asset_weights: Dictionary mapping asset class names to percentages

        Returns:
            Tuple of (ticker_list, weights_array)
        """
        tickers = []
        weights = []

        # Filter out zero weights and sort by weight (descending)
        filtered_weights = {
            k: v for k, v in asset_weights.items() if v > 0 and k in self.ASSET_CLASSES
        }

        for asset_class, weight in filtered_weights.items():
            # Get ticker list for this asset class
            ticker_options = self.ASSET_CLASSES[asset_class]

            # Prefer index tickers (starting with ^) for longer history
            # Fall back to ETF tickers if needed
            selected_ticker = None
            for ticker in ticker_options:
                # Prefer index tickers (^GSPC, ^TNX, etc.) for longer history
                if ticker.startswith("^"):
                    selected_ticker = ticker
                    break

            # If no index ticker found, use first ETF ticker
            if selected_ticker is None and ticker_options:
                selected_ticker = ticker_options[0]

            if selected_ticker:
                tickers.append(selected_ticker)
                weights.append(weight)

        # Normalize weights to sum to 100
        if weights:
            weights_array = np.array(weights)
            weights_array = (weights_array / weights_array.sum()) * 100.0
        else:
            weights_array = np.array([])

        # Note: Crypto assets use Monte Carlo simulation for the full timeline based on available statistics

        return tickers, weights_array

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
                    textposition="auto",  # Auto-position labels inside/outside based on space
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=hover_text,
                    pull=[0.05] * len(labels),  # Slight pull for better visibility
                )
            ]
        )

        fig.update_layout(
            title="Expense Category Distribution",
            showlegend=False,
            height=450,
            margin=dict(l=60, r=60, t=50, b=50),
            legend=dict(visible=False),
        )

        return fig

    def _create_portfolio_pie_chart(self, asset_weights: Dict[str, float]) -> go.Figure:
        """Create an interactive pie chart for portfolio allocation.

        Args:
            asset_weights: Dictionary mapping asset class names to percentages
        """
        # Filter out zero weights
        filtered_weights = {k: v for k, v in asset_weights.items() if v > 0}
        if not filtered_weights:
            # Return empty chart if no weights
            fig = go.Figure()
            fig.update_layout(title="Portfolio Allocation", height=400)
            return fig

        labels = list(filtered_weights.keys())
        values = [filtered_weights[label] for label in labels]

        # Create hover text with percentage
        hover_text = [f"{label}<br>{pct:.1f}%" for label, pct in zip(labels, values)]

        # Color scheme for different asset classes
        color_map = {
            "US Stocks": "#2ecc71",  # Green
            "International Stocks": "#3498db",  # Blue
            "Bonds": "#9b59b6",  # Purple
            "Cash": "#e74c3c",  # Red
            "Crypto": "#f39c12",  # Orange/Gold
            "Real Estate": "#1abc9c",  # Teal
            "Commodities": "#e67e22",  # Dark orange
        }
        colors = [color_map.get(label, "#95a5a6") for label in labels]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    textinfo="label+percent",
                    textposition="auto",  # Auto-position labels inside/outside based on space
                    hovertemplate="%{hovertext}<br><extra>Click to adjust</extra>",
                    hovertext=hover_text,
                    pull=[0.05] * len(labels),  # Slight pull for better visibility
                    marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                )
            ]
        )

        fig.update_layout(
            title="Portfolio Allocation (Click slices to adjust)",
            showlegend=False,
            height=450,
            margin=dict(l=60, r=60, t=60, b=50),
            hovermode="closest",
            legend=dict(visible=False),
        )

        # Add click event support - make slices more interactive
        fig.update_traces(
            marker_line_width=2,
            marker_line_color="white",
        )

        return fig
