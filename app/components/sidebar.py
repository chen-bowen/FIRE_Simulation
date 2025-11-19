"""Sidebar component for user inputs."""

from typing import Dict, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.config import get_config
from app.schemas import ExpenseCategory, SavingsRateProfile, WithdrawalParams


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
            "BND",
            "AGG",
            "VBMFX",
        ],  # Use BND (bond ETF) which maps to VBMFX historically; ^TNX removed (yield index, not bond prices)
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
        current_age = st.sidebar.number_input("Current age", value=self.config.default_current_age, step=1)
        retire_age = st.sidebar.number_input("Retirement age", value=self.config.default_retire_age, step=1)
        plan_until_age = st.sidebar.number_input("Plan until age", value=self.config.default_plan_until_age, step=1)

        # Financial inputs
        st.sidebar.subheader("Savings & Spending")
        initial_balance_str = st.sidebar.text_input(
            "Current savings ($)",
            value=str(int(self.config.default_initial_balance)),
            help="Enter amount (e.g., 200000)",
        )
        initial_balance = self._parse_currency_input(initial_balance_str)
        st.sidebar.caption(f"💵 ${initial_balance:,.0f}")

        # Initialize wage-based savings toggle in session state
        if "use_wage_based_savings" not in st.session_state:
            st.session_state.use_wage_based_savings = False
        if "savings_rate" not in st.session_state:
            st.session_state.savings_rate = 0.15  # Default 15%

        # Initialize wage-based savings variables (will be set later)
        use_wage_based_savings = False
        savings_rate = None
        savings_rate_profile = None
        education_level = ""  # Will be set in dynamic withdrawal section
        use_wage_based_spending = False
        replacement_ratio = None
        current_wage = 0.0

        # Annual contribution input (conditional on wage-based savings)
        # This will be set later if wage-based savings is not used
        annual_contrib = self.config.default_annual_contrib

        # Portfolio allocation
        st.sidebar.markdown("---")
        st.sidebar.subheader("Portfolio Allocation")

        # Initialize portfolio weights in session state
        if "portfolio_weights" not in st.session_state:
            # Default to Moderate 60/40
            st.session_state.portfolio_weights = self.PORTFOLIO_PRESETS["Moderate (60/40)"].copy()

        # Mode selection: Preset or Custom (same pattern as retirement spending)
        if "portfolio_mode_radio" not in st.session_state:
            st.session_state.portfolio_mode_radio = "Use Preset"

        mode_selection = st.sidebar.radio(
            "Portfolio mode",
            options=["Use Preset", "Custom"],
            index=0 if st.session_state.portfolio_mode_radio == "Use Preset" else 1,
            help="Choose a preset allocation or customize your own",
            key="portfolio_mode_radio",
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
                key="preset_selectbox",
            )

            # Track if preset changed to sync slider
            if selected_preset != st.session_state.previous_portfolio_preset:
                if selected_preset in self.PORTFOLIO_PRESETS:
                    st.session_state.portfolio_weights = self.PORTFOLIO_PRESETS[selected_preset].copy()
                    preset_changed = True
                st.session_state.previous_portfolio_preset = selected_preset

            st.sidebar.caption(f"📋 Using preset: **{selected_preset}**")
        else:
            # Custom mode
            st.sidebar.info("🎨 **Custom Mode**: Adjust asset classes and allocations below")
            # Reset previous preset when switching to custom
            if st.session_state.previous_portfolio_preset is not None:
                st.session_state.previous_portfolio_preset = None

        # Multi-Asset mode
        # In preset mode, use portfolio weights directly without checkboxes
        if not is_custom_mode:
            # Use current portfolio weights from preset
            enabled_assets = {k: v for k, v in st.session_state.portfolio_weights.items() if v > 0}
        else:
            # Custom mode: Checkboxes to enable/disable asset classes
            st.sidebar.markdown("**Select asset classes:**")
            enabled_assets = {}
            for asset_class in self.ASSET_CLASSES.keys():
                # Initialize checkbox state if not present
                checkbox_key = f"asset_checkbox_{asset_class}"
                if checkbox_key not in st.session_state:
                    # Default to checked if it has a weight > 0, otherwise unchecked
                    st.session_state[checkbox_key] = (
                        asset_class in st.session_state.portfolio_weights and st.session_state.portfolio_weights[asset_class] > 0
                    )

                # Show checkboxes in Custom mode
                is_enabled = st.sidebar.checkbox(
                    asset_class,
                    key=checkbox_key,
                )
                if is_enabled:
                    # Get current weight or default to 0
                    current_weight = st.session_state.portfolio_weights.get(asset_class, 0.0)
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
            per_new_asset = remaining_weight / len(newly_enabled) if newly_enabled else 0.0
            for asset_class in newly_enabled:
                enabled_assets[asset_class] = per_new_asset
                st.session_state.portfolio_weights[asset_class] = per_new_asset
                input_key = f"multi_asset_input_{asset_class}"
                st.session_state[input_key] = per_new_asset

        # Note: Crypto assets use Monte Carlo simulation for the full timeline based on available statistics

        # Handle sliders differently based on mode
        if is_custom_mode and enabled_assets:
            # Custom mode: Show sliders and handle interactions
            asset_list = list(enabled_assets.keys())

            # Initialize previous weights tracking
            if "prev_multi_asset_slider_values" not in st.session_state:
                st.session_state.prev_multi_asset_slider_values = {}

            # Initialize slider states from current portfolio weights (only on first load or preset change)
            if "multi_asset_sliders_initialized" not in st.session_state or preset_changed:
                for asset_class in asset_list:
                    slider_key = f"multi_asset_slider_{asset_class}"
                    st.session_state[slider_key] = enabled_assets.get(asset_class, 0.0)
                    st.session_state.prev_multi_asset_slider_values[asset_class] = enabled_assets.get(asset_class, 0.0)
                st.session_state.multi_asset_sliders_initialized = True

            # Ensure all current assets have slider state initialized
            for asset_class in asset_list:
                slider_key = f"multi_asset_slider_{asset_class}"
                if slider_key not in st.session_state:
                    default_value = enabled_assets.get(asset_class, 0.0)
                    st.session_state[slider_key] = default_value
                    st.session_state.prev_multi_asset_slider_values[asset_class] = default_value

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

            # Read all slider values
            slider_values = {}
            for asset_class in asset_list:
                slider_key = f"multi_asset_slider_{asset_class}"
                slider_value = st.sidebar.slider(
                    f"{asset_class} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    key=slider_key,
                    help=f"Drag to adjust {asset_class}. Others adjust proportionally to maintain 100% total.",
                )
                slider_values[asset_class] = slider_value

            # Do proportional adjustment
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
                other_assets = {k: v for k, v in slider_values.items() if k != changed_asset}

                if other_assets and abs(delta) > 0.01:
                    # Calculate total of other assets from previous values
                    other_total = sum(prev_values.get(k, 0.0) for k in other_assets.keys())

                    if other_total > 0.01:
                        # Adjust other assets proportionally
                        target_values = slider_values.copy()
                        for asset in other_assets.keys():
                            prev_other_value = prev_values.get(asset, 0.0)
                            proportion = prev_other_value / other_total
                            adjustment = -delta * proportion
                            target_values[asset] = max(0.0, min(100.0, prev_other_value + adjustment))

                        # Store target values for next render
                        st.session_state[target_weights_key] = target_values
                        # Update previous values
                        st.session_state.prev_multi_asset_slider_values = target_values.copy()
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
                            st.session_state.prev_multi_asset_slider_values = target_values.copy()
                            st.rerun()

            # Ensure total is exactly 100% (final normalization)
            total_pct = sum(slider_values.values())
            if abs(total_pct - 100.0) > 0.01 and total_pct > 0.01:
                # Normalize to exactly 100%
                normalization_factor = 100.0 / total_pct
                normalized_values = {k: v * normalization_factor for k, v in slider_values.items()}
                # Store normalized values for next render
                st.session_state[target_weights_key] = normalized_values
                st.session_state.prev_multi_asset_slider_values = normalized_values.copy()
                st.rerun()

            # Update previous values for next comparison
            st.session_state.prev_multi_asset_slider_values = slider_values.copy()

            # Use slider values directly (they should sum to 100% now)
            st.session_state.portfolio_weights = slider_values

            # Show total percentage
            total_display = sum(slider_values.values())
            if abs(total_display - 100.0) < 0.1:
                st.sidebar.caption("✅ Total: 100%")
            else:
                st.sidebar.caption(f"⚠️ Total: {total_display:.1f}% (adjusting...)")
        elif not is_custom_mode:
            # Preset mode: Use portfolio weights directly, no sliders
            st.session_state.portfolio_weights = enabled_assets
        else:
            st.sidebar.info("Please select at least one asset class")

        # Get tickers and weights from portfolio allocation
        # Initialize with defaults in case portfolio allocation fails
        portfolio_tickers = []
        portfolio_weights = np.array([])
        try:
            portfolio_tickers, portfolio_weights = self._get_tickers_for_assets(st.session_state.portfolio_weights)
        except Exception:
            # Fallback to defaults if there's an issue
            portfolio_tickers = list(self.config.default_tickers)
            portfolio_weights = np.array(self.config.default_weights)

        # Retirement spending
        st.sidebar.markdown("---")
        st.sidebar.subheader("Retirement Spending")

        # Initialize session state for radio button
        if "retirement_spending_mode_radio" not in st.session_state:
            st.session_state.retirement_spending_mode_radio = "Simple amount"

        spending_mode = st.sidebar.radio(
            "Planning style",
            options=["Simple amount", "Detailed plan"],
            index=0 if st.session_state.retirement_spending_mode_radio == "Simple amount" else 1,
            help="Start with a single spending number or build a CPI-aware plan with categories.",
            key="retirement_spending_mode_radio",
        )
        st.session_state.retirement_spending_mode = spending_mode
        use_dynamic_withdrawal = spending_mode == "Detailed plan"

        annual_spend = None
        withdrawal_params = None

        if use_dynamic_withdrawal:
            # In Detailed plan mode, retirement spending is calculated from wage and replacement ratio
            # Initialize total_annual_expense - will be calculated from wage/replacement ratio
            total_annual_expense = self.config.default_annual_spend  # Default fallback

            # Initialize category percentages if not already set
            if "category_percentages" not in st.session_state:
                default_pct = 100.0 / len(self.EXPENSE_CATEGORIES)
                st.session_state.category_percentages = {cat: default_pct for cat in self.EXPENSE_CATEGORIES.keys()}
            if "previous_preset" not in st.session_state:
                st.session_state.previous_preset = None
            if "previous_education_level" not in st.session_state:
                st.session_state.previous_education_level = None

            # CPI indexing - always on in Detailed plan mode
            use_cpi = True

            # Pre-retirement savings section - directly visible in Detailed plan mode
            st.sidebar.markdown("---")
            st.sidebar.subheader("Pre-retirement Savings")

            # Wage-based savings is always enabled in Detailed plan mode (unless already retired)
            is_already_retired_check = retire_age <= current_age
            use_wage_based_savings = not is_already_retired_check
            st.session_state.use_wage_based_savings = use_wage_based_savings

            if use_wage_based_savings:
                # Savings rate style selection - shown first
                savings_rate_mode = st.sidebar.radio(
                    "Savings rate style",
                    options=["Constant rate", "Age-based profile"],
                    index=(
                        0 if "savings_rate_mode" not in st.session_state else (0 if st.session_state.savings_rate_mode == "constant" else 1)
                    ),
                    key="savings_rate_mode_radio",
                )
                st.session_state.savings_rate_mode = "constant" if savings_rate_mode == "Constant rate" else "profile"

                if savings_rate_mode == "Constant rate":
                    # Initialize slider value in session state if not present
                    if "savings_rate_slider" not in st.session_state:
                        initial_rate = st.session_state.get("savings_rate", 0.15)
                        st.session_state.savings_rate_slider = min(100.0, max(5.0, initial_rate * 100.0))

                    # Get current slider value and ensure it's within bounds
                    current_slider_value = st.session_state.get("savings_rate_slider", 15.0)
                    # Clamp to exact bounds to prevent bouncing
                    clamped_value = max(5.0, min(100.0, current_slider_value))

                    # Use slider with key - Streamlit manages the state automatically
                    savings_rate_pct = st.sidebar.slider(
                        "Savings rate (% of income)",
                        min_value=5.0,
                        max_value=100.0,
                        value=clamped_value,
                        step=1.0,
                        key="savings_rate_slider",
                    )
                    # Show bounds below slider
                    st.sidebar.caption("Select a savings rate profile between 5% and 100%")
                    # Ensure value stays within bounds (handle any edge cases)
                    savings_rate_pct = max(5.0, min(100.0, savings_rate_pct))
                    # Convert to decimal and store
                    savings_rate = savings_rate_pct / 100.0
                    st.session_state.savings_rate = savings_rate
                    savings_rate_profile = None
                else:
                    savings_presets = {
                        "Young Professional": SavingsRateProfile(
                            age_ranges=[(25, 30), (31, 40), (41, 50), (51, 65)],
                            rates=[0.10, 0.20, 0.25, 0.20],
                        ),
                        "Family Focus": SavingsRateProfile(
                            age_ranges=[(25, 35), (36, 45), (46, 55), (56, 65)],
                            rates=[0.08, 0.15, 0.22, 0.18],
                        ),
                        "Empty Nester": SavingsRateProfile(
                            age_ranges=[(25, 40), (41, 50), (51, 65)],
                            rates=[0.12, 0.20, 0.28],
                        ),
                    }

                    preset_choice = st.sidebar.selectbox(
                        "Choose a savings profile",
                        options=["Custom"] + list(savings_presets.keys()),
                        key="savings_profile_select",
                    )

                    if preset_choice != "Custom":
                        savings_rate_profile = savings_presets[preset_choice]
                        st.session_state.savings_rate_profile = savings_rate_profile
                        st.sidebar.markdown("**Profile overview**")
                        for (start_age, end_age), rate in zip(savings_rate_profile.age_ranges, savings_rate_profile.rates):
                            st.sidebar.write(f"- Age {start_age}-{end_age}: {rate*100:.0f}%")

                        st.sidebar.markdown("**Adjust rates:**")
                        edited_rates = []
                        for (start_age, end_age), rate in zip(savings_rate_profile.age_ranges, savings_rate_profile.rates):
                            # Ensure value is within bounds to prevent bouncing
                            current_rate_pct = min(100.0, max(0.0, rate * 100.0))
                            new_rate = (
                                st.sidebar.slider(
                                    f"Age {start_age}-{end_age}",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=current_rate_pct,
                                    step=1.0,
                                )
                                / 100.0
                            )
                            edited_rates.append(new_rate)

                        if edited_rates != savings_rate_profile.rates:
                            savings_rate_profile = SavingsRateProfile(
                                age_ranges=savings_rate_profile.age_ranges,
                                rates=edited_rates,
                            )
                            st.session_state.savings_rate_profile = savings_rate_profile
                    else:
                        st.sidebar.info("Custom profiles coming soon. Using your constant rate for now.")
                        savings_rate = st.session_state.savings_rate if st.session_state.savings_rate else 0.15
                        savings_rate_profile = None

                # Initialize education level default to Master's degree in Detailed plan mode
                if "current_education_level" not in st.session_state or st.session_state.current_education_level is None:
                    st.session_state.current_education_level = "Master's degree"

                # Get current education level index
                current_education_index = 0
                if st.session_state.current_education_level in self.EDUCATION_LEVELS:
                    current_education_index = self.EDUCATION_LEVELS.index(st.session_state.current_education_level)

                # Current wage input - shown after savings rate settings
                current_wage_str = st.sidebar.text_input(
                    "Current annual wage/salary ($)",
                    value="150000",
                    key="current_wage_input",
                    help="Used for wage growth projections and savings calculations.",
                )
                current_wage = self._parse_currency_input(current_wage_str)
                if current_wage > 0:
                    st.sidebar.caption(f"💵 ${current_wage:,.0f}")

                # Education level - defaults to Master's degree
                education_level = st.sidebar.selectbox(
                    "Education level",
                    options=self.EDUCATION_LEVELS,
                    index=current_education_index,
                    key="education_level_select",
                    help="Used for wage projections and category template.",
                )
                st.session_state.current_education_level = education_level

                # Apply education-based category template if changed
                if education_level and education_level in self.EDUCATION_EXPENSE_PRESETS:
                    if st.session_state.previous_education_level != education_level:
                        st.session_state.category_percentages = self.EDUCATION_EXPENSE_PRESETS[education_level].copy()
                        st.session_state.previous_education_level = education_level
                elif st.session_state.previous_education_level is None:
                    st.session_state.previous_education_level = education_level

                # Show first year savings estimate
                if current_wage > 0:
                    first_year_rate = (
                        savings_rate
                        if savings_rate
                        else (savings_rate_profile.get_rate_for_age(current_age) if savings_rate_profile else 0.15)
                    )
                    first_year_savings = current_wage * first_year_rate
                    savings_pct = first_year_rate * 100
                    st.sidebar.info(f"💾 First year savings: ${first_year_savings:,.0f} ({savings_pct:.0f}% of income)")
                else:
                    from app.services import DataService

                    data_service = DataService()
                    weekly_wage = data_service.get_income_for_education_level(education_level)
                    if weekly_wage:
                        annual_wage = data_service.get_annual_wage(weekly_wage)
                        first_year_rate = (
                            savings_rate
                            if savings_rate
                            else (savings_rate_profile.get_rate_for_age(current_age) if savings_rate_profile else 0.15)
                        )
                        first_year_savings = annual_wage * first_year_rate
                        savings_pct = first_year_rate * 100
                        st.sidebar.info(
                            f"💾 Estimated first year savings: ${first_year_savings:,.0f} " f"({savings_pct:.0f}% of ${annual_wage:,.0f})"
                        )

                # Retirement spending based on lifestyle - direct input (no checkbox)
                st.sidebar.markdown("---")
                st.sidebar.markdown("**Retirement Spending Adjustment**")
                replacement_ratio_pct = st.sidebar.slider(
                    "Replacement ratio (%)",
                    min_value=50.0,
                    max_value=100.0,
                    value=st.session_state.get("replacement_ratio_pct", 80.0),
                    step=5.0,
                    help="Set retirement spending as % of pre-retirement spending (income minus savings).",
                )
                st.session_state.replacement_ratio_pct = replacement_ratio_pct
                replacement_ratio = replacement_ratio_pct / 100.0
                st.session_state.replacement_ratio = replacement_ratio
                use_wage_based_spending = True
                st.session_state.use_wage_based_spending = use_wage_based_spending

                # Show estimated retirement spending
                from app.services import DataService
                from datetime import datetime

                data_service = DataService()
                target_retire_age = retire_age if retire_age > current_age else current_age + 30
                weekly_wage_at_retirement = None
                if education_level:
                    weekly_wage_at_retirement = data_service.get_wage_for_age(
                        education_level, current_age, datetime.now().year, target_retire_age
                    )
                if weekly_wage_at_retirement:
                    annual_wage_at_retirement = data_service.get_annual_wage(weekly_wage_at_retirement)
                    if savings_rate_profile:
                        savings_rate_for_retirement = savings_rate_profile.get_rate_for_age(target_retire_age)
                        if savings_rate_for_retirement is None:
                            savings_rate_for_retirement = savings_rate if savings_rate else 0.15
                    else:
                        savings_rate_for_retirement = savings_rate if savings_rate else 0.15

                    pre_retire_spending_est = annual_wage_at_retirement * (1.0 - savings_rate_for_retirement)
                    retirement_spending = pre_retire_spending_est * replacement_ratio
                    # Use calculated retirement spending as total_annual_expense
                    total_annual_expense = retirement_spending
                    st.sidebar.info(
                        f"💵 Estimated retirement spending: \${retirement_spending:,.0f}\n\n"
                        f"{replacement_ratio*100:.0f}% of \${pre_retire_spending_est:,.0f} pre-retirement spending"
                    )
            else:
                # Already retired - no savings, but still allow wage/education for category templates
                savings_rate = None
                savings_rate_profile = None
                use_wage_based_spending = False
                replacement_ratio = None
                st.sidebar.info("ℹ️ You are already retired - no savings contributions will be applied.")

            # Expense category preset selection - moved below Pre-retirement Savings
            st.sidebar.markdown("---")
            st.sidebar.subheader("Spending Categories")

            preset_options = list(self.EXPENSE_PRESETS.keys()) + ["Custom"]
            selected_preset = st.sidebar.selectbox(
                "Category template",
                options=preset_options,
                index=0,
                help="Templates use BLS household data. Choose Custom to start from an even split.",
            )

            if selected_preset != st.session_state.previous_preset:
                if selected_preset in self.EXPENSE_PRESETS:
                    st.session_state.category_percentages = self.EXPENSE_PRESETS[selected_preset].copy()
                else:
                    default_pct = 100.0 / len(self.EXPENSE_CATEGORIES)
                    st.session_state.category_percentages = {cat: default_pct for cat in self.EXPENSE_CATEGORIES.keys()}
                st.session_state.previous_preset = selected_preset

            # Show pie chart directly - always visible
            percentages = st.session_state.category_percentages
            fig = self._create_pie_chart(percentages, total_annual_expense)
            st.sidebar.plotly_chart(fig, use_container_width=True)

            # Fine-tune percentages - show directly or in expander
            st.sidebar.caption("Need tweaks? Adjust sliders below to rebalance categories.")
            customize = st.sidebar.checkbox(
                "Fine-tune percentages",
                value=selected_preset == "Custom",
                key="customize_category_percentages",
            )
            if customize or selected_preset == "Custom":
                st.sidebar.markdown("**Adjust percentages:**")
                new_percentages = {}
                for category in self.EXPENSE_CATEGORIES.keys():
                    new_percentages[category] = st.sidebar.slider(
                        f"{category}",
                        min_value=0.0,
                        max_value=100.0,
                        value=percentages.get(category, 0.0),
                        step=0.1,
                        help=self.EXPENSE_CATEGORIES[category],
                    )

                total_pct = sum(new_percentages.values())
                if total_pct > 0:
                    normalized = {k: (v / total_pct) * 100.0 for k, v in new_percentages.items()}
                    st.session_state.category_percentages = normalized
                else:
                    st.session_state.category_percentages = new_percentages

            expense_categories = [
                ExpenseCategory(name=cat, percentage=st.session_state.category_percentages[cat]) for cat in self.EXPENSE_CATEGORIES.keys()
            ]

            withdrawal_params = WithdrawalParams(
                expense_categories=expense_categories,
                total_annual_expense=total_annual_expense,
                use_cpi_adjustment=use_cpi,
            )

            if withdrawal_params and (current_wage > 0 or education_level):
                withdrawal_params.current_wage = current_wage if current_wage > 0 else None
                withdrawal_params.education_level = education_level if education_level else None

        else:
            annual_spend_str = st.sidebar.text_input(
                "Annual retirement spending ($)",
                value=str(int(self.config.default_annual_spend)),
                key="simple_spending_input",
                help="Use a single number if you don't need CPI-aware categories.",
            )
            annual_spend = self._parse_currency_input(annual_spend_str)
            st.sidebar.caption(f"💵 \${annual_spend:,.0f}/year · \${annual_spend/12:,.0f}/month")

        # Check if already retired
        is_already_retired = retire_age <= current_age

        # Show annual contribution input only in "Simple amount" mode AND NOT using wage-based savings AND NOT already retired
        if not use_dynamic_withdrawal and not use_wage_based_savings and not is_already_retired:
            annual_contrib_str = st.sidebar.text_input(
                "Annual savings before retirement ($)",
                value=str(int(self.config.default_annual_contrib)),
                help="Enter amount (e.g., 10000). Fixed annual contribution amount (ignored if wage-based savings is enabled).",
            )
            try:
                annual_contrib = float(annual_contrib_str.replace(",", "").replace("$", "").strip()) if annual_contrib_str.strip() else 0.0
            except ValueError:
                annual_contrib = 0.0
            st.sidebar.caption(f"💵 ${annual_contrib:,.0f}")
        else:
            # Set a placeholder value - either Detailed plan mode, wage-based savings, or already retired
            annual_contrib = 0.0
            if is_already_retired:
                st.sidebar.info("ℹ️ You are already retired - no savings contributions will be applied.")

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
            start = st.text_input("Data start (YYYY-MM-DD)", value=self.config.default_start_date)
            end = st.text_input("Data end (YYYY-MM-DD)", value=self.config.default_end_date)

            # Manual ticker/weight override
            use_custom_tickers = st.checkbox(
                "Use custom tickers/weights",
                value=False,
                help="Override portfolio allocation with manual ticker/weight input",
            )

            if use_custom_tickers:
                st.warning("⚠️ Custom tickers/weights will override the portfolio allocation above.")
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
                tickers_input = ",".join(portfolio_tickers) if portfolio_tickers else ",".join(self.config.default_tickers)
                weights_input = (
                    ",".join(map(str, portfolio_weights)) if len(portfolio_weights) > 0 else ",".join(map(str, self.config.default_weights))
                )

            # Simulation settings
            st.markdown("**Simulation Settings**")
            n_paths = st.number_input("MC paths", value=self.config.default_mc_paths, min_value=100, step=100)
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
            "savings_rate_profile": savings_rate_profile,
            "education_level": education_level if education_level else None,
            "current_year": current_year,
            "use_wage_based_spending": use_wage_based_spending,
            "replacement_ratio": replacement_ratio,
            # Auto-track pre-retirement spending when wage-based spending is enabled (covers both savings and spending)
            "pre_retire_spending_tracked": use_wage_based_savings or use_wage_based_spending,
        }

    def _parse_currency_input(self, raw_value: Optional[str]) -> float:
        """Convert currency-like text inputs to float."""
        if not raw_value:
            return 0.0
        try:
            clean_value = raw_value.replace(",", "").replace("$", "").strip()
            return float(clean_value) if clean_value else 0.0
        except ValueError:
            return 0.0

    def _get_tickers_for_assets(self, asset_weights: Dict[str, float]) -> Tuple[list[str], np.ndarray]:
        """Map asset class names to ticker symbols and return tickers with weights.

        Args:
            asset_weights: Dictionary mapping asset class names to percentages

        Returns:
            Tuple of (ticker_list, weights_array)
        """
        tickers = []
        weights = []

        # Filter out zero weights and sort by weight (descending)
        filtered_weights = {k: v for k, v in asset_weights.items() if v > 0 and k in self.ASSET_CLASSES}

        for asset_class, weight in filtered_weights.items():
            # Get ticker list for this asset class
            ticker_options = self.ASSET_CLASSES[asset_class]

            # Prefer index tickers (starting with ^) for longer history
            # EXCEPT for Bonds - prefer ETFs over yield indices (^TNX)
            # Fall back to ETF tickers if needed
            selected_ticker = None
            if asset_class == "Bonds":
                # For bonds, use first ticker (BND) - don't prefer ^TNX (yield index)
                if ticker_options:
                    selected_ticker = ticker_options[0]
            else:
                # For other assets, prefer index tickers (^GSPC, etc.) for longer history
                for ticker in ticker_options:
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
        """Validate user inputs.

        Supports both pre-retirement and already-retired scenarios.
        """
        # Check if already retired
        is_already_retired = inputs["retire_age"] <= inputs["current_age"]

        if is_already_retired:
            # For already retired, plan_until_age must be greater than current_age
            if inputs["plan_until_age"] <= inputs["current_age"]:
                st.error("Plan until age must be greater than current age (you are already retired)")
                st.stop()
        else:
            # For pre-retirement, standard validation applies
            if inputs["plan_until_age"] <= inputs["retire_age"]:
                st.error("Plan until age must be greater than retirement age")
                st.stop()

        if len(inputs["tickers"]) == 0:
            st.error("At least one ticker must be provided")
            st.stop()

        if len(inputs["weights"]) != len(inputs["tickers"]):
            st.warning("Number of weights doesn't match number of tickers. Will be adjusted automatically.")

    def _create_pie_chart(self, percentages: Dict[str, float], total_expense: float) -> go.Figure:
        """Create an interactive pie chart for expense categories."""
        labels = list(percentages.keys())
        values = [percentages[label] for label in labels]

        # Calculate dollar amounts for display
        amounts = [(pct / 100.0) * total_expense for pct in values]

        # Create hover text with both percentage and dollar amount
        hover_text = [f"{label}<br>{pct:.1f}% = ${amt:,.0f}/year" for label, pct, amt in zip(labels, values, amounts)]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    textinfo="label+percent",
                    textposition="outside",  # Always place labels outside to prevent truncation
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=hover_text,
                    pull=[0.05] * len(labels),  # Slight pull for better visibility
                    textfont=dict(size=11),  # Slightly smaller font to fit better
                )
            ]
        )

        fig.update_layout(
            title="Expense Category Distribution",
            showlegend=False,
            height=450,
            margin=dict(l=80, r=80, t=50, b=50),  # Increased left/right margins for labels
            legend=dict(visible=False),
        )

        return fig

    def _create_portfolio_pie_chart(self, asset_weights: Dict[str, float], initial_balance: Optional[float] = None) -> go.Figure:
        """Create an interactive pie chart for portfolio allocation.

        Args:
            asset_weights: Dictionary mapping asset class names to percentages
            initial_balance: Optional initial portfolio value to display in center
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

        # Determine if we should use a donut chart (hole) to show initial balance in center
        hole_size = 0.4 if initial_balance is not None else 0.0

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
                    marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
                    hole=hole_size,  # Create donut chart if showing initial balance
                )
            ]
        )

        # Add initial balance annotation in center if provided
        if initial_balance is not None:
            fig.add_annotation(
                text=f"<b>${initial_balance:,.0f}</b><br>Initial Value",
                x=0.5,
                y=0.5,
                font_size=14,
                showarrow=False,
            )

        fig.update_layout(
            title="Portfolio Allocation",
            showlegend=False,
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            hovermode="closest",
            legend=dict(visible=False),
        )
        fig.update_traces(
            marker_line_width=2,
            marker_line_color="white",
        )

        return fig
