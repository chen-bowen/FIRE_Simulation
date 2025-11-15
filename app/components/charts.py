"""Chart components for visualization.

This module handles all charting and visualization for the retirement planner:
- Portfolio path charts with percentile bands
- Terminal wealth histograms with fine-grained bins
- Comparison charts between different simulation methods
- Interactive Plotly charts with hover information

Key features:
- Adaptive styling for different simulation types
- Non-overlapping percentile labels
- Sample path visualization with appropriate density
- Professional financial chart styling
"""

import numpy as np
import plotly.graph_objs as go
import streamlit as st

from app.schemas import SimulationResult


class ChartComponent:
    """Charts for the retirement planner.

    This class provides all visualization functionality:
    - Portfolio path charts with percentile bands and sample paths
    - Terminal wealth distribution histograms
    - Comparison charts between simulation methods
    - Adaptive styling based on simulation type (Historical vs Monte Carlo)
    """

    def plot_simulation_paths(
        self, result: SimulationResult, title: str, current_age: int = None
    ) -> None:
        """Plot simulation paths with percentile bands and sample paths.

        Args:
            result: Simulation result data
            title: Chart title
            current_age: Current age for age-based x-axis labels (optional)
        """
        years = np.arange(result.horizon_periods) / result.periods_per_year
        # Use age if provided, otherwise use years
        if current_age is not None:
            x = current_age + years
            xaxis_title = "Age"
        else:
            x = years
            xaxis_title = "Years"

        fig = go.Figure()

        # Add shaded confidence band (P10 to P90)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.p90_path,
                name="P90",
                line=dict(color="rgba(158, 202, 225, 0.0)", width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.p10_path,
                name="80% Confidence Band",
                line=dict(color="rgba(158, 202, 225, 0.0)", width=0),
                fill="tonexty",
                fillcolor="rgba(158, 202, 225, 0.2)",
                showlegend=True,
            )
        )

        # Add percentile lines with better styling
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.p90_path,
                name="P90 (90th Percentile)",
                line=dict(color="#3182bd", width=1.5, dash="dash"),
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.median_path,
                name="Median",
                line=dict(color="#08519c", width=3),
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=result.p10_path,
                name="P10 (10th Percentile)",
                line=dict(color="#3182bd", width=1.5, dash="dash"),
                showlegend=True,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            showlegend=True,
        )

        # Format x-axis to show integer values with adaptive tick interval
        if current_age is not None:
            max_age = current_age + (result.horizon_periods / result.periods_per_year)
            age_range = max_age - current_age
            # Adaptive tick interval: 5 years for long ranges, 2-3 for shorter
            if age_range > 30:
                dtick = 10
            elif age_range > 15:
                dtick = 5
            else:
                dtick = 2
            fig.update_xaxes(tickmode="linear", dtick=dtick)

        st.plotly_chart(fig, use_container_width=True)

    def plot_terminal_wealth_histogram(self, result: SimulationResult) -> None:
        """Plot histogram of terminal wealth with fine-grained bins and non-overlapping labels."""
        # Use more bins for finer granularity
        n_bins = min(60, max(40, int(len(result.terminal_balances) ** 0.6)))

        fig = go.Figure(
            data=[
                go.Histogram(
                    x=result.terminal_balances,
                    nbinsx=n_bins,
                    name="Terminal Wealth",
                    opacity=0.7,
                    marker_color="steelblue",
                    marker_line=dict(color="navy", width=0.5),
                )
            ]
        )

        # Calculate percentiles and their positions
        percentiles = [10, 25, 50, 75, 90]
        colors = ["red", "orange", "green", "orange", "red"]
        labels = ["P10", "P25", "P50", "P75", "P90"]

        # Get percentile values
        percentile_values = [
            np.percentile(result.terminal_balances, p) for p in percentiles
        ]

        # Calculate y positions to avoid overlap
        y_positions = [0.9, 0.8, 0.7, 0.6, 0.5]  # Staggered y positions

        for i, (p, color, label, value, y_pos) in enumerate(
            zip(percentiles, colors, labels, percentile_values, y_positions)
        ):
            fig.add_vline(
                x=value,
                line_dash="solid",
                line_color=color,
                line_width=2,
                annotation_text=f"{label}: ${value:,.0f}",
                annotation_position="top",
                annotation_font_size=9,
                annotation_y=y_pos,
                annotation_xshift=10 if i % 2 == 0 else -10,  # Alternate left/right
            )

        # Add summary statistics as text box
        mean_val = np.mean(result.terminal_balances)
        std_val = np.std(result.terminal_balances)
        median_val = np.median(result.terminal_balances)

        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Mean: ${mean_val:,.0f}<br>Median: ${median_val:,.0f}<br>Std: ${std_val:,.0f}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font_size=9,
        )

        fig.update_layout(
            title="Distribution of Terminal Wealth (Fine-Grained)",
            xaxis_title="Terminal Balance ($)",
            yaxis_title="Count",
            showlegend=False,
            hovermode="x unified",
            margin=dict(t=80, b=50, l=50, r=50),
        )

        st.plotly_chart(fig, use_container_width=True)

    def plot_comparison_chart(
        self,
        historical_result: SimulationResult,
        mc_result: SimulationResult,
        current_age: int = None,
    ) -> None:
        """Plot comparison between historical and Monte Carlo results.

        Args:
            historical_result: Historical simulation result
            mc_result: Monte Carlo simulation result
            current_age: Current age for age-based x-axis labels (optional)
        """
        years_hist = (
            np.arange(historical_result.horizon_periods)
            / historical_result.periods_per_year
        )
        years_mc = np.arange(mc_result.horizon_periods) / mc_result.periods_per_year

        # Use age if provided, otherwise use years
        if current_age is not None:
            x_hist = current_age + years_hist
            x_mc = current_age + years_mc
            xaxis_title = "Age"
        else:
            x_hist = years_hist
            x_mc = years_mc
            xaxis_title = "Years"

        fig = go.Figure()

        # Historical paths
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=historical_result.median_path,
                name="Historical Median",
                line=dict(color="#3182bd", dash="solid"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=historical_result.p90_path,
                name="Historical P90",
                line=dict(color="#9ecae1", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=historical_result.p10_path,
                name="Historical P10",
                line=dict(color="#9ecae1", dash="dash"),
            )
        )

        # Monte Carlo paths
        fig.add_trace(
            go.Scatter(
                x=x_mc,
                y=mc_result.median_path,
                name="Monte Carlo Median",
                line=dict(color="#31a354", dash="solid"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_mc,
                y=mc_result.p90_path,
                name="Monte Carlo P90",
                line=dict(color="#c7e9c0", dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_mc,
                y=mc_result.p10_path,
                name="Monte Carlo P10",
                line=dict(color="#c7e9c0", dash="dash"),
            )
        )

        fig.update_layout(
            title="Historical vs Monte Carlo Comparison",
            xaxis_title=xaxis_title,
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            showlegend=True,
        )

        # Format x-axis to show integer values with adaptive tick interval
        if current_age is not None:
            max_age_hist = current_age + (
                historical_result.horizon_periods / historical_result.periods_per_year
            )
            max_age_mc = current_age + (
                mc_result.horizon_periods / mc_result.periods_per_year
            )
            max_age = max(max_age_hist, max_age_mc)
            age_range = max_age - current_age
            # Adaptive tick interval: 5 years for long ranges, 2-3 for shorter
            if age_range > 30:
                dtick = 10
            elif age_range > 15:
                dtick = 5
            else:
                dtick = 2
            fig.update_xaxes(tickmode="linear", dtick=dtick)

        st.plotly_chart(fig, use_container_width=True)
