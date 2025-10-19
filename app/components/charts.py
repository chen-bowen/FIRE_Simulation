"""Chart components for visualization."""

import streamlit as st
import plotly.graph_objs as go
import numpy as np
from app.schemas import SimulationResult


class ChartComponent:
    """Charts for the retirement planner."""

    def plot_simulation_paths(self, result: SimulationResult, title: str) -> None:
        """Plot simulation paths with percentile bands and sample paths."""
        x = np.arange(result.horizon_periods) / result.periods_per_year

        fig = go.Figure()

        # Add percentile bands
        fig.add_trace(go.Scatter(x=x, y=result.p90_path, name="P90", line=dict(color="#9ecae1"), fill=None))
        fig.add_trace(go.Scatter(x=x, y=result.median_path, name="Median", line=dict(color="#3182bd", width=2)))
        fig.add_trace(
            go.Scatter(x=x, y=result.p10_path, name="P10", line=dict(color="#9ecae1"), fill="tonexty", fillcolor="rgba(158, 202, 225, 0.3)")
        )

        # Add sample individual paths (fewer for Monte Carlo, more for Historical)
        if hasattr(result, "sample_paths") and result.sample_paths is not None:
            # Show fewer paths for Monte Carlo (smoother, more theoretical)
            # Show more paths for Historical (real market data with variation)
            if "Monte Carlo" in title:
                n_sample_paths = min(10, len(result.sample_paths))  # Fewer MC paths
            else:
                n_sample_paths = min(30, len(result.sample_paths))  # More historical paths

            for i in range(n_sample_paths):
                # Different styling for Monte Carlo vs Historical
                if "Monte Carlo" in title:
                    # More subtle for Monte Carlo (theoretical, smooth)
                    line_color = "rgba(128,128,128,0.2)"
                    line_width = 0.5
                else:
                    # More visible for Historical (real market data)
                    line_color = "rgba(128,128,128,0.4)"
                    line_width = 1

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=result.sample_paths[i],
                        name=f"Sample {i+1}" if i < 3 else None,  # Only show legend for first 3
                        line=dict(color=line_color, width=line_width),
                        showlegend=i < 3,
                    )
                )

        fig.update_layout(title=title, xaxis_title="Years", yaxis_title="Portfolio Value ($)", hovermode="x unified", showlegend=True)

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
        percentile_values = [np.percentile(result.terminal_balances, p) for p in percentiles]

        # Calculate y positions to avoid overlap
        max_count = max(
            [
                len(
                    [
                        x
                        for x in result.terminal_balances
                        if abs(x - val) < (result.terminal_balances.max() - result.terminal_balances.min()) / n_bins
                    ]
                )
                for val in percentile_values
            ]
        )
        y_positions = [0.9, 0.8, 0.7, 0.6, 0.5]  # Staggered y positions

        for i, (p, color, label, value, y_pos) in enumerate(zip(percentiles, colors, labels, percentile_values, y_positions)):
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

    def plot_comparison_chart(self, historical_result: SimulationResult, mc_result: SimulationResult) -> None:
        """Plot comparison between historical and Monte Carlo results."""
        x_hist = np.arange(historical_result.horizon_periods) / historical_result.periods_per_year
        x_mc = np.arange(mc_result.horizon_periods) / mc_result.periods_per_year

        fig = go.Figure()

        # Historical paths
        fig.add_trace(
            go.Scatter(x=x_hist, y=historical_result.median_path, name="Historical Median", line=dict(color="#3182bd", dash="solid"))
        )
        fig.add_trace(go.Scatter(x=x_hist, y=historical_result.p90_path, name="Historical P90", line=dict(color="#9ecae1", dash="dash")))
        fig.add_trace(go.Scatter(x=x_hist, y=historical_result.p10_path, name="Historical P10", line=dict(color="#9ecae1", dash="dash")))

        # Monte Carlo paths
        fig.add_trace(go.Scatter(x=x_mc, y=mc_result.median_path, name="Monte Carlo Median", line=dict(color="#31a354", dash="solid")))
        fig.add_trace(go.Scatter(x=x_mc, y=mc_result.p90_path, name="Monte Carlo P90", line=dict(color="#c7e9c0", dash="dash")))
        fig.add_trace(go.Scatter(x=x_mc, y=mc_result.p10_path, name="Monte Carlo P10", line=dict(color="#c7e9c0", dash="dash")))

        fig.update_layout(
            title="Historical vs Monte Carlo Comparison",
            xaxis_title="Years",
            yaxis_title="Portfolio Value ($)",
            hovermode="x unified",
            showlegend=True,
        )

        st.plotly_chart(fig, use_container_width=True)
