"""
trend.py — Sociopath-it Visualization Module
--------------------------------------------
Flexible line/bar trend visualizer with semantic or continuous color modes.

Features:
- Line and bar trend charts
- Event markers (vertical lines)
- Shading between lines or area fill
- Smooth trend lines
- Interactive Plotly version
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
from matplotlib import cm
from ..utils.style import set_style, generate_semantic_palette, apply_titles, get_color, format_tick_labels


def trend(
    df,
    x,
    y,
    group=None,
    kind="line",
    title="Trend Over Time",
    subtitle=None,
    style_mode="viridis",
    figsize=(7, 5),
    marker="o",
    smooth=False,
    annotate=True,
    event_lines=None,
    shade_between=None,
    fill_area=False,
    bar_color_mode="group",  # "group" or "continuous"
):
    """
    Sociopath-it trend line or bar chart with optional smoothing, shading, and events.
    """
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # --- Color palette handling ---
    if group:
        # Filter out groups with no meaningful data (all NaN, all zeros, or negligible values)
        def has_meaningful_data(group_df, col):
            vals = group_df[col].dropna()
            if vals.empty:
                return False
            # Check if all values are effectively zero (within numerical precision)
            if (vals.abs() < 1e-10).all():
                return False
            # Check if standard deviation is essentially zero (no variation)
            if len(vals) > 1 and vals.std() < 1e-10:
                # But keep if the constant value is meaningful
                return vals.abs().mean() > 1e-10
            return True

        unique_groups = [g for g in df[group].unique() if has_meaningful_data(df[df[group] == g], y)]
        thirds = max(1, len(unique_groups) // 3)
        gdict = {
            "positive": list(unique_groups[:thirds]),
            "neutral": list(unique_groups[thirds:2*thirds]),
            "negative": list(unique_groups[2*thirds:])
        }
        palette = generate_semantic_palette(gdict, mode=style_mode)
    else:
        cmap = cm.get_cmap("viridis")
        palette = {"default": cmap(0.6)}

    # --- Plot types ---
    if kind == "bar":
        # Determine color strategy
        if bar_color_mode == "continuous" and not group:
            # Continuous colormap based on y values
            norm = plt.Normalize(df[y].min(), df[y].max())
            colors = cm.viridis(norm(df[y].values))
            sns.barplot(data=df, x=x, y=y, palette=colors, ax=ax)
        elif group:
            sns.barplot(data=df, x=x, y=y, hue=group, palette=palette, ax=ax)
        else:
            sns.barplot(data=df, x=x, y=y, color=palette["default"], ax=ax)

    else:  # Line plot
        group_data = {}
        label_positions = []  # Store label positions for collision detection
        if group:
            for g, dfg in df.groupby(group):
                color = palette.get(g, "grey")
                dfg = dfg.sort_values(x).reset_index(drop=True)
                xs, ys_raw = dfg[x].values, dfg[y].values
                ys = gaussian_filter1d(ys_raw, sigma=1) if smooth else ys_raw
                group_data[g] = (xs, ys)

                if fill_area:
                    ax.fill_between(xs, 0, ys, color=color, alpha=0.25)
                ax.plot(xs, ys, marker=marker, label=str(g), color=color, lw=2, zorder=3)

                if annotate:
                    last_val = ys[-1]
                    label_positions.append((g, xs[-1], last_val, color))

            # Apply dynamic offset to prevent overlapping labels
            if annotate and label_positions:
                # Sort by y-value to detect overlaps
                label_positions.sort(key=lambda p: p[2])
                adjusted_positions = []
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                min_spacing = y_range * 0.04  # Minimum 4% of y-range between labels

                for i, (g, x_pos, y_pos, color) in enumerate(label_positions):
                    adjusted_y = y_pos
                    # Check for collision with previous labels
                    for prev_y in adjusted_positions:
                        if abs(adjusted_y - prev_y) < min_spacing:
                            # Shift label up or down to avoid overlap
                            if adjusted_y < prev_y:
                                adjusted_y = prev_y - min_spacing
                            else:
                                adjusted_y = prev_y + min_spacing
                    adjusted_positions.append(adjusted_y)

                    # Draw the label at adjusted position
                    ax.text(x_pos, adjusted_y, f"{y_pos:.2f}",
                            ha="left", va="center", fontsize=10, color="black", weight="bold",
                            bbox=dict(facecolor="white", edgecolor='#333333', linewidth=1.5,
                                      boxstyle="round,pad=0.3", alpha=0.95))
        else:
            df_sorted = df.sort_values(x).reset_index(drop=True)
            xs, ys = df_sorted[x].values, df_sorted[y].values
            if fill_area:
                ax.fill_between(xs, 0, ys, color=palette["default"], alpha=0.25)
            ax.plot(xs, ys, color=palette["default"], lw=2, marker=marker)

        # --- Optional shading between groups ---
        if shade_between and len(shade_between) == 2:
            g1, g2 = shade_between
            if g1 in group_data and g2 in group_data:
                xs1, ys1 = group_data[g1]
                xs2, ys2 = group_data[g2]
                if np.array_equal(xs1, xs2):
                    ax.fill_between(xs1, ys1, ys2, color="grey", alpha=0.15)

    # --- Event lines ---
    if event_lines:
        event_color = get_color('warning', style_mode)
        for x_val, label in event_lines.items():
            ax.axvline(x=x_val, color=event_color, linestyle="--", lw=1.5, alpha=0.7, zorder=2)
            ax.text(x_val, ax.get_ylim()[1] * 0.95, label,
                    rotation=90, va="top", ha="right", fontsize=9, color="black", weight="bold",
                    bbox=dict(facecolor="white", edgecolor='#333333', linewidth=1.5,
                              boxstyle="round,pad=0.3", alpha=0.95))

    # --- Widen y-axis window to avoid misleading narrow ranges ---
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Add 15% padding on each side for better context
    padding = y_range * 0.15
    ax.set_ylim(y_min - padding, y_max + padding)

    # For proportions/percentages (values between 0-1), ensure we show meaningful context
    if y_max <= 1.0 and y_min >= 0:
        # If the range is very narrow (less than 20% of full scale), widen to at least 30%
        if y_range < 0.2:
            center = (y_min + y_max) / 2
            ax.set_ylim(max(0, center - 0.15), min(1.0, center + 0.15))

    # --- Styling ---
    ax.set_xlabel(x.replace("_", " ").title(), fontsize=12, weight="bold", color="black")
    ax.set_ylabel(y.replace("_", " ").title(), fontsize=12, weight="bold", color="black")
    ax.grid(axis="y", linestyle=":", color="grey", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if group:
        legend = ax.legend(
            title=group.replace("_", " ").title(),
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="grey",
            fontsize=12,
            title_fontsize=13,
        )
        legend.get_title().set_fontweight("bold")
        legend.get_frame().set_linewidth(1.5)
        legend.get_frame().set_alpha(0.95)

    # --- Title layout ---
    apply_titles(fig, title, subtitle)

    # Format tick labels: bold and angled
    format_tick_labels(ax)

    # Adjust layout based on whether legend is present
    if group:
        fig.tight_layout(rect=(0, 0, 0.85, 0.9 if subtitle else 0.94))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    return fig, ax


def trend_interactive(
    df,
    x,
    y,
    group=None,
    kind="line",
    title="Trend Over Time",
    subtitle=None,
    style_mode="viridis",
    smooth=False,
    event_lines=None,
    fill_area=False,
):
    """Interactive Plotly version of trend chart."""
    set_style(style_mode)

    # Color palette
    if group:
        # Filter out groups with no meaningful data
        def has_meaningful_data(group_df, col):
            vals = group_df[col].dropna()
            if vals.empty:
                return False
            if (vals.abs() < 1e-10).all():
                return False
            if len(vals) > 1 and vals.std() < 1e-10:
                return vals.abs().mean() > 1e-10
            return True

        unique_groups = [g for g in df[group].unique() if has_meaningful_data(df[df[group] == g], y)]
        metric = df[df[group].isin(unique_groups)].groupby(group)[y].mean()
        thirds = max(1, len(metric) // 3)
        gdict = {"positive": list(metric.index[:thirds]),
                 "neutral": list(metric.index[thirds:2*thirds]),
                 "negative": list(metric.index[2*thirds:])}
        palette = generate_semantic_palette(gdict, mode=style_mode)
        # Convert to hex
        palette_hex = {}
        for k, v in palette.items():
            if isinstance(v, tuple):
                palette_hex[k] = f"rgba({int(v[0]*255)},{int(v[1]*255)},{int(v[2]*255)},{v[3] if len(v)>3 else 1})"
            else:
                palette_hex[k] = v
        palette = palette_hex
    else:
        cmap = cm.get_cmap("viridis")
        rgba = cmap(0.6)
        palette = {"default": f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]})"}

    fig = go.Figure()

    if kind == "bar":
        if group:
            for g in df[group].unique():
                dfg = df[df[group] == g]
                fig.add_trace(go.Bar(
                    x=dfg[x], y=dfg[y],
                    name=str(g),
                    marker_color=palette.get(g, "grey")
                ))
        else:
            fig.add_trace(go.Bar(x=df[x], y=df[y], marker_color=palette["default"]))
    else:
        if group:
            for g, dfg in df.groupby(group):
                dfg = dfg.sort_values(x)
                xs = dfg[x]
                ys = dfg[y]
                if smooth:
                    ys = gaussian_filter1d(ys.values, sigma=1)

                mode = "lines+markers"
                fill = "tozeroy" if fill_area else None

                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode=mode,
                    name=str(g),
                    line=dict(color=palette.get(g, "grey"), width=2),
                    fill=fill,
                    fillcolor=palette.get(g, "grey") if fill_area else None,
                ))
        else:
            fig.add_trace(go.Scatter(
                x=df[x], y=df[y],
                mode="lines+markers",
                line=dict(color=palette["default"], width=2),
                fill="tozeroy" if fill_area else None,
            ))

    # Event lines
    if event_lines:
        event_color = get_color('warning', style_mode)
        for x_val, label in event_lines.items():
            fig.add_vline(x=x_val, line=dict(color=event_color, width=2, dash="dash"),
                         annotation_text=label, annotation_position="top right")

    fig.update_layout(
        title=f"<b>{title}</b><br><span style='color:grey'>{subtitle or ''}</span>",
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title(),
        template="plotly_white",
        height=600,
        width=1000,
        hovermode="x unified",
    )

    return fig
