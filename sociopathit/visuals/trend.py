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
from ..utils.style import set_style, generate_semantic_palette, apply_titles


def trend(
    df,
    x,
    y,
    group=None,
    kind="line",
    title="Trend Over Time",
    subtitle=None,
    style_mode="viridis",
    figsize=(10, 6),
    marker="o",
    smooth=False,
    annotate=True,
    event_lines=None,
    shade_between=None,
    fill_area=False,
):
    """
    Sociopath-it trend line or bar chart with optional smoothing.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    x, y : str
        Column names for x and y axes.
    group : str, optional
        Column to group lines by.
    kind : str, default "line"
        'line' or 'bar'.
    title, subtitle : str
        Plot titles.
    style_mode : str, default "viridis"
        Style theme.
    figsize : tuple
        Figure size.
    marker : str
        Marker style for line plots.
    smooth : bool
        Apply gaussian smoothing.
    annotate : bool
        Show value annotations.
    event_lines : dict, optional
        Dict of {x_value: label} for vertical event markers.
    shade_between : list, optional
        List of two group names to shade between.
    fill_area : bool
        Fill area under lines.
    """
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Color palette
    if group:
        metric = df.groupby(group)[y].mean()
        thirds = max(1, len(metric) // 3)
        gdict = {"positive": list(metric.index[:thirds]),
                 "neutral": list(metric.index[thirds:2*thirds]),
                 "negative": list(metric.index[2*thirds:])}
        palette = generate_semantic_palette(gdict, mode=style_mode)
    else:
        cmap = cm.get_cmap("viridis")
        palette = {"default": cmap(0.6)}

    if kind == "bar":
        sns.barplot(data=df, x=x, y=y, hue=group, palette=palette, ax=ax)
    else:
        # Store data for shading
        group_data = {}

        if group:
            for g, dfg in df.groupby(group):
                color = palette.get(g, "grey")
                dfg = dfg.sort_values(x).reset_index(drop=True)
                xs = dfg[x].values
                ys_raw = dfg[y].values
                ys = gaussian_filter1d(ys_raw, sigma=1) if smooth else ys_raw

                group_data[g] = (xs, ys)

                # Plot line
                if fill_area:
                    ax.fill_between(xs, 0, ys, color=color, alpha=0.2)
                ax.plot(xs, ys, marker=marker, label=str(g), color=color, lw=2, zorder=3)

                # Annotation with background box to prevent overlap
                if annotate:
                    last_val = ys[-1]
                    ax.text(xs[-1], last_val, f"{last_val:.1f}",
                            ha="left", va="center", fontsize=9, color=color, weight="bold",
                            bbox=dict(facecolor="white", edgecolor=color, linewidth=1.0,
                                     boxstyle="round,pad=0.3", alpha=0.95))
        else:
            df_sorted = df.sort_values(x).reset_index(drop=True)
            xs = df_sorted[x].values
            ys = df_sorted[y].values

            if fill_area:
                ax.fill_between(xs, 0, ys, color=palette["default"], alpha=0.3)
            ax.plot(xs, ys, color=palette["default"], lw=2, marker=marker)

        # Shade between two groups
        if shade_between and len(shade_between) == 2:
            g1, g2 = shade_between
            if g1 in group_data and g2 in group_data:
                xs1, ys1 = group_data[g1]
                xs2, ys2 = group_data[g2]
                # Interpolate to common x values if needed
                if len(xs1) == len(xs2) and np.array_equal(xs1, xs2):
                    ax.fill_between(xs1, ys1, ys2, alpha=0.15, color="grey")

    # Event lines with clear annotations
    if event_lines:
        for x_val, label in event_lines.items():
            ax.axvline(x=x_val, color="red", linestyle="--", lw=1.5, alpha=0.7, zorder=2)
            ax.text(x_val, ax.get_ylim()[1] * 0.95, label,
                   rotation=90, va="top", ha="right", fontsize=9, color="red", weight="bold",
                   bbox=dict(facecolor="white", edgecolor="red", linewidth=1.0,
                            boxstyle="round,pad=0.3", alpha=0.9))

    ax.set_xlabel(x.replace("_", " ").title(), fontsize=12, weight="bold", color="grey")
    ax.set_ylabel(y.replace("_", " ").title(), fontsize=12, weight="bold", color="grey")
    ax.grid(axis="y", color="grey", linestyle=":", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_titles(fig, title, subtitle)
    if group:
        ax.legend(title=group.replace("_", " ").title(), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    plt.show()
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
        metric = df.groupby(group)[y].mean()
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
        for x_val, label in event_lines.items():
            fig.add_vline(x=x_val, line=dict(color="red", width=2, dash="dash"),
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
