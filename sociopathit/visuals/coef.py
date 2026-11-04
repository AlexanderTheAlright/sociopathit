"""
coef.py: Sociopath-it Visualization Module
---------------------------------------------
Coefficient plot with 95% confidence intervals.

Extras:
- Significance stars and bolded text for p-values
- Continuous or discrete color modes
- Multiple CI styles: line | bracket | spike
- Interactive version with corrected spike margins
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.colors import Normalize
from matplotlib import cm

from ..utils.style import set_style, generate_semantic_palette, apply_titles, format_tick_labels, get_continuous_cmap

# ══════════════════════════════════════════════════════════════════════════════
# Helper for significance stars
# ══════════════════════════════════════════════════════════════════════════════
def sigstars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# STATIC VERSION
# ══════════════════════════════════════════════════════════════════════════════
def coef(
    df,
    term_col="term",
    estimate_col="estimate",
    lower_col="conf.low",
    upper_col="conf.high",
    p_col=None,
    group_col=None,
    title=None,
    subtitle=None,
    style_mode="viridis",
    annotate=True,
    figsize=(9, 7),
    sort="asc",
    ci_style="bracket",
    color_mode="continuous",
):
    set_style(style_mode)
    df = df.copy()
    df = df.sort_values(estimate_col, ascending=(sort == "asc")).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Color logic
    if color_mode == "continuous":
        norm = Normalize(vmin=df[estimate_col].min(), vmax=df[estimate_col].max())
        if style_mode == "reviewer3":
            colors = ["#000000"] * len(df)
        else:
            cmap = matplotlib.colormaps[get_continuous_cmap(style_mode)]
            colors = [cmap(0.15 + 0.7 * norm(val)) for val in df[estimate_col]]
    elif group_col and group_col in df.columns:
        groups = df[group_col].unique().tolist()
        thirds = max(1, len(groups) // 3)
        group_dict = {
            "positive": groups[:thirds],
            "neutral": groups[thirds: 2 * thirds],
            "negative": groups[2 * thirds:],
        }
        palette = generate_semantic_palette(group_dict, mode=style_mode)
        colors = [palette.get(v, "grey") for v in df[group_col]]
    else:
        colors = ["#4c78a8"] * len(df)

    # Plot
    for i, row in df.iterrows():
        est, lo, hi = row[estimate_col], row[lower_col], row[upper_col]
        sig = p_col and p_col in df.columns and row[p_col] < 0.05
        stars = sigstars(row[p_col]) if (p_col and p_col in df.columns) else ""

        # CI style
        if ci_style == "line":
            ax.plot([lo, hi], [i, i], color="grey", lw=1.4)
        elif ci_style == "bracket":
            ax.plot([lo, hi], [i, i], color="grey", lw=1.2)
            ax.plot([lo, lo], [i - 0.2, i + 0.2], color="grey", lw=1)
            ax.plot([hi, hi], [i - 0.2, i + 0.2], color="grey", lw=1)
        elif ci_style == "spike":
            ax.plot([lo, hi], [i, i], color="grey", lw=1)
            ax.plot([lo, lo], [i, i + 0.25], color="grey", lw=1)
            ax.plot([hi, hi], [i, i + 0.25], color="grey", lw=1)

        # Point marker
        ax.scatter(
            est,
            i,
            s=150 if sig else 100,
            color=colors[i],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )

        # Annotation with background box for clarity
        if annotate:
            ax.text(
                est,
                i + 0.35,
                f"{est:.2f}{stars}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold" if sig else "normal",
                color="black",
                bbox=dict(facecolor="white", edgecolor='#333333',
                         linewidth=1.5, boxstyle="round,pad=0.3", alpha=0.95),
                zorder=4,
            )

    # Zero line
    ax.axvline(0, color="grey", linestyle="--", lw=1, alpha=0.7)

    # ─── Widen x-axis window to avoid misleading narrow ranges ───────────────
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min

    # Add 20% padding on each side for better context
    padding = x_range * 0.20
    ax.set_xlim(x_min - padding, x_max + padding)

    ax.set_yticks(range(len(df)))
    # Leave room above the top row so its value label stays inside the axes
    ax.set_ylim(-0.6, len(df) - 0.1)
    # Bold y-axis labels when significant
    y_labels = []
    for i, row in df.iterrows():
        label = str(row[term_col]).replace("_", " ").title()
        sig = p_col and p_col in df.columns and row[p_col] < 0.05
        y_labels.append(label)

    ax.set_yticklabels(y_labels, fontsize=12, color="#222222")

    # Bold the significant labels
    if p_col and p_col in df.columns:
        for i, (tick, row) in enumerate(zip(ax.get_yticklabels(), df.iterrows())):
            idx, row_data = row
            if row_data[p_col] < 0.05:
                tick.set_fontweight('bold')
    ax.set_xlabel("Coefficient Estimate", fontsize=13, weight="bold", color="black")
    ax.tick_params(axis="x", labelsize=11)

    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color("black")

    apply_titles(fig, title, subtitle)

    # Format tick labels: bold (no rotation for coefficient plots)
    format_tick_labels(ax, rotation_x=0)

    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE VERSION
# ══════════════════════════════════════════════════════════════════════════════
def coef_interactive(
    df,
    term_col="term",
    estimate_col="estimate",
    lower_col="conf.low",
    upper_col="conf.high",
    p_col=None,
    group_col=None,
    title=None,
    subtitle=None,
    style_mode="viridis",
    sort="asc",
    ci_style="bracket",
    color_mode="continuous",
):
    set_style(style_mode)
    df = df.copy()
    df = df.sort_values(estimate_col, ascending=(sort == "asc")).reset_index(drop=True)

    # Colors
    if color_mode == "continuous":
        norm = Normalize(vmin=df[estimate_col].min(), vmax=df[estimate_col].max())
        cmap = matplotlib.colormaps[get_continuous_cmap(style_mode)]
        shades = [0.0] * len(df) if style_mode == "reviewer3" else 0.15 + 0.7 * norm(df[estimate_col])
        colors = [
            "rgba(0, 0, 0, 1.00)" if style_mode == "reviewer3" else
            f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a:.2f})"
            for r, g, b, a in cmap(shades)
        ]
    elif group_col and group_col in df.columns:
        groups = df[group_col].unique().tolist()
        thirds = max(1, len(groups) // 3)
        group_dict = {
            "positive": groups[:thirds],
            "neutral": groups[thirds: 2 * thirds],
            "negative": groups[2 * thirds:],
        }
        palette = generate_semantic_palette(group_dict, mode=style_mode)
        colors = [palette.get(v, "grey") for v in df[group_col]]
    else:
        colors = ["#4c78a8"] * len(df)

    fig = go.Figure()

    for i, row in df.iterrows():
        est, lo, hi = row[estimate_col], row[lower_col], row[upper_col]
        p = row[p_col] if (p_col and p_col in df.columns) else 1.0
        stars = sigstars(p)
        sig = p < 0.05

        # CI line
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[i, i],
                mode="lines",
                line=dict(color="grey", width=1.2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        if ci_style in ["bracket", "spike"]:
            tick_len = 0.25
            fig.add_trace(go.Scatter(x=[lo, lo], y=[i - tick_len, i + tick_len],
                                     mode="lines", line=dict(color="grey", width=1), hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[hi, hi], y=[i - tick_len, i + tick_len],
                                     mode="lines", line=dict(color="grey", width=1), hoverinfo="skip"))

        # Point + label
        fig.add_trace(
            go.Scatter(
                x=[est],
                y=[i],
                mode="markers+text",
                marker=dict(
                    size=18 if sig else 14,
                    color=colors[i],
                    line=dict(color="white", width=0.7)
                ),
                text=[f"<b>{est:.2f}{stars}</b>" if sig else f"{est:.2f}"],
                textposition="top center",
                textfont=dict(size=14, color="#111111"),
                hovertemplate=f"<b>{row[term_col]}</b><br>Estimate: {est:.2f}<br>95% CI: [{lo:.2f}, {hi:.2f}]<br>p = {p:.2f}<extra></extra>",
                showlegend=False,
            )
        )

    fig.add_vline(x=0, line=dict(color="grey", width=1, dash="dash"))

    # Bold y-axis labels when significant
    ticktext_list = []
    for i, row in df.iterrows():
        label = str(row[term_col]).replace("_", " ").title()
        p = row[p_col] if (p_col and p_col in df.columns) else 1.0
        if p < 0.05:
            label = f"<b>{label}</b>"
        ticktext_list.append(label)

    fig.update_yaxes(
        tickvals=list(range(len(df))),
        ticktext=ticktext_list,
        autorange="reversed",
        range=[-0.5, len(df) - 0.5],  # prevents clipping of spikes
    )
    fig.update_layout(
        title=f"<b>{title}</b><br><span style='color:grey'>{subtitle or ''}</span>",
        template="plotly_white",
        height=750,
        width=1000,
        margin=dict(t=100, b=80, l=140, r=80),
        xaxis_title="Coefficient Estimate",
        yaxis_title="",
        showlegend=False,
    )
    return fig
