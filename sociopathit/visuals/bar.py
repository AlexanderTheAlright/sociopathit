"""
bar.py — Sociopath-it Visualization Module 🧱
--------------------------------------------
Flexible categorical comparisons:
- vertical, horizontal, or stacked bars
- optional highlight bar
- consistent Sociopath-it styling
- Plotly interactive counterpart
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from ..utils.style import (
    set_style,
    generate_semantic_palette,
    apply_titles,
    get_data_element_kwargs,
)


# ══════════════════════════════════════════════════════════════════════════════
# STATIC VERSION
# ══════════════════════════════════════════════════════════════════════════════

def bar(
    df,
    x,
    y,
    title=None,
    subtitle=None,
    palette=None,
    n=None,
    style_mode="viridis",
    orientation="vertical",        # 'vertical', 'horizontal', 'stacked'
    highlight=None,                # highlight label
    highlight_color="#D62828",
    trace_line=False,
    trace_arrow=True,
    sort="none",                   # 'none', 'asc', or 'desc'
    group_spacing=None,            # e.g. [(0,2), (3,5)] or int for split index
):
    """
    Sociopath-it bar plot with optional sorting, grouping gaps, and curved trace line with arrowhead.
    """

    # ─────────────────────────────────────────────
    # Sort and group spacing
    # ─────────────────────────────────────────────
    df = df.copy()
    if sort == "asc":
        df = df.sort_values(y, ascending=True)
    elif sort == "desc":
        df = df.sort_values(y, ascending=False)

    # Add gaps between groups if requested
    if isinstance(group_spacing, int):
        split_points = [group_spacing]
    elif isinstance(group_spacing, (list, tuple)):
        split_points = [g[1] for g in group_spacing]
    else:
        split_points = []

    # Apply pseudo-gap by inserting NaN rows
    if split_points:
        dfs = []
        last = 0
        for sp in split_points:
            dfs.append(df.iloc[last:sp])
            dfs.append(
                {x: f"", y: np.nan}
            )  # add blank separator
            last = sp
        dfs.append(df.iloc[last:])
        df = (
            pd.concat([pd.DataFrame(d) if not isinstance(d, dict) else pd.DataFrame([d]) for d in dfs])
            .reset_index(drop=True)
        )

    # ─────────────────────────────────────────────
    # Styling setup
    # ─────────────────────────────────────────────
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    if palette is None:
        groups = {"positive": [v for v in df[x].dropna().unique().tolist() if v != ""]}
        palette = generate_semantic_palette(groups, mode=style_mode)

    colors = [
        "white" if v == "" else (
            highlight_color if (highlight and v == highlight) else palette.get(v, cm.get_cmap("viridis")(0.6))
        )
        for v in df[x]
    ]
    kwargs = get_data_element_kwargs()

    # ─────────────────────────────────────────────
    # Main plotting logic
    # ─────────────────────────────────────────────
    if orientation == "horizontal":
        ax.barh(df[x], df[y], color=colors, **kwargs)
        ax.set_xlabel(y.title(), fontsize=12, weight="bold", color="grey")
        ax.set_ylabel("")
        for i, val in enumerate(df[y]):
            if not np.isnan(val):
                ax.text(val + (df[y].max() * 0.015), i, f"{val:,}", va="center", fontsize=9, color="grey")

    elif orientation == "stacked":
        cols = [c for c in df.columns if c not in [x, y]]
        bottom = np.zeros(len(df))
        for c in cols:
            vals = df[c].values
            ax.bar(df[x], vals, bottom=bottom, label=c, color=palette.get(c, cm.get_cmap("viridis")(0.6)), **kwargs)
            bottom += vals
        ax.legend(frameon=False)
        ax.set_ylabel("Total")
        ax.set_xlabel(x.title())

    else:  # vertical
        ax.bar(df[x], df[y], color=colors, **kwargs)
        ax.set_xlabel(x.title(), fontsize=12, weight="bold", color="grey")
        ax.set_ylabel(y.title(), fontsize=12, weight="bold", color="grey")

        for i, val in enumerate(df[y]):
            if not np.isnan(val):
                ax.text(i, val + (df[y].max() * 0.03), f"{val:,}", ha="center", fontsize=9, color="grey")

        # ─────────────────────────────────────────────
        # Optional trace line and arrowhead
        # ─────────────────────────────────────────────
        if trace_line:
            # Smooth curve between bar tops
            valid_mask = ~df[y].isna()
            x_idx = np.arange(len(df))[valid_mask]
            y_vals = df[y][valid_mask].values

            # cubic spline smoothing
            spl = make_interp_spline(x_idx, y_vals, k=2)
            xs = np.linspace(x_idx.min(), x_idx.max(), 300)
            ys = spl(xs)

            # draw the curve + dots
            ax.plot(xs, ys, color="grey", lw=1.3, alpha=0.85, zorder=3)
            ax.scatter(x_idx[:-1], y_vals[:-1], color="grey", s=15, zorder=4)

            if trace_arrow:
                # ---- Arrow at end of curve with better positioning ----
                x_end, y_end = xs[-1], ys[-1]
                # Use larger step back for clearer direction
                step_back = min(10, len(xs) // 10)
                x_prev, y_prev = xs[-step_back], ys[-step_back]

                ax.annotate(
                    "",
                    xy=(x_end, y_end),
                    xytext=(x_prev, y_prev),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="grey",
                        lw=2.0,
                        alpha=0.85,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=6,
                )


    # ─────────────────────────────────────────────
    # Styling and finishing touches
    # ─────────────────────────────────────────────
    ax.grid(axis="y" if orientation != "horizontal" else "x", linestyle=":", color="grey", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    apply_titles(fig, title or f"{y.title()} by {x.title()}", subtitle, n=n)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    plt.show()
    return fig, ax
