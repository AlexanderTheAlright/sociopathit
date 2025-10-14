"""
cluster.py — Sociopath-it Visualization Module 🧬
------------------------------------------------
Hierarchical clustering dendrogram for numeric data.

Features:
- Ward, average, complete, or single linkage
- Distance metric choice
- Optional color threshold and truncation
- Sociopath-it styling and semantic color support
- Interactive Plotly dendrogram counterpart
- Heatmap-cluster hybrid visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, dendrogram
from ..utils.style import set_style, apply_titles, generate_semantic_palette, get_continuous_cmap


# ══════════════════════════════════════════════════════════════════════════════
# STATIC VERSION
# ══════════════════════════════════════════════════════════════════════════════

def cluster(
    df,
    method="ward",
    metric="euclidean",
    title=None,
    subtitle=None,
    style_mode="viridis",
    color_threshold=None,
    truncate_mode=None,
    p=None,
    orientation="top",
    leaf_rotation=90,
    leaf_font_size=10,
    show_leaf_counts=True,
):
    """
    Sociopath-it hierarchical cluster dendrogram (robust).
    """
    set_style(style_mode)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    fig.set_facecolor("white")
    ax.set_facecolor("white")

    # Select numeric columns
    data = df.select_dtypes(include=[np.number])
    if data.empty:
        raise ValueError("No numeric columns found for clustering.")

    # Compute linkage matrix
    Z = linkage(data, method=method, metric=metric)

    # --- Assemble keyword args cleanly ---
    dendro_kwargs = dict(
        ax=ax,
        color_threshold=color_threshold,
        orientation=orientation,
        leaf_rotation=leaf_rotation,
        leaf_font_size=leaf_font_size,
        show_leaf_counts=show_leaf_counts,
    )

    if truncate_mode is not None:
        dendro_kwargs["truncate_mode"] = truncate_mode
        dendro_kwargs["p"] = int(p or 12)

    # --- Draw dendrogram ---
    dendrogram(Z, **dendro_kwargs)

    # Sociopath-it style cleanup
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", color="grey", linewidth=0.7)
    apply_titles(fig, title or "Hierarchical Cluster Dendrogram", subtitle)
    fig.tight_layout(rect=(0, 0, 1, 0.9 if subtitle else 0.94))
    plt.show()
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE VERSION
# ══════════════════════════════════════════════════════════════════════════════

def cluster_interactive(
    df,
    method="ward",
    metric="euclidean",
    title=None,
    subtitle=None,
    style_mode="viridis",
):
    """
    Interactive Plotly dendrogram for Sociopath-it clustering.

    Parameters
    ----------
    df : DataFrame
        Numeric dataset to cluster.
    method : str, default "ward"
        Linkage method ('ward', 'average', 'complete', 'single').
    metric : str, default "euclidean"
        Distance metric.
    """
    set_style(style_mode)
    data = df.select_dtypes(include=[np.number])
    if data.empty:
        raise ValueError("No numeric columns found for clustering.")

    fig = ff.create_dendrogram(
        data.values,
        labels=data.index.astype(str),
        linkagefun=lambda x: linkage(x, method=method, metric=metric),
        orientation="bottom",
    )

    fig.update_layout(
        title=f"<b>{title or 'Hierarchical Cluster Dendrogram'}</b><br><span style='color:grey'>{subtitle or ''}</span>",
        template="plotly_white",
        height=600,
        margin=dict(t=80, b=60, l=60, r=20),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HEATMAP-CLUSTER HYBRID
# ══════════════════════════════════════════════════════════════════════════════

def heatmap_cluster(
    df,
    method="ward",
    metric="euclidean",
    title=None,
    subtitle=None,
    style_mode="viridis",
    cmap=None,
    annot=False,
    figsize=(12, 10),
):
    """
    Sociopath-it clustered heatmap with dendrograms.

    Combines hierarchical clustering with heatmap visualization,
    showing dendrograms on both axes.

    Parameters
    ----------
    df : pd.DataFrame
        Numeric dataset to cluster and visualize.
    method : str, default "ward"
        Linkage method.
    metric : str, default "euclidean"
        Distance metric.
    title, subtitle : str
        Plot titles.
    style_mode : str
        Sociopath-it style theme: fiery (dark red heat), viridis,
        sentiment (RdYlGn), plainjane (RdBu), reviewer3 (grayscale).
    cmap : str, optional
        Colormap for heatmap. If None, uses style_mode's continuous colormap.
    annot : bool
        Show values in cells.
    figsize : tuple
        Figure size.
    """
    set_style(style_mode)

    # Use style-specific continuous colormap if not provided
    if cmap is None:
        cmap = get_continuous_cmap(style_mode)

    # Select numeric columns
    data = df.select_dtypes(include=[np.number])
    if data.empty:
        raise ValueError("No numeric columns found for clustering.")

    # Use seaborn's clustermap which does hierarchical clustering
    g = sns.clustermap(
        data,
        method=method,
        metric=metric,
        cmap=cmap,
        annot=annot,
        fmt=".2f" if annot else "",
        figsize=figsize,
        cbar_kws={"shrink": 0.8, "label": "Value"},
        dendrogram_ratio=(0.15, 0.15),
        linewidths=0.5,
        linecolor="white",
        center=0 if "Rd" in cmap else None,
    )

    # Apply Sociopath-it styling with proper spacing for title/subtitle
    # Position title in top-left area with more space
    title_text = title or "Clustered Heatmap"
    subtitle_text = subtitle or ""

    # Clear default title
    g.fig.suptitle("")

    # Add title and subtitle in top-left with proper spacing
    # The clustermap leaves space at top, use it efficiently
    if subtitle_text:
        # Title higher, subtitle below it
        g.fig.text(
            0.15, 0.98,
            title_text,
            fontsize=16,
            fontweight="bold",
            color="#111111",
            ha="left",
            va="top",
        )
        g.fig.text(
            0.15, 0.96,
            subtitle_text,
            fontsize=12,
            color="grey",
            ha="left",
            va="top",
        )
    else:
        # Just title
        g.fig.text(
            0.15, 0.98,
            title_text,
            fontsize=16,
            fontweight="bold",
            color="#111111",
            ha="left",
            va="top",
        )

    plt.show()
    return g.fig, g.ax_heatmap
