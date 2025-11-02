"""
style.py — The Sociopath-it World of Data Visualization 🎨
-----------------------------------------------------------
Core style engine for Sociopath-it plots.

Implements multiple thematic styles reflecting interpretive mood:
- 'fiery'     → intense reds, oranges, purples (high emotional contrast)
- 'viridis'   → balanced, perceptually uniform viridis-based default
- 'sentiment' → green-positive, red-negative moral economy tone
- 'plainjane' → light red/blue academic contrast
- 'reviewer3' → grayscale, journal-safe, publishable
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════════════
# I. STYLE CONFIGURATION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

AVAILABLE_STYLES = ["fiery", "viridis", "sentiment", "plainjane", "reviewer3"]
ACTIVE_STYLE = "viridis"

def set_style(mode: str = "viridis"):
    """
    Apply a Sociopath-it visual style theme.

    Parameters
    ----------
    mode : str
        One of {'fiery','viridis','sentiment','plainjane','reviewer3'}.
    """
    mode = mode.lower()
    if mode not in AVAILABLE_STYLES:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {AVAILABLE_STYLES}.")

    plt.style.use("default")
    plt.rcParams.update({
        # Canvas and spines
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#000000",
        "axes.linewidth": 1.0,

        # Default figure size (smaller) and DPI (higher)
        "figure.figsize": (8.0, 5.0),
        "figure.dpi": 300,

        # Grid - improved defaults
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E0E0E0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.75,
        "grid.alpha": 0.5,
        "axes.axisbelow": True,

        # Typography - larger, clearer fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlecolor": "#000000",
        "axes.labelsize": 11,
        "axes.labelweight": "bold",
        "axes.labelcolor": "#000000",

        # Tick parameters
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Legend - larger and clearer
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": False,
        "legend.edgecolor": "#CCCCCC",
        "legend.facecolor": "white",
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "legend.borderpad": 0.6,
        "legend.labelspacing": 0.6,
        "legend.handlelength": 2.0,
        "legend.handleheight": 0.7,
        "legend.handletextpad": 0.8,
        "legend.borderaxespad": 0.5,
        "legend.columnspacing": 2.0,

        # Lines and markers
        "lines.linewidth": 2.5,
        "lines.markersize": 7,
        "lines.markeredgewidth": 1.5,

        # Export settings
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "savefig.facecolor": "white",
    })

    global ACTIVE_STYLE
    ACTIVE_STYLE = mode


def format_tick_labels(ax, rotation_x=0, bold=True):
    """
    Apply Sociopath-it tick label formatting: bold text with optional rotation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to format
    rotation_x : int or float
        Rotation angle for x-tick labels (default 0° for horizontal)
    bold : bool
        Whether to make labels bold (default True)
    """
    weight = 'bold' if bold else 'normal'

    # Format x-tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight(weight)
        label.set_rotation(rotation_x)
        if rotation_x != 0:
            label.set_ha('right')

    # Format y-tick labels
    for label in ax.get_yticklabels():
        label.set_fontweight(weight)


# ══════════════════════════════════════════════════════════════════════════════
# II. SEMANTIC COLOR PALETTES
# ══════════════════════════════════════════════════════════════════════════════

def generate_semantic_palette(groups: dict, mode: str = None):
    """
    Generate a semantic color palette depending on the active or specified style.

    For small numbers of items (2-4), uses explicitly defined high-contrast colors
    to ensure maximum visual distinction. For 5+ items, uses colormap sampling.

    Parameters
    ----------
    groups : dict
        e.g., {'positive': [...], 'neutral': [...], 'negative': [...]}
    mode : str, optional
        Override style manually.

    Returns
    -------
    palette : dict  {variable: RGBA tuple}
    """
    style = (mode or globals().get("ACTIVE_STYLE", "viridis")).lower()
    palette = {}

    # Define explicit high-contrast color sets for small numbers of items
    # Format: {style: {n_items: [color1, color2, ...]}}
    DISCRETE_PALETTES = {
        "viridis": {
            1: [(0.127568, 0.566949, 0.550556, 1.0)],  # Teal middle
            2: [(0.267004, 0.004874, 0.329415, 1.0),   # Dark purple
                (0.993248, 0.906157, 0.143936, 1.0)],  # Bright yellow
            3: [(0.267004, 0.004874, 0.329415, 1.0),   # Dark purple
                (0.127568, 0.566949, 0.550556, 1.0),   # Teal
                (0.993248, 0.906157, 0.143936, 1.0)],  # Bright yellow
            4: [(0.267004, 0.004874, 0.329415, 1.0),   # Dark purple
                (0.153364, 0.497000, 0.557724, 1.0),   # Blue-teal
                (0.468667, 0.856531, 0.434154, 1.0),   # Yellow-green
                (0.993248, 0.906157, 0.143936, 1.0)],  # Bright yellow
        },
        "sentiment": {
            1: [(0.5, 0.5, 0.5, 1.0)],                 # Gray
            2: [(0.133, 0.545, 0.133, 1.0),            # Forest green
                (0.839, 0.153, 0.157, 1.0)],           # Crimson red
            3: [(0.133, 0.545, 0.133, 1.0),            # Forest green
                (0.5, 0.5, 0.5, 1.0),                  # Gray
                (0.839, 0.153, 0.157, 1.0)],           # Crimson red
            4: [(0.000, 0.392, 0.000, 1.0),            # Dark green
                (0.133, 0.545, 0.133, 1.0),            # Forest green
                (0.839, 0.153, 0.157, 1.0),            # Crimson red
                (0.545, 0.000, 0.000, 1.0)],           # Dark red
        },
        "fiery": {
            1: [(0.8, 0.4, 0.0, 1.0)],                 # Orange middle
            2: [(0.050383, 0.029803, 0.527975, 1.0),  # Dark purple/blue
                (0.940015, 0.975158, 0.131326, 1.0)], # Bright yellow
            3: [(0.050383, 0.029803, 0.527975, 1.0),  # Dark purple/blue
                (0.796338, 0.278826, 0.473089, 1.0),  # Pink/magenta
                (0.940015, 0.975158, 0.131326, 1.0)], # Bright yellow
            4: [(0.050383, 0.029803, 0.527975, 1.0),  # Dark purple/blue
                (0.618896, 0.124865, 0.608374, 1.0),  # Purple
                (0.905384, 0.500563, 0.173636, 1.0),  # Orange
                (0.940015, 0.975158, 0.131326, 1.0)], # Bright yellow
        },
        "plainjane": {
            1: [(0.5, 0.5, 0.5, 1.0)],                 # Gray
            2: [(0.122, 0.467, 0.706, 1.0),            # Blue
                (1.0, 1.0, 1.0, 1.0)],                 # White
            3: [(0.122, 0.467, 0.706, 1.0),            # Blue
                (0.5, 0.5, 0.5, 1.0),                  # Gray
                (1.0, 1.0, 1.0, 1.0)],                 # White
            4: [(0.031, 0.188, 0.420, 1.0),            # Dark blue
                (0.122, 0.467, 0.706, 1.0),            # Light blue
                (0.7, 0.7, 0.7, 1.0),                  # Light gray
                (1.0, 1.0, 1.0, 1.0)],                 # White
        },
        "reviewer3": {
            1: [(0.0, 0.0, 0.0, 1.0)],                 # Black
            2: [(0.0, 0.0, 0.0, 1.0),                  # Pure black
                (1.0, 1.0, 1.0, 1.0)],                 # Pure white
            3: [(0.0, 0.0, 0.0, 1.0),                  # Black
                (0.5, 0.5, 0.5, 1.0),                  # Medium gray
                (1.0, 1.0, 1.0, 1.0)],                 # White
            4: [(0.0, 0.0, 0.0, 1.0),                  # Black
                (0.33, 0.33, 0.33, 1.0),               # Dark gray
                (0.67, 0.67, 0.67, 1.0),               # Light gray
                (1.0, 1.0, 1.0, 1.0)],                 # White
        },
    }

    # Setup colormaps for 5+ items (legacy behavior)
    if style == "fiery":
        cmaps = {"positive": cm.plasma, "neutral": cm.magma, "negative": cm.inferno}
        ranges = {"positive": (0.0, 1.0), "neutral": (0.0, 1.0), "negative": (0.0, 1.0)}
    elif style == "sentiment":
        cmaps = {"positive": cm.Greens, "neutral": cm.Greys, "negative": cm.Reds_r}
        ranges = {"positive": (0.3, 1.0), "neutral": (0.2, 0.8), "negative": (0.3, 1.0)}
    elif style == "plainjane":
        cmaps = {"positive": cm.Blues, "neutral": cm.Greys, "negative": cm.Reds}
        ranges = {"positive": (0.2, 1.0), "neutral": (0.2, 0.8), "negative": (0.2, 1.0)}
    elif style == "reviewer3":
        cmaps = {"positive": cm.Greys, "neutral": cm.Greys, "negative": cm.Greys}
        ranges = {"positive": (0.0, 1.0), "neutral": (0.0, 1.0), "negative": (0.0, 1.0)}
    else:  # viridis
        cmaps = {"positive": cm.viridis, "neutral": cm.viridis, "negative": cm.viridis}
        ranges = {"positive": (0.0, 1.0), "neutral": (0.0, 1.0), "negative": (0.0, 1.0)}

    # Build palette
    for group, items in groups.items():
        if not items:
            continue

        n_items = len(items)

        # For 1-4 items, use explicit discrete palette for maximum contrast
        if n_items <= 4:
            colors = DISCRETE_PALETTES.get(style, DISCRETE_PALETTES["viridis"])[n_items]
            for item, color in zip(items, colors):
                palette[item] = color
        else:
            # For 5+ items, use colormap sampling
            g = group.lower()
            key = "positive" if g.startswith("pos") else \
                  "neutral" if g.startswith("neu") else \
                  "negative" if g.startswith("neg") else "positive"
            cmap, (low, high) = cmaps[key], ranges[key]

            vals = np.linspace(low, high, n_items)
            for item, v in zip(items, vals):
                palette[item] = cmap(v)

    return palette


# ══════════════════════════════════════════════════════════════════════════════
# II.B. CONTINUOUS COLORMAPS FOR HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════

# Dictionary mapping style modes to their primary colormaps
COLORS_DICT = {
    "fiery": cm.inferno,
    "viridis": cm.viridis,
    "sentiment": cm.RdYlGn,
    "plainjane": cm.RdBu_r,
    "reviewer3": cm.Greys,
}


def get_continuous_cmap(mode: str = None):
    """
    Get continuous colormap for heatmaps, correlation matrices, etc.

    Parameters
    ----------
    mode : str, optional
        Style mode. If None, uses global ACTIVE_STYLE.

    Returns
    -------
    cmap_name : str
        Matplotlib colormap name suitable for continuous data.

    Examples
    --------
    Fiery: Dark red → white (heat aesthetic)
    Viridis: Standard perceptually uniform viridis
    Sentiment: Red-white-green diverging (correlation)
    Plainjane: Blue-white-red diverging
    Reviewer3: Grayscale white-to-black
    """
    style = (mode or globals().get("ACTIVE_STYLE", "viridis")).lower()

    cmap_mapping = {
        "fiery": "Reds",           # Dark red → white for "heat" aesthetic
        "viridis": "viridis",      # Perceptually uniform
        "sentiment": "RdYlGn",     # Red-yellow-green diverging
        "plainjane": "RdBu_r",     # Blue-white-red diverging
        "reviewer3": "Greys",      # Grayscale for publication
    }

    return cmap_mapping.get(style, "viridis")


# ══════════════════════════════════════════════════════════════════════════════
# III. TITLING SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def apply_titles(fig, title=None, subtitle=None, n=None):
    """
    Sociopath-it title logic:
    - If subtitle exists: title and subtitle at center-left
    - If no subtitle: centered title
    - Optional n count in bottom right
    """
    if title is None:
        return

    if subtitle:
        # Center-left placement for title + subtitle
        fig.text(
            0.02,
            0.97,
            f"{title}",
            fontsize=15,
            weight="bold",
            color="black",
            ha="left",
            va="center",
        )
        fig.text(
            0.02,
            0.94,
            f"{subtitle}",
            fontsize=11,
            color="grey",
            ha="left",
            va="center",
        )
    else:
        # Centered title when no subtitle
        fig.suptitle(
            f"{title}",
            fontsize=15,
            weight="bold",
            color="black",
            y=0.96,
            ha="center",
        )

    if n is not None:
        fig.text(
            0.99,
            0.01,
            f"(n = {n:,})",
            fontsize=9,
            color="grey",
            ha="right",
            va="bottom",
        )


# ══════════════════════════════════════════════════════════════════════════════
# IV. DATA ELEMENT DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

def get_data_element_kwargs():
    """Return default kwargs for Sociopath-it data elements."""
    return {"edgecolor": "black", "linewidth": 1.0}


# ══════════════════════════════════════════════════════════════════════════════
# V. LEGEND UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def draw_legend_group(ax, fig, title, var_list, palette,
                      start_x, start_y, line_height=0.04, spacing=0.04):
    """Draw structured legend group (bars/lines)."""
    y = start_y
    fig.text(start_x, y, title, transform=ax.transAxes,
             fontsize=11, weight="bold", color="#333333", ha="left", va="top")
    y -= line_height
    for var in var_list:
        if var not in palette:
            continue
        rect = Rectangle((start_x, y - 0.015), 0.015, 0.025,
                         facecolor=palette[var], transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)
        fig.text(start_x + 0.02, y, var,
                 transform=ax.transAxes, fontsize=10, color="#333333",
                 ha="left", va="center")
        y -= line_height
    return y - spacing


def draw_scatter_legend(ax, fig, title, var_list, palette,
                        start_x, start_y, line_height=0.04, spacing=0.04):
    """Draw structured legend group (scatter)."""
    y = start_y
    fig.text(start_x, y, title, transform=ax.transAxes,
             fontsize=11, weight="bold", color="#333333", ha="left", va="top")
    y -= line_height
    for var in var_list:
        if var not in palette:
            continue
        ax.add_line(Line2D([start_x], [y - 0.01],
                           transform=ax.transAxes,
                           marker="o", color="w",
                           markerfacecolor=palette[var],
                           markersize=8, markeredgecolor="grey"))
        fig.text(start_x + 0.02, y, var,
                 transform=ax.transAxes, fontsize=10, color="#333333",
                 ha="left", va="center")
        y -= line_height
    return y - spacing


# ══════════════════════════════════════════════════════════════════════════════
# VI. SEMANTIC COLOR RETRIEVAL (THEME-AWARE)
# ══════════════════════════════════════════════════════════════════════════════

def get_color(semantic: str, mode: str = None) -> str:
    """
    Get theme-aware color for semantic use cases.

    For reviewer3, all colors map to grayscale (black/white/gray).
    For other themes, returns appropriate colors.

    Parameters
    ----------
    semantic : str
        One of: 'negative', 'positive', 'neutral', 'warning', 'highlight',
                'line', 'threshold', 'reference', 'primary', 'secondary'
    mode : str, optional
        Override style. If None, uses ACTIVE_STYLE.

    Returns
    -------
    color : str
        Color string (hex, rgb, or named color).

    Examples
    --------
    >>> get_color('warning')  # Returns 'red' in most themes, '#333333' in reviewer3
    >>> get_color('positive', mode='reviewer3')  # Returns '#666666'
    """
    style = (mode or globals().get("ACTIVE_STYLE", "viridis")).lower()

    if style == "reviewer3":
        # Pure grayscale mapping for publication
        color_map = {
            'negative': '#000000',      # Black for negative/warnings
            'positive': '#666666',      # Medium gray for positive
            'neutral': '#999999',       # Light gray for neutral
            'warning': '#000000',       # Black for warnings/thresholds
            'highlight': '#333333',     # Dark gray for highlights
            'line': '#666666',          # Medium gray for lines
            'threshold': '#000000',     # Black for thresholds
            'reference': '#000000',     # Black for reference lines
            'primary': '#333333',       # Dark gray primary
            'secondary': '#999999',     # Light gray secondary
            'increasing': '#666666',    # Medium gray
            'decreasing': '#333333',    # Dark gray
        }
    elif style == "sentiment":
        color_map = {
            'negative': '#d62728',      # Red
            'positive': '#2ca02c',      # Green
            'neutral': '#7f7f7f',       # Gray
            'warning': '#d62728',       # Red
            'highlight': '#ff7f0e',     # Orange
            'line': '#1f77b4',          # Blue
            'threshold': '#d62728',     # Red
            'reference': '#d62728',     # Red
            'primary': '#1f77b4',       # Blue
            'secondary': '#ff7f0e',     # Orange
            'increasing': '#2ca02c',    # Green
            'decreasing': '#d62728',    # Red
        }
    elif style == "fiery":
        color_map = {
            'negative': '#d62728',      # Red
            'positive': '#ff7f0e',      # Orange
            'neutral': '#8c564b',       # Brown
            'warning': '#d62728',       # Red
            'highlight': '#ff7f0e',     # Orange
            'line': '#e377c2',          # Pink
            'threshold': '#d62728',     # Red
            'reference': '#d62728',     # Red
            'primary': '#d62728',       # Red
            'secondary': '#ff7f0e',     # Orange
            'increasing': '#ff7f0e',    # Orange
            'decreasing': '#8c564b',    # Brown
        }
    elif style == "plainjane":
        color_map = {
            'negative': '#d62728',      # Red
            'positive': '#1f77b4',      # Blue
            'neutral': '#7f7f7f',       # Gray
            'warning': '#d62728',       # Red
            'highlight': '#ff7f0e',     # Orange
            'line': '#1f77b4',          # Blue
            'threshold': '#d62728',     # Red
            'reference': '#d62728',     # Red
            'primary': '#1f77b4',       # Blue
            'secondary': '#d62728',     # Red
            'increasing': '#1f77b4',    # Blue
            'decreasing': '#d62728',    # Red
        }
    else:  # viridis
        color_map = {
            'negative': '#d62728',      # Red
            'positive': '#2ca02c',      # Green
            'neutral': '#7f7f7f',       # Gray
            'warning': '#d62728',       # Red
            'highlight': '#ff7f0e',     # Orange
            'line': '#1f77b4',          # Blue
            'threshold': '#d62728',     # Red
            'reference': '#d62728',     # Red
            'primary': '#440154',       # Viridis dark purple
            'secondary': '#fde724',     # Viridis yellow
            'increasing': '#2ca02c',    # Green
            'decreasing': '#d62728',    # Red
        }

    return color_map.get(semantic, '#333333')
