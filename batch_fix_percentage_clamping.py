"""
Batch fix percentage clamping in scatter.py, boxplot.py, density.py, and hist.py
"""

import re

files_to_fix = [
    'sociopathit/visuals/scatter.py',
    'sociopathit/visuals/boxplot.py',
    'sociopathit/visuals/density.py',
    'sociopathit/visuals/hist.py',
]

# Pattern to find the old padding code
old_pattern_scatter = r'''    # ─── Widen y-axis window to avoid misleading narrow ranges ───────────────
    y_min, y_max = ax\.get_ylim\(\)
    y_range = y_max - y_min

    # Add 20% padding on each side for better context and to avoid visual exaggeration
    padding = y_range \* 0\.20
    ax\.set_ylim\(y_min - padding, y_max \+ padding\)

    # For proportions/percentages \(values between 0-100\), ensure we show meaningful context
    if y_max <= 100\.0 and y_min >= 0:
        # If the range is very narrow, widen to show at least 20% of the full scale
        if y_range < 20:  # Less than 20 percentage points
            center = \(y_min \+ y_max\) / 2
            new_min = max\(0, center - 15\)  # At least 30% window \(15% on each side\)
            new_max = min\(100\.0, center \+ 15\)
            ax\.set_ylim\(new_min, new_max\)
    # For proportions \(0-1 scale\)
    elif y_max <= 1\.0 and y_min >= 0:
        if y_range < 0\.2:  # Less than 0\.2 \(20% of scale\)
            center = \(y_min \+ y_max\) / 2
            ax\.set_ylim\(max\(0, center - 0\.15\), min\(1\.0, center \+ 0\.15\)\)'''

new_code_scatter = '''    # ─── Widen y-axis window to avoid misleading narrow ranges ───────────────
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Check if data appears to be percentage (0-100) or proportion (0-1) BEFORE padding
    is_percentage = (y_max <= 100.0 and y_min >= 0)
    is_proportion = (y_max <= 1.0 and y_min >= 0)

    # Add 20% padding on each side for better context and to avoid visual exaggeration
    padding = y_range * 0.20
    new_min = y_min - padding
    new_max = y_max + padding

    # For percentage/proportion data, clamp to valid ranges
    if is_percentage:
        # Don't go below 0 or above 100 for percentage data
        new_min = max(0, new_min)
        new_max = min(100.0, new_max)
        # If the range is very narrow, widen to show at least 20% of the full scale
        if y_range < 20:  # Less than 20 percentage points
            center = (y_min + y_max) / 2
            new_min = max(0, center - 15)  # At least 30% window (15% on each side)
            new_max = min(100.0, center + 15)
    elif is_proportion:
        # Don't go below 0 or above 1 for proportion data
        new_min = max(0, new_min)
        new_max = min(1.0, new_max)
        if y_range < 0.2:  # Less than 0.2 (20% of scale)
            center = (y_min + y_max) / 2
            new_min = max(0, center - 0.15)
            new_max = min(1.0, center + 0.15)

    ax.set_ylim(new_min, new_max)'''

# For boxplot - has slightly different logic (vertical orientation check)
old_pattern_boxplot = r'''    # ─── Widen y-axis window to avoid misleading narrow ranges ───────────────
    if orientation == "vertical":
        y_min, y_max = ax\.get_ylim\(\)
        y_range = y_max - y_min

        # Add 20% padding on each side for better context
        padding = y_range \* 0\.20
        ax\.set_ylim\(y_min - padding, y_max \+ padding\)

        # For proportions/percentages \(values between 0-100\), ensure meaningful context
        if y_max <= 100\.0 and y_min >= 0:
            if y_range < 20:  # Less than 20 percentage points
                center = \(y_min \+ y_max\) / 2
                new_min = max\(0, center - 15\)
                new_max = min\(100\.0, center \+ 15\)
                ax\.set_ylim\(new_min, new_max\)
        # For proportions \(0-1 scale\)
        elif y_max <= 1\.0 and y_min >= 0:
            if y_range < 0\.2:
                center = \(y_min \+ y_max\) / 2
                ax\.set_ylim\(max\(0, center - 0\.15\), min\(1\.0, center \+ 0\.15\)\)'''

new_code_boxplot = '''    # ─── Widen y-axis window to avoid misleading narrow ranges ───────────────
    if orientation == "vertical":
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        # Check if data appears to be percentage (0-100) or proportion (0-1) BEFORE padding
        is_percentage = (y_max <= 100.0 and y_min >= 0)
        is_proportion = (y_max <= 1.0 and y_min >= 0)

        # Add 20% padding on each side for better context
        padding = y_range * 0.20
        new_min = y_min - padding
        new_max = y_max + padding

        # For percentage/proportion data, clamp to valid ranges
        if is_percentage:
            # Don't go below 0 or above 100 for percentage data
            new_min = max(0, new_min)
            new_max = min(100.0, new_max)
            if y_range < 20:  # Less than 20 percentage points
                center = (y_min + y_max) / 2
                new_min = max(0, center - 15)
                new_max = min(100.0, center + 15)
        elif is_proportion:
            # Don't go below 0 or above 1 for proportion data
            new_min = max(0, new_min)
            new_max = min(1.0, new_max)
            if y_range < 0.2:
                center = (y_min + y_max) / 2
                new_min = max(0, center - 0.15)
                new_max = min(1.0, center + 0.15)

        ax.set_ylim(new_min, new_max)'''

# For hist and density - only top padding
old_pattern_hist = r'''    # ─── Widen y-axis window to avoid misleading narrow ranges ───────────────
    y_min, y_max = ax\.get_ylim\(\)
    y_range = y_max - y_min

    # Add 20% padding on top for better context
    padding = y_range \* 0\.20
    ax\.set_ylim\(y_min, y_max \+ padding\)'''

new_code_hist = '''    # ─── Widen y-axis window to avoid misleading narrow ranges ───────────────
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Add 20% padding on top for better context (keep bottom at natural value)
    padding = y_range * 0.20
    ax.set_ylim(y_min, y_max + padding)'''

for filepath in files_to_fix:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original = content

        if 'scatter.py' in filepath:
            content = re.sub(old_pattern_scatter, new_code_scatter, content, flags=re.MULTILINE)
        elif 'boxplot.py' in filepath:
            content = re.sub(old_pattern_boxplot, new_code_boxplot, content, flags=re.MULTILINE)
        elif 'hist.py' in filepath or 'density.py' in filepath:
            content = re.sub(old_pattern_hist, new_code_hist, content, flags=re.MULTILINE)

        if content != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[OK] Fixed {filepath}")
        else:
            print(f"[SKIP] No changes needed in {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to process {filepath}: {e}")

print("\nDone!")
