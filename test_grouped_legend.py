"""
Test to verify grouped bar chart legend is on the right outside the plot.
"""
import pandas as pd
import sociopathit as sp
import matplotlib.pyplot as plt

# Create sample data
grouped_data = pd.DataFrame({
    'wave': ['W1', 'W2', 'W3', 'W4'],
    'Group A': [100, 110, 105, 115],
    'Group B': [95, 102, 98, 108],
    'Group C': [88, 95, 92, 100]
})

fig, ax = sp.visuals.bar.bar(
    grouped_data,
    x='wave',
    y='wave',  # dummy
    orientation='grouped',
    title='Grouped Bar Chart - Legend Placement Test',
    subtitle='Legend should be on the right outside the plot area',
    style_mode='viridis'
)
plt.savefig('test_grouped_legend_placement.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved test_grouped_legend_placement.png")
print("Legend should be on the right outside the plot area.")
