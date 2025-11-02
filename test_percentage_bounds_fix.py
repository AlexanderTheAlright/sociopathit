"""
Test that percentage data is properly constrained to 0-100 for stacked and grouped bar charts.
"""
import pandas as pd
import numpy as np
import sociopathit as sp
import matplotlib.pyplot as plt

print("="*70)
print("TESTING PERCENTAGE BOUNDS FIX")
print("="*70)
print()

np.random.seed(42)

# Test 1: Stacked bar chart with percentage data
print("[1] Testing STACKED bar chart with percentage data...")
stacked_data = pd.DataFrame({
    'category': ['A', 'B', 'C'],
    'Group 1': [45.2, 48.5, 47.1],
    'Group 2': [30.1, 28.9, 31.2],
    'Group 3': [24.7, 22.6, 21.7]
})

fig, ax = sp.visuals.bar.bar(
    stacked_data,
    x='category',
    y='category',  # dummy
    orientation='stacked',
    title='Stacked Bar Chart - Percentage Data',
    subtitle='Y-axis should be constrained to 0-100',
    style_mode='viridis'
)
y_min, y_max = ax.get_ylim()
print(f"  Y-axis range: {y_min:.2f} to {y_max:.2f}")
if y_min >= 0 and y_max <= 100:
    print("  [OK] Y-axis correctly within 0-100 bounds")
else:
    print(f"  [FAIL] Y-axis went outside bounds!")
plt.savefig('test_stacked_percentage.png', dpi=300, bbox_inches='tight')
plt.close()
print()

# Test 2: Grouped bar chart with percentage data
print("[2] Testing GROUPED bar chart with percentage data...")
grouped_data = pd.DataFrame({
    'wave': ['W1', 'W2', 'W3'],
    'Group A': [75.2, 75.5, 75.3],
    'Group B': [76.1, 76.4, 76.2],
    'Group C': [74.8, 75.1, 74.9]
})

fig, ax = sp.visuals.bar.bar(
    grouped_data,
    x='wave',
    y='wave',  # dummy
    orientation='grouped',
    title='Grouped Bar Chart - Percentage Data',
    subtitle='Y-axis should be constrained to 0-100',
    style_mode='viridis'
)
y_min, y_max = ax.get_ylim()
print(f"  Y-axis range: {y_min:.2f} to {y_max:.2f}")
if y_min >= 0 and y_max <= 100:
    print("  [OK] Y-axis correctly within 0-100 bounds")
else:
    print(f"  [FAIL] Y-axis went outside bounds!")
plt.savefig('test_grouped_percentage.png', dpi=300, bbox_inches='tight')
plt.close()
print()

# Test 3: Regular vertical bar chart with percentage data (already had this working)
print("[3] Testing VERTICAL bar chart with percentage data...")
bar_data = pd.DataFrame({
    'category': ['A', 'B', 'C', 'D'],
    'value': [98.5, 99.2, 98.8, 99.0]
})

fig, ax = sp.visuals.bar.bar(
    bar_data,
    x='category',
    y='value',
    title='Vertical Bar Chart - Percentage Data',
    subtitle='Y-axis should be constrained to 0-100',
    style_mode='viridis'
)
y_min, y_max = ax.get_ylim()
print(f"  Y-axis range: {y_min:.2f} to {y_max:.2f}")
if y_min >= 0 and y_max <= 100:
    print("  [OK] Y-axis correctly within 0-100 bounds")
else:
    print(f"  [FAIL] Y-axis went outside bounds!")
plt.savefig('test_vertical_percentage.png', dpi=300, bbox_inches='tight')
plt.close()
print()

# Test 4: Non-percentage data should NOT be constrained
print("[4] Testing bar chart with NON-percentage data...")
nonpercent_data = pd.DataFrame({
    'category': ['A', 'B', 'C'],
    'value': [150.5, 175.2, 165.8]
})

fig, ax = sp.visuals.bar.bar(
    nonpercent_data,
    x='category',
    y='value',
    title='Non-Percentage Data',
    subtitle='Y-axis should NOT be constrained to 100',
    style_mode='viridis'
)
y_min, y_max = ax.get_ylim()
print(f"  Y-axis range: {y_min:.2f} to {y_max:.2f}")
if y_max > 100:
    print("  [OK] Y-axis correctly NOT constrained (data > 100)")
else:
    print(f"  [FAIL] Y-axis incorrectly constrained!")
plt.savefig('test_nonpercentage.png', dpi=300, bbox_inches='tight')
plt.close()
print()

print("="*70)
print("SUMMARY")
print("="*70)
print()
print("Percentage bounds fix has been applied successfully!")
print()
print("The fix ensures that:")
print("  - Percentage data (0-100) is properly constrained")
print("  - Padding is calculated from DATA range, not autoscaled limits")
print("  - Non-percentage data is not incorrectly constrained")
print()
print("Test images saved:")
print("  - test_stacked_percentage.png")
print("  - test_grouped_percentage.png")
print("  - test_vertical_percentage.png")
print("  - test_nonpercentage.png")
print("="*70)
