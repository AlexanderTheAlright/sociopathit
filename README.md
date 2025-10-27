# Sociopath-it

**Sociopath-it** is a suite of Python tools for data cleaning, harmonization, analysis, and visualization. Built for sociologists who love code and distrust objectivity just enough to do it right.

---

## Overview

Sociopath-it is structured into five main modules, each named with the proper degree of irony:

| Module | Description                                                                                                              |
|:-------|:-------------------------------------------------------------------------------------------------------------------------|
| `data/` | Survey data management, file discovery, longitudinal alignment, and harmonization. Because your data is never clean enough on the first pass. |
| `cleansing/` | Additional harmonization utilities for variable recoding and codebook sanity. |
| `analyses/` | Statistical, causal, and machine learning tools. From OLS to SEM, propensity scores to random forests, and everything in between. |
| `visuals/` | Comprehensive visualization suite. White backgrounds, semantic palettes, and Bourdieu-approved taste. All 24 plot types, static and interactive. |
| `utils/` | Style functions and color palettes. The infrastructure behind the aesthetics. |

---

## Installation

```bash
pip install --force-reinstall --no-deps git+https://github.com/AlexanderTheAlright/sociopathit.git
```

---

## Package Structure

### Data Module (`sociopathit.data`)

Survey and panel data management tools. Handles the messy reality of longitudinal research.

- **discovery**: Find data files, resolve paths, detect wave structures from filenames
- **loading**: Load surveys and Stata files with automatic preprocessing and variable filtering
- **metadata**: Extract survey metadata, wave information, and ID variables
- **longitudinal**: Detect panel structure, align waves, build long-form datasets
- **preparation**: Harmonize variables, handle missing codes, prepare for analysis

**Example workflow:**
```python
from sociopathit.data.loading import load_all_surveys
from sociopathit.data.preparation import build_harmonized_dataset

# Load all surveys with specific variables
surveys = load_all_surveys('data/', target_vars=['age', 'income', 'jobsat'])

# Build harmonized long-form dataset
df_long = build_harmonized_dataset(surveys, target_vars=['age', 'income', 'jobsat'])
```

### Analyses Module (`sociopathit.analyses`)

Statistical modeling, causal inference, and machine learning. The full toolkit.

- **regress**: OLS, logistic, Poisson, multilevel regression with publication-ready output
- **pubtable**: Generate publication tables (proportions, descriptives, regression comparisons)
- **descriptive**: Correlation matrices, crosstabs with chi-square, group summaries
- **sem**: Path analysis, mediation models, indirect effects (new!)
- **causal**: Propensity scores (IPW, matching), difference-in-differences, IV, RDD
- **panel**: Fixed effects, random effects, first-difference models
- **ml**: Scikit-learn pipelines, feature importance, classification and regression models
- **text_analysis**: TF-IDF, topic modeling, text similarity, readability scores
- **network**: Create networks from edgelists, adjacency matrices, co-occurrence, correlation

**Example workflow:**
```python
from sociopathit.analyses.regress import ols
from sociopathit.analyses.sem import mediation
from sociopathit.visuals.coef import coef

# Fit regression
model = ols(df, 'outcome', ['age', 'education', 'income'])

# Test mediation
med_model = mediation(df, x='treatment', m='mediator', y='outcome')
indirect, se = med_model.indirect_effect(['treatment', 'mediator', 'outcome'])

# Visualize coefficients
coef(model.get_estimates())
```

### Visuals Module (`sociopathit.visuals`)

24 visualization types, all static and interactive. Consistent style, no compromises.

**Basic Charts:**
- **bar**: Bar charts with annotations, subplots, interactive versions
- **hist**: Histograms with trace outlines, thresholds, group coloring
- **boxplot**: Box plots, violin plots, raincloud plots with point overlays
- **pie**: Pie charts with auto-collapse of small categories
- **scatter**: Scatter plots with regression lines and confidence intervals

**Statistical Visualizations:**
- **density**: KDE plots, ridgeline plots, raincloud combinations
- **pair**: Pair plots (scatterplot matrices) with grouping
- **heatmap**: Correlation heatmaps with annotations
- **cluster**: Hierarchical clustering dendrograms and clustered heatmaps
- **coef**: Coefficient plots with confidence intervals and significance stars

**Regression & Diagnostics:**
- **margins**: Marginal effects plots for regression models
- **oddsratio**: Odds ratio forest plots for logistic models
- **residuals**: Full regression diagnostics (residuals vs fitted, Q-Q plots, scale-location)
- **ice**: Individual Conditional Expectation plots with PDP overlays
- **feature**: Feature importance waterfalls and bar charts (SHAP-style)

**Advanced Visualizations:**
- **factormap**: 2D and 3D factor maps for PCA/MCA with variance explained scree plots
- **dag**: Causal directed acyclic graphs (requires networkx)
- **cooccur**: Co-occurrence network graphs
- **trend**: Time series with smoothing, events, shaded areas
- **waterfall**: Cumulative contribution waterfalls
- **wordcloud**: Word clouds with highlighted groups and gradients

**Specialized Charts:**
- **hierarchical**: Treemaps and sunburst charts for hierarchical data
- **flow**: Sankey and alluvial diagrams for flow visualization
- **horizon**: Horizon charts for space-efficient time series
- **geographic**: Point maps, choropleths, hexbin density maps

**Example workflow:**
```python
from sociopathit.visuals.coef import coef_interactive
from sociopathit.visuals.margins import margins
from sociopathit.visuals.ice import ice

# Interactive coefficient plot
fig = coef_interactive(model.get_estimates(), style_mode='viridis')
fig.show()

# Marginal effects
margins(model, variable='education', ci=True)

# ICE plot for machine learning model
ice(ml_model, X=df, feature='age', sample_size=50, pdp=True)
```

### Utils Module (`sociopathit.utils`)

Style infrastructure and color palettes.

- **style**: Color palettes (viridis, sentiment, fiery, plainjane), plot themes, annotation helpers

---

## Testing

All modules are comprehensively tested in three test notebooks:

- **tests/utilities_test.ipynb**: All data utilities (discovery, loading, metadata, longitudinal, preparation)
- **tests/analyses_test.ipynb**: All analysis modules (regression, SEM, causal, panel, ML, text, network)
- **tests/visuals_test.ipynb**: All 24 visualization types (static and interactive)

---

## Style Philosophy

Plots follow strict aesthetic guidelines: white backgrounds, semantic color palettes, clear annotations, and absolutely no gratuitous 3D pie charts. Every visualization has both static (matplotlib/seaborn) and interactive (plotly) versions.

Available style modes:
- **viridis**: Perceptually uniform gradient (default)
- **sentiment**: Red (negative) to green (positive) semantic coloring
- **fiery**: Warm gradient for emphasis
- **plainjane**: Simple greys and blues for minimalist presentations
- **reviewer3**: Black and white gradient for your toughest reviewer

---

## ⚠️ Disclaimer

This package will not fix your model fit, your p-values, or your moral compass. But it will make your plots beautiful and your workflow reproducible.
