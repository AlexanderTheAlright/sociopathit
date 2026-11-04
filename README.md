# Sociopath-it

For those with ethics but not morals.

Sociopath-it is a Python package for social scientists who work with survey data. It loads and harmonizes survey files, runs common analyses (regression, panel, causal, SEM, text), and draws figures in one house style.

## Installation

```bash
pip install git+https://github.com/AlexanderTheAlright/sociopathit.git
```

The core install covers data handling, the analyses and the figures. Some functions need extra libraries, grouped as pip extras:

| Extra | Installs | Used by |
|:--|:--|:--|
| `text` | wordcloud, nltk | `visuals.wordcloud`, `analyses.text_analysis` (NLTK tokenizing and stopwords) |
| `network` | networkx | `visuals.dag`, `visuals.cooccur` |
| `labels` | adjustText | label placement in `visuals.factormap` |
| `excel` | openpyxl | the metaset workbook in `cleansing.harmonize` |
| `stats` | pingouin | reliability statistics in `cleansing.harmonize` |
| `ml` | shap | `analyses.ml.shap_analysis` |
| `panel` | linearmodels | no function yet; the panel models use statsmodels |
| `all` | all of the above | |

```bash
pip install "sociopathit[text] @ git+https://github.com/AlexanderTheAlright/sociopathit.git"
pip install "sociopathit[all] @ git+https://github.com/AlexanderTheAlright/sociopathit.git"
```

When an extra is missing, the function that needs it raises an `ImportError`; for `labels`, `excel`, `network` and `text` the message names the extra to install.

## Quick start

```python
import numpy as np
import pandas as pd
from sociopathit.analyses.regress import ols
from sociopathit.visuals.coef import coef
from sociopathit.visuals.bar import bar
from sociopathit.utils.style import save_figure

rng = np.random.default_rng(1)
df = pd.DataFrame({
    "age": rng.integers(20, 65, 300),
    "education": rng.integers(1, 6, 300),
    "income": rng.normal(50, 15, 300),
})
df["jobsat"] = 2 + 0.01 * df["age"] + 0.2 * df["education"] + rng.normal(0, 1, 300)

# OLS with HC1 standard errors; get_tidy() returns term, estimate, conf.low, conf.high, p.value
model = ols(df, "jobsat", ["age", "education", "income"])
print(model.summary())
estimates = model.get_tidy()

# Coefficient plot, saved as coef_jobsat.png and coef_jobsat.svg
fig, ax = coef(estimates, p_col="p.value", title="Predictors of job satisfaction")
save_figure(fig, "coef_jobsat")

# Grouped bars in black and white: one column per group
shares = pd.DataFrame({"wave": ["2023", "2024", "2025"], "value": [0, 0, 0],
                       "Canada": [41.2, 45.0, 52.5], "United States": [35.5, 38.0, 40.8]})
fig, ax = bar(shares, x="wave", y="value", orientation="grouped",
              title="Share satisfied, by country", style_mode="reviewer3")
save_figure(fig, "bar_reviewer3")
```

Static plotting functions return `(fig, ax)`. Functions ending in `_interactive` return a Plotly figure.

## Modules

### `sociopathit.data`

| Module | What it does |
|:--|:--|
| `discovery` | Finds data files and identifies waves from file names (`discover_data`, `find_latest_wave`). |
| `loading` | Loads Stata files and folders of surveys with ID detection (`load_stata`, `load_all_surveys`, `load_and_combine`). |
| `metadata` | Summarizes surveys and extracts wave and ID variables (`summarize_surveys`, `extract_wave_info`, `get_id_vars`). |
| `longitudinal` | Detects panel structure, aligns waves and builds long-form data (`detect_longitudinal`, `align_longitudinal_data`, `build_longit_from_dir`). |
| `preparation` | Harmonizes column names and codes and prepares data for models (`harmonize_columns`, `build_harmonized_dataset`, `prepare_for_analysis`). |

### `sociopathit.cleansing`

| Module | What it does |
|:--|:--|
| `harmonize` | Metaset-driven harmonization of a folder of survey files into processed datasets (`discover_datasets`, `load_dataset`, `harmonize_all`). |

### `sociopathit.analyses`

| Module | What it does |
|:--|:--|
| `regress` | OLS, logit, Poisson and multilevel models with survey weights and HC1 standard errors (`ols`, `logit`, `poisson`, `multilevel`, `regress`, `compare_models`, `marginal_effects`). |
| `pubtable` | HTML tables of proportions, descriptives and regression results (`proportion_table`, `descriptive_table`, `regression_table`, `save_table`). |
| `descriptive` | Correlation matrices, crosstabs with chi-square, grouped summaries and normality tests (`correlation_matrix`, `crosstab`, `group_summary`, `distribution_test`). |
| `sem` | Path analysis, mediation and CFA (`path_analysis`, `mediation`, `cfa`). |
| `causal` | Propensity scores, difference-in-differences, IV and regression discontinuity (`propensity_score`, `difference_in_differences`, `instrumental_variables`, `regression_discontinuity`). |
| `panel` | Fixed effects, random effects and first differences, with the Hausman test (`fixed_effects`, `random_effects`, `first_differences`, `hausman_test`). |
| `ml` | scikit-learn pipelines, training with cross-validation and feature importance (`build_pipeline`, `train_model`, `feature_importance`, `shap_analysis`). |
| `model_io` | Saves and loads model results as JSON or CSV, including CSV for R (`save_model`, `load_model`, `export_for_r`). |
| `text_analysis` | Text cleaning, readability, similarity, TF-IDF, n-grams and sentiment (`clean_text`, `complexity_scores`, `create_tfidf_matrix`, `analyze_sentiment`, `analyze_corpus`). |
| `network` | Edge lists, adjacency matrices, co-occurrence and bipartite networks (`create_edgelist`, `create_adjacency_matrix`, `cooccurrence_network`, `network_summary`). |

### `sociopathit.visuals`

Each module has a static matplotlib function and, in most cases, an `_interactive` Plotly version.

| Module | What it draws |
|:--|:--|
| `bar` | Vertical, horizontal, stacked and grouped bars; faceted bars (`bar`, `bar_subplots`). |
| `hist` | Histograms with optional group colouring and threshold lines (`histogram`). |
| `boxplot` | Box and violin plots (`boxplot`, `boxplot_subplots`). |
| `density` | Kernel densities, ridgelines and raincloud plots (`kde`, `ridgeline`, `raincloud`, `ryder`). |
| `scatter` | Scatter plots with linear or LOWESS fits and a 95% prediction band (`scatterplot`). |
| `trend` | Lines or bars over time, with event lines and shading (`trend`). |
| `pie` | Pie charts that collapse small categories (`pie`). |
| `pair` | Scatterplot matrices (`pair`). |
| `heatmap` | Correlation and distribution heatmaps (`heatmap`, `heatmap_distribution`). |
| `cluster` | Hierarchical clustering dendrograms and clustered heatmaps (`cluster`, `heatmap_cluster`). |
| `coef` | Coefficient plots with confidence intervals and significance stars (`coef`). |
| `oddsratio` | Odds ratio forest plots for logistic models (`oddsratio`, `oddsratio_comparison`). |
| `margins` | Predicted values across a predictor, overall or by group (`margins`, `margins_comparison`). |
| `residuals` | Regression diagnostic panels and residual histograms (`residuals`, `residuals_histogram`). |
| `ice` | Individual conditional expectation curves with a partial dependence line (`ice`, `ice_comparison`). |
| `feature` | Feature importance bars and waterfalls (`feature_importance_bar`, `feature_waterfall`). |
| `factormap` | Factor maps for PCA, MCA or CA in 2D and 3D, and scree plots (`factormap`, `factormap_3d`, `variance_explained`). |
| `apc` | Age-period-cohort heatmaps (`apc_heatmap`). |
| `dag` | Causal DAGs (`dag`, `dag_from_formula`). Needs `network`. |
| `cooccur` | Word co-occurrence networks (`cooccur`). Needs `network`. |
| `waterfall` | Waterfall charts of cumulative change (`waterfall`). |
| `wordcloud` | Word clouds with highlighted groups (`wordcloud`). Needs `text`. |
| `hierarchical` | Treemaps and sunbursts (`treemap`, `sunburst`). |
| `flow` | Sankey and alluvial diagrams (`sankey`, `alluvial`). |
| `horizon` | Horizon charts for many time series (`horizon`, `horizon_panel`). |
| `geographic` | Point maps, choropleths and hexbin maps (`point_map`, `choropleth`, `hexbin_map`). |

### `sociopathit.utils`

| Module | What it does |
|:--|:--|
| `style` | Style modes, palettes, titles, the shared legend below the axes, and `save_figure` (PNG and SVG). |
| `optional` | Imports optional libraries on demand and names the pip extra when one is missing. |

## Style modes

Every plotting function takes `style_mode`. The default is `"viridis"`.

| Mode | Look | Use it for |
|:--|:--|:--|
| `viridis` | Viridis colours, perceptually uniform | Default colour figures |
| `reviewer3` | Black, white and greys, with hatching on bars | Print journals and black-and-white submissions |
| `sentiment` | Green for positive, red for negative | Scales with a good and a bad end |
| `fiery` | Reds, oranges and purples | High-contrast colour figures |
| `plainjane` | Light red and blue | Two-sided contrasts in plain colours |

```python
fig, ax = bar(shares, x="wave", y="value", orientation="grouped", style_mode="reviewer3")
```

Figures have white backgrounds, a title centred over the plot, and one legend centred below the axes. `save_figure(fig, "name")` writes `name.png` (300 dpi) and `name.svg`.

## Tests

```bash
pip install pytest
pytest -q
```

The smoke tests in `tests/` call every public function in `sociopathit.visuals` and `sociopathit.analyses` on small synthetic data. Functions whose optional extra is not installed are skipped.

## License

MIT. See `LICENSE`.
