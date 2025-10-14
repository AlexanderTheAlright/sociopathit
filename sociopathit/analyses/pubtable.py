"""
pubtable.py  Sociopath-it Publication Tables Module
----------------------------------------------------
Create publication-ready HTML tables with clean, minimal formatting.

Features:
- Proportion tables with confidence intervals
- Descriptive statistics tables
- Regression coefficient tables with significance stars
- Multilevel model support
- Proper decimal places and formatting
- Export to HTML with copy-to-Excel capability

Styling philosophy:
- Minimal, clean design
- Publication standards (APA/ASA)
- 2-3 decimal places max
- Clear significance indicators
- Responsive HTML layout
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# HELPER FUNCTIONS
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def _format_number(x: float, decimals: int = 2) -> str:
    """Format number with specified decimal places."""
    if pd.isna(x):
        return ""
    return f"{x:.{decimals}f}"


def _format_pvalue(p: float) -> str:
    """Format p-value following publication standards."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "<.001"
    elif p < 0.01:
        return f"{p:.3f}"
    else:
        return f"{p:.2f}"


def _significance_stars(p: float) -> str:
    """Return significance stars for p-value."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""


def _ci_string(lower: float, upper: float, decimals: int = 2) -> str:
    """Format confidence interval as [lower, upper]."""
    if pd.isna(lower) or pd.isna(upper):
        return ""
    return f"[{lower:.{decimals}f}, {upper:.{decimals}f}]"


def _apply_table_style() -> str:
    """Return CSS styling for publication tables."""
    return """
    <style>
        .pubtable {
            font-family: 'Times New Roman', Times, serif;
            border-collapse: collapse;
            margin: 20px auto;
            font-size: 11pt;
            width: auto;
            max-width: 100%;
        }
        .pubtable th {
            border-top: 2px solid #000;
            border-bottom: 1px solid #000;
            padding: 8px 12px;
            text-align: left;
            font-weight: bold;
        }
        .pubtable td {
            padding: 6px 12px;
            text-align: left;
        }
        .pubtable tbody tr:last-child td {
            border-bottom: 2px solid #000;
        }
        .pubtable .number {
            text-align: right;
            font-feature-settings: 'tnum';
        }
        .pubtable caption {
            caption-side: top;
            text-align: left;
            font-weight: bold;
            padding-bottom: 10px;
            font-size: 12pt;
        }
        .pubtable .note {
            font-size: 9pt;
            font-style: italic;
            padding-top: 5px;
            border-top: none;
        }
        .pubtable .indent {
            padding-left: 24px;
        }
    </style>
    """


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# PROPORTION TABLES
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def proportion_table(
    df: pd.DataFrame,
    row_var: str,
    col_var: Optional[str] = None,
    weight_var: Optional[str] = None,
    ci: bool = True,
    ci_level: float = 0.95,
    decimals: int = 1,
    title: str = "Proportion Table",
    show_n: bool = True,
) -> str:
    """
    Create publication-ready proportion table with confidence intervals.

    Parameters
    ----------
    df : DataFrame
        Input data.
    row_var : str
        Variable for table rows.
    col_var : str, optional
        Variable for table columns (for cross-tabulation).
    weight_var : str, optional
        Survey weight variable.
    ci : bool, default True
        Include confidence intervals.
    ci_level : float, default 0.95
        Confidence level (0-1).
    decimals : int, default 1
        Decimal places for percentages.
    title : str
        Table title.
    show_n : bool, default True
        Show sample sizes.

    Returns
    -------
    str
        HTML table string.

    Examples
    --------
    >>> df = pd.DataFrame({'gender': ['M', 'F', 'M', 'F'], 'response': ['Yes', 'No', 'Yes', 'Yes']})
    >>> html = proportion_table(df, row_var='response', col_var='gender', title='Response by Gender')
    """
    # Handle weights
    if weight_var:
        weights = df[weight_var]
    else:
        weights = np.ones(len(df))

    # Simple frequency table (one-way)
    if col_var is None:
        counts = df.groupby(row_var).size()
        weighted_counts = df.groupby(row_var).apply(lambda x: weights[x.index].sum())
        total = weighted_counts.sum()
        props = weighted_counts / total * 100

        # Confidence intervals using Wilson score interval
        z = stats.norm.ppf(1 - (1 - ci_level) / 2)
        results = []

        for cat in props.index:
            p = props[cat] / 100
            n = counts[cat]

            if ci and n > 0:
                # Wilson score interval
                denom = 1 + z**2 / n
                center = (p + z**2 / (2*n)) / denom
                margin = z * np.sqrt(p * (1-p) / n + z**2 / (4*n**2)) / denom
                ci_lower = max(0, (center - margin) * 100)
                ci_upper = min(100, (center + margin) * 100)
                ci_str = _ci_string(ci_lower, ci_upper, decimals)
            else:
                ci_str = ""

            results.append({
                row_var: cat,
                "%": _format_number(props[cat], decimals),
                "95% CI": ci_str if ci else None,
                "n": int(counts[cat]) if show_n else None
            })

        result_df = pd.DataFrame(results)
        if not ci:
            result_df = result_df.drop(columns=["95% CI"])
        if not show_n:
            result_df = result_df.drop(columns=["n"])

    # Cross-tabulation (two-way)
    else:
        crosstab = pd.crosstab(
            df[row_var],
            df[col_var],
            values=weights,
            aggfunc='sum',
            normalize='columns'
        ) * 100

        result_df = crosstab.round(decimals)
        result_df = result_df.reset_index()

    # Generate HTML
    html = _apply_table_style()
    html += f'<table class="pubtable">'
    html += f'<caption>{title}</caption>'
    html += '<thead><tr>'

    for col in result_df.columns:
        html += f'<th class="{"number" if col != row_var and col != col_var else ""}">{col}</th>'
    html += '</tr></thead><tbody>'

    for _, row in result_df.iterrows():
        html += '<tr>'
        for i, (col, val) in enumerate(row.items()):
            css_class = "number" if i > 0 else ""
            html += f'<td class="{css_class}">{val}</td>'
        html += '</tr>'

    html += '</tbody></table>'

    # Add note about significance
    if ci:
        html += f'<p class="note" style="margin-left: 20px; font-size: 9pt; font-style: italic;">Note: {int(ci_level*100)}% confidence intervals shown.</p>'

    return html


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# DESCRIPTIVE STATISTICS TABLES
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def descriptive_table(
    df: pd.DataFrame,
    variables: List[str],
    group_var: Optional[str] = None,
    weight_var: Optional[str] = None,
    stats: List[str] = ["mean", "sd", "min", "max"],
    decimals: int = 2,
    title: str = "Descriptive Statistics",
    var_labels: Optional[Dict[str, str]] = None,
) -> str:
    """
    Create publication-ready descriptive statistics table.

    Parameters
    ----------
    df : DataFrame
        Input data.
    variables : list of str
        Variables to describe.
    group_var : str, optional
        Variable to group by.
    weight_var : str, optional
        Survey weight variable.
    stats : list of str, default ["mean", "sd", "min", "max"]
        Statistics to compute: "mean", "sd", "median", "min", "max", "n".
    decimals : int, default 2
        Decimal places.
    title : str
        Table title.
    var_labels : dict, optional
        Mapping of variable names to display labels.

    Returns
    -------
    str
        HTML table string.

    Examples
    --------
    >>> df = pd.DataFrame({'age': [25, 30, 35], 'income': [50000, 60000, 70000]})
    >>> html = descriptive_table(df, variables=['age', 'income'])
    """
    def weighted_stat(series, weights, stat_name):
        """Calculate weighted statistic."""
        if weights is None:
            weights = np.ones(len(series))

        mask = ~series.isna()
        s = series[mask]
        w = weights[mask]

        if len(s) == 0:
            return np.nan

        if stat_name == "mean":
            return np.average(s, weights=w)
        elif stat_name == "sd":
            mean = np.average(s, weights=w)
            variance = np.average((s - mean)**2, weights=w)
            return np.sqrt(variance)
        elif stat_name == "median":
            return np.median(s)  # Weighted median is complex, using unweighted
        elif stat_name == "min":
            return s.min()
        elif stat_name == "max":
            return s.max()
        elif stat_name == "n":
            return len(s)

    # Get weights
    weights = df[weight_var] if weight_var else None

    results = []

    if group_var is None:
        # Overall statistics
        for var in variables:
            row = {"Variable": var_labels.get(var, var) if var_labels else var}
            for stat_name in stats:
                val = weighted_stat(df[var], weights, stat_name)
                if stat_name == "n":
                    row[stat_name.upper()] = int(val) if not pd.isna(val) else ""
                else:
                    row[stat_name.capitalize()] = _format_number(val, decimals)
            results.append(row)
    else:
        # Grouped statistics
        for var in variables:
            var_label = var_labels.get(var, var) if var_labels else var
            for i, (group, group_df) in enumerate(df.groupby(group_var)):
                row = {"Variable": var_label if i == 0 else "", "Group": str(group)}
                group_weights = weights[group_df.index] if weights is not None else None

                for stat_name in stats:
                    val = weighted_stat(group_df[var], group_weights, stat_name)
                    if stat_name == "n":
                        row[stat_name.upper()] = int(val) if not pd.isna(val) else ""
                    else:
                        row[stat_name.capitalize()] = _format_number(val, decimals)
                results.append(row)

    result_df = pd.DataFrame(results)

    # Generate HTML
    html = _apply_table_style()
    html += f'<table class="pubtable">'
    html += f'<caption>{title}</caption>'
    html += '<thead><tr>'

    for col in result_df.columns:
        html += f'<th class="{"number" if col not in ["Variable", "Group"] else ""}">{col}</th>'
    html += '</tr></thead><tbody>'

    for _, row in result_df.iterrows():
        html += '<tr>'
        for col, val in row.items():
            css_class = "number" if col not in ["Variable", "Group"] else ""
            if col == "Variable" and val == "":
                css_class += " indent"
            html += f'<td class="{css_class}">{val}</td>'
        html += '</tr>'

    html += '</tbody></table>'

    if weight_var:
        html += f'<p class="note" style="margin-left: 20px; font-size: 9pt; font-style: italic;">Note: Statistics weighted by {weight_var}.</p>'

    return html


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# REGRESSION COEFFICIENT TABLES
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def regression_table(
    models: Union[pd.DataFrame, List[pd.DataFrame]],
    model_names: Optional[List[str]] = None,
    decimals: int = 3,
    title: str = "Regression Results",
    show_se: bool = True,
    show_ci: bool = False,
    show_stars: bool = True,
    var_labels: Optional[Dict[str, str]] = None,
    coef_order: Optional[List[str]] = None,
    stats_rows: Optional[Dict[str, List]] = None,
) -> str:
    """
    Create publication-ready regression coefficient table.

    Parameters
    ----------
    models : DataFrame or list of DataFrame
        Model results. Each DataFrame should have columns:
        ['term', 'estimate', 'std.error', 'statistic', 'p.value']
        Or ['term', 'estimate', 'conf.low', 'conf.high', 'p.value']
    model_names : list of str, optional
        Names for each model column.
    decimals : int, default 3
        Decimal places for coefficients.
    title : str
        Table title.
    show_se : bool, default True
        Show standard errors in parentheses below coefficients.
    show_ci : bool, default False
        Show confidence intervals instead of standard errors.
    show_stars : bool, default True
        Add significance stars to coefficients.
    var_labels : dict, optional
        Mapping of term names to display labels.
    coef_order : list of str, optional
        Order to display coefficients.
    stats_rows : dict, optional
        Additional statistics rows, e.g., {'N': [100, 150], 'R²': [0.45, 0.52]}.

    Returns
    -------
    str
        HTML table string.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> df = pd.DataFrame({'y': [1, 2, 3], 'x': [1, 2, 3]})
    >>> model = sm.OLS(df['y'], sm.add_constant(df['x'])).fit()
    >>> results_df = pd.DataFrame({
    ...     'term': model.params.index,
    ...     'estimate': model.params.values,
    ...     'std.error': model.bse.values,
    ...     'p.value': model.pvalues.values
    ... })
    >>> html = regression_table(results_df, title='OLS Results')
    """
    # Normalize to list of DataFrames
    if isinstance(models, pd.DataFrame):
        models = [models]

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]

    # Get all unique terms
    all_terms = []
    for model_df in models:
        for term in model_df['term']:
            if term not in all_terms:
                all_terms.append(term)

    # Apply custom order if provided
    if coef_order:
        ordered_terms = [t for t in coef_order if t in all_terms]
        remaining = [t for t in all_terms if t not in coef_order]
        all_terms = ordered_terms + remaining

    # Build result rows
    rows = []
    for term in all_terms:
        # Coefficient row
        coef_row = {"": var_labels.get(term, term) if var_labels else term}

        for i, model_df in enumerate(models):
            term_data = model_df[model_df['term'] == term]

            if term_data.empty:
                coef_row[model_names[i]] = ""
            else:
                est = term_data['estimate'].iloc[0]
                p = term_data['p.value'].iloc[0]

                # Format coefficient with stars
                coef_str = _format_number(est, decimals)
                if show_stars:
                    coef_str += _significance_stars(p)

                coef_row[model_names[i]] = coef_str

        rows.append(coef_row)

        # Standard error or CI row
        if show_se or show_ci:
            se_row = {"": ""}
            for i, model_df in enumerate(models):
                term_data = model_df[model_df['term'] == term]

                if term_data.empty:
                    se_row[model_names[i]] = ""
                else:
                    if show_ci and 'conf.low' in term_data.columns:
                        ci_low = term_data['conf.low'].iloc[0]
                        ci_high = term_data['conf.high'].iloc[0]
                        se_row[model_names[i]] = _ci_string(ci_low, ci_high, decimals)
                    elif show_se and 'std.error' in term_data.columns:
                        se = term_data['std.error'].iloc[0]
                        se_row[model_names[i]] = f"({_format_number(se, decimals)})"
                    else:
                        se_row[model_names[i]] = ""

            rows.append(se_row)

    result_df = pd.DataFrame(rows)

    # Add statistics rows if provided
    if stats_rows:
        for stat_name, stat_values in stats_rows.items():
            stat_row = {"": stat_name}
            for i, val in enumerate(stat_values):
                stat_row[model_names[i]] = _format_number(val, 2) if isinstance(val, (int, float)) else str(val)
            result_df = pd.concat([result_df, pd.DataFrame([stat_row])], ignore_index=True)

    # Generate HTML
    html = _apply_table_style()
    html += f'<table class="pubtable">'
    html += f'<caption>{title}</caption>'
    html += '<thead><tr>'

    for col in result_df.columns:
        html += f'<th class="{"number" if col != "" else ""}">{col}</th>'
    html += '</tr></thead><tbody>'

    for idx, row in result_df.iterrows():
        html += '<tr>'
        for i, (col, val) in enumerate(row.items()):
            css_class = "number" if i > 0 else ""
            # Indent SE/CI rows
            if i == 0 and val == "":
                css_class += " indent"
            html += f'<td class="{css_class}">{val}</td>'
        html += '</tr>'

    html += '</tbody></table>'

    if show_stars:
        html += '<p class="note" style="margin-left: 20px; font-size: 9pt; font-style: italic;">Note: *** p<.001, ** p<.01, * p<.05</p>'

    return html


# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
# SAVE TO FILE
# PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

def save_table(html: str, filepath: str):
    """
    Save HTML table to file.

    Parameters
    ----------
    html : str
        HTML string.
    filepath : str
        Output file path (.html).
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('<!DOCTYPE html>\n<html>\n<head>\n<meta charset="UTF-8">\n</head>\n<body>\n')
        f.write(html)
        f.write('\n</body>\n</html>')

    print(f"Table saved to: {filepath}")
