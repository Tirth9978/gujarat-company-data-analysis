"""statistical_analysis.py - Descriptive statistics, association tests,
correlation, regression fitting, outlier detection and time-series EDA.

Implements the IT462 modules: Descriptive Summary, Univariate & Bivariate
Analysis, Correlation Analysis & Data fitting, Missing Data Analysis and
Advanced EDA (trend / seasonality).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


# ---------------------------------------------------------------------------
# Descriptive summary (measures of centre, spread, shape)
# ---------------------------------------------------------------------------
def five_number_summary(s: pd.Series) -> dict:
    """Tukey's five-number summary + IQR (course textbook: J.W. Tukey)."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {}
    q1, med, q3 = x.quantile([0.25, 0.50, 0.75])
    return {
        "min": float(x.min()), "q1": float(q1), "median": float(med),
        "q3": float(q3), "max": float(x.max()), "iqr": float(q3 - q1),
    }


def descriptive_stats(s: pd.Series) -> dict:
    """Full descriptive summary: centre, spread, shape (skew/kurtosis)."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {}
    q1, med, q3 = x.quantile([0.25, 0.50, 0.75])
    mean = float(x.mean())
    sd = float(x.std())
    mad = float((x - med).abs().median())                     # Median Abs. Deviation
    modes = x.mode().tolist()[:3]
    return {
        "count": int(x.count()),
        "mean": mean,
        "median": float(med),
        "mode": modes,
        "std": sd,
        "variance": float(x.var()),
        "iqr": float(q3 - q1),
        "mad": mad,
        "cv_pct": round(float(sd / mean * 100), 2) if mean else None,
        "skewness": round(float(x.skew()), 3),
        "kurtosis": round(float(x.kurtosis()), 3),
        "p01": float(x.quantile(0.01)), "p05": float(x.quantile(0.05)),
        "p25": float(q1), "p50": float(med), "p75": float(q3),
        "p95": float(x.quantile(0.95)), "p99": float(x.quantile(0.99)),
        "min": float(x.min()), "max": float(x.max()),
    }


def categorical_distribution(s: pd.Series, top: int = 20) -> dict:
    """Frequency table (counts + percentages) for a categorical field."""
    vc = s.value_counts(dropna=True)
    out = vc.head(top).to_dict()
    return {str(k): {"count": int(v), "pct": round(float(v / s.notna().sum() * 100), 2)}
            for k, v in out.items()}


def group_statistics(df: pd.DataFrame, group_col: str, value_col: str,
                     agg: str = "median") -> pd.DataFrame:
    """Grouped aggregate table (e.g. median capital by CompanyClass)."""
    g = df.groupby(group_col, dropna=True)[value_col]
    out = g.agg(n="count", mean="mean", median="median",
                std="std", max="max").reset_index()
    out = out.round(2)
    return out.sort_values(agg, ascending=False)


# ---------------------------------------------------------------------------
# Missing-data mechanisms (MCAR / MAR / MNAR evidence)
# ---------------------------------------------------------------------------
def missingness_by_group(df: pd.DataFrame, group_col: str,
                         flag_col: str = "block_missing") -> pd.DataFrame:
    """Share of rows with block-missing details within each group."""
    rates = df.groupby(group_col, dropna=False)[flag_col].mean().mul(100).round(2)
    counts = df.groupby(group_col, dropna=False)[flag_col].size()
    return pd.DataFrame({"pct_block_missing": rates, "n": counts}).sort_values(
        "pct_block_missing", ascending=False)


def chi2_test(table: pd.DataFrame) -> dict:
    """Chi-square test of independence + Cramer's V for a crosstab."""
    chi2, p, dof, _ = sps.chi2_contingency(table.values)
    n = table.values.sum()
    r, k = table.shape
    cramers_v = float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if min(r, k) > 1 else np.nan
    return {"chi2": round(float(chi2), 2), "p_value": float(p), "dof": int(dof),
            "cramers_v": round(cramers_v, 4),
            "significant_5pct": bool(p < 0.05)}


def missingness_mechanism_report(df: pd.DataFrame) -> dict:
    """Evidence-based discussion of the missingness mechanism.

    Compares block-missingness across eras and statuses; a significant
    association with an observed variable rules out pure MCAR.
    """
    rep = {}
    era = pd.cut(df["RegistrationYear"], bins=[1850, 1990, 2000, 2010, 2015, 2027],
                 labels=["1850-1990", "1991-2000", "2001-2010", "2011-2015", "2016-2026"])
    t_era = pd.crosstab(era, df["block_missing"])
    rep["by_era"] = missingness_by_group(df.assign(Era=era), "Era").to_dict(orient="index")
    rep["chi2_era"] = chi2_test(t_era)
    t_status = pd.crosstab(df["CompanyStatusGroup"], df["block_missing"])
    rep["by_status"] = missingness_by_group(df, "CompanyStatusGroup").to_dict(orient="index")
    rep["chi2_status"] = chi2_test(t_status)
    return rep


# ---------------------------------------------------------------------------
# Association: crosstab, correlation, regression fit
# ---------------------------------------------------------------------------
def crosstab_analysis(df: pd.DataFrame, col1: str, col2: str,
                      normalize: bool = False) -> dict:
    """Crosstab (+ chi-square) between two categorical variables."""
    ct = pd.crosstab(df[col1], df[col2])
    res = {"table": ct,
           "chi2": chi2_test(ct)}
    if normalize:
        res["row_pct"] = pd.crosstab(df[col1], df[col2], normalize="index").mul(100).round(2)
    return res


def correlation_analysis(df: pd.DataFrame, cols: list[str],
                         use_log: bool = True) -> dict:
    """Pearson & Spearman correlation matrices for the given columns.

    Capitals are log10-transformed by default because raw values span
    more than six orders of magnitude (course: Correlation Analysis).
    """
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if use_log:
        sub = np.log10(sub.where(sub > 0))  # log10 undefined at 0 -> NaN
        sub = sub.dropna()
    pear = sub.corr(method="pearson").round(4)
    spear = sub.corr(method="spearman").round(4)
    pear_p = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i < j:
                r, p = sps.pearsonr(sub[a], sub[b])
                pear_p.loc[a, b] = pear_p.loc[b, a] = p
    return {"pearson": pear, "spearman": spear,
            "pearson_pvals": pear_p, "n_used": int(len(sub))}


def linear_fit(x: pd.Series, y: pd.Series, log10: bool = True) -> dict:
    """Simple OLS fit y ~ x (optionally in log10 space) + goodness of fit."""
    d = pd.concat([pd.to_numeric(x, errors="coerce"),
                   pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    xv, yv = d.iloc[:, 0].values.astype(float), d.iloc[:, 1].values.astype(float)
    if log10:
        keep = (xv > 0) & (yv > 0)
        xv, yv = np.log10(xv[keep]), np.log10(yv[keep])
    slope, intercept, r, p, se = sps.linregress(xv, yv)
    yhat = intercept + slope * xv
    rmse = float(np.sqrt(np.mean((yv - yhat) ** 2)))
    return {"slope": round(float(slope), 4), "intercept": round(float(intercept), 4),
            "r": round(float(r), 4), "r2": round(float(r) ** 2, 4),
            "p_value": float(p), "std_err": round(float(se), 5),
            "rmse_log10": round(rmse, 4), "n": int(len(xv)),
            "space": "log10" if log10 else "raw"}


# ---------------------------------------------------------------------------
# Outliers & concentration
# ---------------------------------------------------------------------------
def iqr_outliers(s: pd.Series, log10: bool = True) -> dict:
    """Tukey fence (1.5 x IQR) outlier detection, optionally on log10."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if log10:
        x = np.log10(x[x > 0])
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (x < lo) | (x > hi)
    return {"lower_fence": float(10 ** lo) if log10 else float(lo),
            "upper_fence": float(10 ** hi) if log10 else float(hi),
            "n_outliers": int(mask.sum()),
            "pct_outliers": round(float(mask.mean() * 100), 2),
            "space": "log10" if log10 else "raw"}


def zscore_outliers(s: pd.Series, threshold: float = 3.0, log10: bool = True) -> dict:
    """Z-score outliers (|z| > threshold), optionally on log10 values."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    if log10:
        x = np.log10(x[x > 0])
    z = (x - x.mean()) / x.std()
    mask = z.abs() > threshold
    return {"threshold": threshold, "n_outliers": int(mask.sum()),
            "pct_outliers": round(float(mask.mean() * 100), 2),
            "space": "log10" if log10 else "raw"}


def concentration_profile(s: pd.Series, top_pct: float = 1.0) -> dict:
    """Capital concentration: top-x% share, Gini, Lorenz points (Pareto)."""
    x = pd.to_numeric(s, errors="coerce").dropna()
    x = x[x > 0].sort_values()
    total = float(x.sum())
    n = len(x)
    k = max(int(round(n * top_pct / 100)), 1)
    top_share = float(x.iloc[-k:].sum() / total * 100)
    cum = x.cumsum() / total
    lorenz_y = np.concatenate([[0.0], cum.values])
    lorenz_x = np.linspace(0, 1, len(lorenz_y))
    gini = float(1 - 2 * np.trapezoid(lorenz_y, lorenz_x))
    return {"top_pct": top_pct, "top_share_pct": round(top_share, 2),
            "gini": round(gini, 4), "n": n,
            "lorenz_x": lorenz_x.tolist(), "lorenz_y": lorenz_y.tolist(),
            "median_top": float(x.iloc[-k:].median())}


# ---------------------------------------------------------------------------
# Time-series EDA (trend + seasonality)
# ---------------------------------------------------------------------------
def yearly_registrations(df: pd.DataFrame,
                         year_col: str = "RegistrationYear") -> pd.Series:
    """Company registrations per year (1890-2026)."""
    return df[year_col].dropna().astype(int).value_counts().sort_index()


def monthly_registrations(df: pd.DataFrame,
                          month_col: str = "RegistrationMonth") -> pd.Series:
    """Average registrations per calendar month (seasonality check)."""
    return df[month_col].dropna().astype(int).value_counts().sort_index()


def trend_summary(year_counts: pd.Series, recent_from: int = 2015,
                  end_year: int | None = None) -> dict:
    """Peak year, recent-window CAGR and decade totals (trend detection).

    ``end_year`` lets the caller exclude a partial final year (the extract
    snapshot covers 2026 only through June), so CAGR compares full years.
    """
    peak_year = int(year_counts.idxmax())
    peak_n = int(year_counts.max())
    end = int(end_year) if end_year is not None else int(year_counts.index.max())
    base = year_counts.get(recent_from, np.nan)
    last = year_counts.get(end, np.nan)
    n_years = end - recent_from
    cagr = float(((last / base) ** (1 / n_years) - 1) * 100) if base and base > 0 and n_years > 0 else None
    decades = (year_counts.groupby((year_counts.index // 10) * 10).sum())
    return {"peak_year": peak_year, "peak_n": peak_n,
            "cagr_end_year": end,
            f"cagr_{recent_from}_to_{end}_pct": round(cagr, 2) if cagr else None,
            "decade_totals": {int(k): int(v) for k, v in decades.items()}}
