"""data_loader.py - Load and inspect the Gujarat company master dataset.

Follows the IT462 "Data Import & Cleaning" module: import from CSV,
data inspection, structure overview, and a column inventory that maps
each field to its measurement scale (nominal / ordinal / interval / ratio).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "EDA_Project_Dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_JSON = PROCESSED_DIR / "analysis_results.json"

# Measurement-scale map (course module: Measurement Scales)
MEASUREMENT_SCALES = {
    "CIN": "Nominal (identifier)",
    "CompanyName": "Nominal",
    "CompanyROCcode": "Nominal",
    "CompanyCategory": "Nominal",
    "CompanySubCategory": "Nominal",
    "CompanyClass": "Nominal",
    "AuthorizedCapital": "Ratio (INR)",
    "PaidupCapital": "Ratio (INR)",
    "CompanyRegistrationdate_date": "Interval (date)",
    "Registered_Office_Address": "Nominal (free text)",
    "Listingstatus": "Nominal (binary)",
    "CompanyStatus": "Nominal",
    "CompanyStateCode": "Nominal",
    "CompanyIndian/Foreign Company": "Nominal",
    "nic_code": "Nominal (coded sector)",
    "CompanyIndustrialClassification": "Nominal",
}


def get_data_paths() -> dict:
    """Return the canonical project paths as a dict of Path objects."""
    return {
        "project_root": PROJECT_ROOT,
        "raw": RAW_DATA,
        "processed_dir": PROCESSED_DIR,
        "figures": FIGURES_DIR,
        "results_json": RESULTS_JSON,
    }


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the *original* dataset exactly as supplied (read-only).

    The raw file is never modified; all cleaning happens on copies.
    """
    src = Path(path) if path is not None else RAW_DATA
    df = pd.read_csv(src, low_memory=False)
    return df


def load_cleaned(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned dataset produced by the analysis pipeline."""
    src = Path(path) if path is not None else PROCESSED_DIR / "company_master_cleaned.csv"
    df = pd.read_csv(src, low_memory=False)
    if "CompanyRegistrationdate_date" in df.columns:
        df["CompanyRegistrationdate_date"] = pd.to_datetime(
            df["CompanyRegistrationdate_date"], errors="coerce"
        )
    return df


def basic_overview(df: pd.DataFrame) -> dict:
    """High-level structure overview of a dataframe."""
    dtypes = df.dtypes.astype(str)
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
        "n_numeric": int(dtypes.isin(["int64", "float64"]).sum()),
        "n_object": int((dtypes == "object").sum()),
        "n_datetime": int(dtypes.str.startswith("datetime").sum()),
        "n_duplicate_rows": int(df.duplicated().sum()),
        "n_total_missing_cells": int(df.isna().sum().sum()),
        "pct_missing_cells": round(float(df.isna().sum().sum()) / df.size * 100, 2),
    }


def column_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, role, scale, uniques and missing counts."""
    rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        if pd.api.types.is_numeric_dtype(s) and col not in ("nic_code",):
            role = "Numeric"
        elif col == "nic_code":
            role = "Coded category"
        elif "date" in col.lower():
            role = "Datetime"
        else:
            role = "Categorical / text"
        rows.append(
            {
                "column": col,
                "dtype": dtype,
                "role": role,
                "measurement_scale": MEASUREMENT_SCALES.get(col, "-"),
                "n_unique": int(s.nunique(dropna=True)),
                "n_missing": int(s.isna().sum()),
                "pct_missing": round(float(s.isna().mean() * 100), 3),
            }
        )
    return pd.DataFrame(rows)


def quick_head(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Preview helper used by the website and notebook."""
    return df.head(n)
