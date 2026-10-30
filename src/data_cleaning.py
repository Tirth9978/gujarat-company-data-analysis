"""data_cleaning.py - Data-quality diagnostics and reproducible cleaning.

Implements the IT462 "Data Import & Cleaning" and "Missing Data Analysis"
modules: duplicate detection, invalid entries, inconsistent formats,
type conversion, missingness profiling (MCAR/MAR/MNAR evidence) and a
configurable cleaning pipeline that never mutates the raw data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants describing known quality issues found during inspection
# ---------------------------------------------------------------------------
SENTINEL_YEARS = (1111,)          # placeholder dates like 1111-01-01
BLOCK_MISSING_COLS = [
    "CompanyCategory", "CompanyClass", "CompanySubCategory",
    "AuthorizedCapital", "PaidupCapital", "Listingstatus", "nic_code",
]
CORE_IDENT_COLS = ["CIN", "CompanyName", "CompanyROCcode", "CompanyStatus",
                   "CompanyRegistrationdate_date", "CompanyStateCode"]

# NIC 2008 two-digit section mapping (International Standard Industrial
# Classification of All Economic Activities, Revision 4 - used by MCA).
NIC_SECTIONS = {
    "A": ("01", "03", "Agriculture, forestry & fishing"),
    "B": ("05", "09", "Mining & quarrying"),
    "C": ("10", "33", "Manufacturing"),
    "D": ("35", "35", "Electricity, gas & steam"),
    "E": ("36", "39", "Water supply & waste mgmt."),
    "F": ("41", "43", "Construction"),
    "G": ("45", "47", "Wholesale & retail trade"),
    "H": ("49", "53", "Transport & storage"),
    "I": ("55", "56", "Accommodation & food"),
    "J": ("58", "63", "Information & communication"),
    "K": ("64", "66", "Financial & insurance"),
    "L": ("68", "68", "Real estate"),
    "M": ("69", "75", "Professional & scientific"),
    "N": ("77", "82", "Administrative & support"),
    "O": ("84", "84", "Public administration"),
    "P": ("85", "85", "Education"),
    "Q": ("86", "88", "Health & social work"),
    "R": ("90", "93", "Arts, entertainment & recreation"),
    "S": ("94", "96", "Other service activities"),
    "T": ("97", "98", "Households as employers"),
    "U": ("99", "99", "Extraterritorial bodies"),
}

COUNTRY_FIXES = {
    "91": "India",                 # +91 calling code recorded instead of name
    "usa": "United States",
    "united states of america": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
    "uae": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "republic of korea": "Korea, South",
    "south korea": "Korea, South",
    "holland": "Netherlands",
    "hong kong, china": "Hong Kong",
}


def profile_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing-value table (counts and percentages)."""
    miss = df.isna().sum().to_frame("n_missing")
    miss["pct_missing"] = (miss["n_missing"] / len(df) * 100).round(3)
    return miss.sort_values("n_missing", ascending=False)


def detect_block_missing(df: pd.DataFrame,
                         cols: list[str] | None = None) -> pd.Series:
    """Boolean mask: rows where *all* core optional fields are missing.

    In this dataset ~23% of rows lack every company-detail field at once,
    i.e. missingness occurs in blocks rather than at random per column.
    """
    cols = BLOCK_MISSING_COLS if cols is None else [c for c in cols if c in df.columns]
    return df[cols].isna().all(axis=1)


def find_duplicate_rows(df: pd.DataFrame, subset: str | list | None = None) -> pd.DataFrame:
    """Return duplicated rows (kept for inspection, not removed here)."""
    return df[df.duplicated(subset=subset, keep=False)]


def find_invalid_dates(df: pd.DataFrame,
                       date_col: str = "CompanyRegistrationdate_date") -> pd.DataFrame:
    """Rows whose registration date is a placeholder (e.g. 1111-01-01)."""
    raw = df[date_col].astype(str)
    year = raw.str.slice(0, 4)
    invalid_mask = year.isin([str(y) for y in SENTINEL_YEARS])
    # also unparsable strings
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    invalid_mask = invalid_mask | parsed.isna()
    return df.loc[invalid_mask, [date_col]].assign(issue="sentinel/unparsable date")


def find_capital_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where Paid-up capital exceeds Authorized capital (rule breach)."""
    if not {"AuthorizedCapital", "PaidupCapital"}.issubset(df.columns):
        return pd.DataFrame()
    mask = df["AuthorizedCapital"].notna() & df["PaidupCapital"].notna() & (
        df["PaidupCapital"] > df["AuthorizedCapital"]
    )
    return df.loc[mask, ["CIN", "CompanyName", "AuthorizedCapital", "PaidupCapital"]]


def standardize_country(df: pd.DataFrame,
                        col: str = "CompanyIndian/Foreign Company") -> pd.Series:
    """Normalise country names; '91' is treated as India (calling code)."""
    s = df[col].astype("string").str.strip()
    lowered = s.str.lower()
    fixed = lowered.map(COUNTRY_FIXES)
    out = s.where(fixed.isna(), fixed.str.title().replace("Korea, South", "Korea, South"))
    # Title-casing mangles names like 'USA' handled above; keep clean labels
    out = out.replace({"Usa": "United States", "Uk": "United Kingdom",
                       "Uae": "United Arab Emirates", "U.S.A.": "United States"})
    return out


def nic_to_section(code) -> str | float:
    """Map a 5-digit NIC 2008 code to its one-letter section (A-U)."""
    if pd.isna(code):
        return np.nan
    try:
        s = str(int(code)).zfill(5)
    except (ValueError, TypeError):
        return np.nan
    two = s[:2]
    for sec, (lo, hi, _label) in NIC_SECTIONS.items():
        if lo <= two <= hi:
            return sec
    return np.nan


def parse_registration_date(df: pd.DataFrame,
                            date_col: str = "CompanyRegistrationdate_date") -> pd.Series:
    """Parse registration dates; sentinel 1111-* become NaT (invalid)."""
    dates = pd.to_datetime(df[date_col], errors="coerce")
    sentinel = dates.dt.year.isin(SENTINEL_YEARS)
    return dates.mask(sentinel)


def classify_status(status: pd.Series) -> pd.Series:
    """Group 18 granular MCA statuses into 4 analysis-friendly classes."""
    s = status.astype("string").str.strip()
    groups = pd.Series("Other / NA", index=s.index, dtype="object")
    groups[s.eq("Active")] = "Active"
    groups[s.str.contains("striking|Strike Off", case=False, na=False)] = "Strike Off / in process"
    groups[s.str.contains("Dissolv|Amalgamat|Converted|Liquidation|CIRP",
                          case=False, na=False)] = "Converted / dissolved / liquidation"
    groups[s.str.contains("Inactive|Dormant", case=False, na=False)] = "Inactive / dormant"
    return groups


def capital_bucket(capital: pd.Series) -> pd.Series:
    """Log-scale capital buckets (INR) for readable grouped analysis."""
    bins = [0, 1e5, 1e6, 1e7, 1e8, 1e9, np.inf]
    labels = ["<= 1 L", "1-10 L", "10 L-1 Cr", "1-10 Cr", "10-100 Cr", "> 100 Cr"]
    return pd.cut(capital, bins=bins, labels=labels, include_lowest=True)


def clean_pipeline(df: pd.DataFrame,
                   drop_block_missing: bool = False,
                   fix_country: bool = True,
                   parse_dates: bool = True,
                   drop_duplicates: bool = True,
                   add_features: bool = True) -> tuple[pd.DataFrame, dict]:
    """Reproducible cleaning pipeline. Returns (clean_df, log).

    The input dataframe is NEVER mutated - operations run on a copy.
    Each step is logged with before/after evidence for transparency.
    """
    log: dict = {"steps": [], "rows_in": int(len(df))}
    c = df.copy()

    # 1. duplicates -------------------------------------------------------
    if drop_duplicates:
        n0 = int(c.duplicated(subset=["CIN"]).sum())
        exact0 = int(c.duplicated().sum())
        c = c.drop_duplicates().drop_duplicates(subset=["CIN"], keep="first")
        log["steps"].append({"step": "deduplicate", "exact_duplicates_removed": exact0,
                             "duplicate_cin_removed": n0, "rows_after": int(len(c))})

    # 2. dates ------------------------------------------------------------
    if parse_dates:
        parsed = pd.to_datetime(c["CompanyRegistrationdate_date"], errors="coerce")
        n_invalid = int((parsed.isna() | parsed.dt.year.isin(SENTINEL_YEARS)).sum())
        c["CompanyRegistrationdate_date"] = parsed.mask(parsed.dt.year.isin(SENTINEL_YEARS))
        log["steps"].append({"step": "parse_dates",
                             "sentinel_or_invalid_set_to_nat": n_invalid})
        log["n_invalid_dates"] = n_invalid

    # 3. country standardisation -----------------------------------------
    if fix_country and "CompanyIndian/Foreign Company" in c.columns:
        n_code91 = int(c["CompanyIndian/Foreign Company"].astype(str)
                       .str.strip().eq("91").sum())
        c["CompanyIndian/Foreign Company"] = standardize_country(c)
        log["steps"].append({"step": "standardize_country",
                             "code_91_mapped_to_india": n_code91})
        log["n_country_code91"] = n_code91

    # 4. numeric coercion --------------------------------------------------
    for col in ["AuthorizedCapital", "PaidupCapital"]:
        if col in c.columns:
            c[col] = pd.to_numeric(c[col], errors="coerce")

    # 5. block-missing flag / optional removal -----------------------------
    mask = detect_block_missing(c)
    c["block_missing"] = mask
    log["n_block_missing"] = int(mask.sum())
    log["pct_block_missing"] = round(float(mask.mean() * 100), 2)
    if drop_block_missing:
        c = c.loc[~mask].copy()
        log["steps"].append({"step": "drop_block_missing",
                             "rows_removed": log["n_block_missing"],
                             "rows_after": int(len(c))})
    else:
        log["steps"].append({"step": "flag_block_missing",
                             "rows_flagged": log["n_block_missing"]})

    # 6. derived features ---------------------------------------------------
    if add_features:
        c["CompanyStatusGroup"] = classify_status(c["CompanyStatus"])
        c["RegistrationYear"] = c["CompanyRegistrationdate_date"].dt.year
        c["RegistrationMonth"] = c["CompanyRegistrationdate_date"].dt.month
        c["RegistrationMonthName"] = c["CompanyRegistrationdate_date"].dt.strftime("%b")
        c["IsListed"] = c["Listingstatus"].eq("Listed")
        c["IsForeign"] = c["CompanyIndian/Foreign Company"].ne("India") & \
            c["CompanyIndian/Foreign Company"].notna()
        ratio = c["PaidupCapital"] / c["AuthorizedCapital"].replace(0, np.nan)
        c["CapitalRatio"] = ratio.clip(0, 10)  # guard against wild ratios
        c["CapitalViolation"] = c["PaidupCapital"] > c["AuthorizedCapital"]
        c["NICSection"] = c["nic_code"].map(nic_to_section)
        c["CapitalBucket"] = capital_bucket(c["AuthorizedCapital"])
        log["steps"].append({"step": "derive_features",
                             "added": ["CompanyStatusGroup", "RegistrationYear",
                                       "RegistrationMonth", "IsListed", "IsForeign",
                                       "CapitalRatio", "CapitalViolation",
                                       "NICSection", "CapitalBucket"]})

    log["rows_out"] = int(len(c))
    log["rows_dropped"] = log["rows_in"] - log["rows_out"]
    return c, log


def data_quality_register(df: pd.DataFrame) -> dict:
    """Consolidated quality register: every issue found, with counts."""
    inv = df.isna()
    return {
        "n_rows": int(len(df)),
        "n_exact_duplicates": int(df.duplicated().sum()),
        "n_duplicate_cin": int(df["CIN"].duplicated().sum()),
        "invalid_dates": int(len(find_invalid_dates(df))),
        "capital_violations": int(len(find_capital_violations(df))),
        "n_zero_paidup": int((df["PaidupCapital"] == 0).sum()),
        "n_zero_authorized": int((df["AuthorizedCapital"] == 0).sum()),
        "n_country_code_91": int(df["CompanyIndian/Foreign Company"]
                                 .astype(str).str.strip().eq("91").sum()),
        "missing_table": profile_missingness(df).to_dict(orient="index"),
        "n_block_missing": int(detect_block_missing(df).sum()),
        "pct_block_missing": round(float(detect_block_missing(df).mean() * 100), 2),
        "n_any_missing_cells": int(inv.sum().sum()),
    }
