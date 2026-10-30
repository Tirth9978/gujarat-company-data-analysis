"""Pytest suite for the EDA project src package.

Tests use small synthetic frames plus the real cleaned dataset when present,
so the suite is fast and does not depend on network or the 63 MB raw file.
Run:  pytest tests/ -q   (from the project root)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data_cleaning as dc  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import insights as ins  # noqa: E402
from src import statistical_analysis as sa  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic():
    """A tiny, hand-checkable company registry."""
    return pd.DataFrame(
        {
            "CIN": [f"U0000GJ200{i}PLC00000{i}" for i in range(6)],
            "CompanyName": [f"Company {i}" for i in range(6)],
            "CompanyROCcode": ["ROC Ahmedabad"] * 6,
            "CompanyCategory": ["Company limited by shares"] * 6,
            "CompanySubCategory": ["Non-government company"] * 6,
            "CompanyClass": ["Private", "Public", "Private", "Private", "Public", "Private"],
            "AuthorizedCapital": [1e5, 1e7, 0.0, 5e5, 2e9, 1e6],
            "PaidupCapital": [1e5, 5e6, 0.0, 7e5, 1e9, 0.0],  # row 3: paid>auth
            "CompanyRegistrationdate_date": [
                "2020-03-15", "1995-06-01", "1111-01-01", "2010-12-31",
                "2005-04-02", "not-a-date",
            ],
            "Listingstatus": ["Unlisted"] * 6,
            "CompanyStatus": ["Active", "Strike Off", "Active", "Active",
                              "Under Liquidation", "Dormant under section 455"],
            "CompanyStateCode": ["gujarat"] * 6,
            "CompanyIndian/Foreign Company": ["India", "91", "USA", "India", "India", "India"],
            "nic_code": [74999.0, 10100.0, 62010.0, 45200.0, 35101.0, np.nan],
        }
    )


# ---------------------------------------------------------------------------
# data_loader
# ---------------------------------------------------------------------------
def test_raw_file_exists():
    assert dl.RAW_DATA.exists(), "raw dataset must ship with the project"


def test_overview_shapes(synthetic):
    ov = dl.basic_overview(synthetic)
    assert ov["n_rows"] == 6 and ov["n_cols"] == len(synthetic.columns)
    assert ov["n_numeric"] == 3  # two capitals + nic_code
    assert ov["n_duplicate_rows"] == 0


def test_column_inventory_scales(synthetic):
    inv = dl.column_inventory(synthetic).set_index("column")
    assert inv.loc["AuthorizedCapital", "measurement_scale"].startswith("Ratio")
    assert inv.loc["CIN", "measurement_scale"].startswith("Nominal")
    assert inv.loc["nic_code", "role"] == "Coded category"


# ---------------------------------------------------------------------------
# data_cleaning
# ---------------------------------------------------------------------------
def test_block_missing_detection(synthetic):
    df = synthetic.copy()
    df.loc[2, ["CompanyCategory", "CompanySubCategory", "CompanyClass",
               "AuthorizedCapital", "PaidupCapital", "Listingstatus", "nic_code"]] = np.nan
    mask = dc.detect_block_missing(df)
    assert mask.sum() == 1 and not mask.iloc[0]


def test_sentinel_dates_flagged(synthetic):
    clean, log = dc.clean_pipeline(synthetic, add_features=False, fix_country=False)
    assert log["n_invalid_dates"] == 2  # 1111-01-01 + 'not-a-date'
    parsed = clean["CompanyRegistrationdate_date"]
    assert parsed.isna().sum() == 2


def test_country_normalisation(synthetic):
    out = dc.standardize_country(synthetic)
    vals = out.tolist()
    assert vals[1] == "India"          # '91' -> India (calling code)
    assert vals[2] == "United States"  # 'USA' -> full name
    assert vals[0] == "India"


def test_capital_violation_flag(synthetic):
    clean, _ = dc.clean_pipeline(synthetic, add_features=True, fix_country=False)
    assert int(clean["CapitalViolation"].sum()) == 1
    assert clean.loc[2, "PaidupCapital"] == 0  # zero capital preserved, not dropped


def test_status_grouping(synthetic):
    clean, _ = dc.clean_pipeline(synthetic, add_features=True, fix_country=False)
    g = clean["CompanyStatusGroup"].tolist()
    assert g[0] == "Active"
    assert g[1] == "Strike Off / in process"
    assert g[4] == "Converted / dissolved / liquidation"
    assert g[5] == "Inactive / dormant"


def test_nic_section_mapping():
    assert dc.nic_to_section(74999.0) == "M"   # 74999 -> professional activities
    assert dc.nic_to_section(10100.0) == "C"   # 10xxx -> manufacturing food
    assert dc.nic_to_section(35101.0) == "D"   # 35xxx -> electricity
    assert dc.nic_to_section(np.nan) is np.nan or pd.isna(dc.nic_to_section(np.nan))


def test_pipeline_never_mutates_input(synthetic):
    before = synthetic.copy(deep=True)
    dc.clean_pipeline(synthetic, add_features=True)
    pd.testing.assert_frame_equal(synthetic, before)


def test_cleaned_dataset_matches_expected_shape():
    """Integration check against the shipped processed data (if present)."""
    f = dl.PROCESSED_DIR / "company_master_cleaned.csv"
    if not f.exists():
        pytest.skip("processed dataset not built yet")
    df = pd.read_csv(f, usecols=["CIN", "block_missing", "RegistrationYear"])
    assert len(df) == 222_915
    assert df["CIN"].is_unique
    assert df["block_missing"].sum() == 51_773


# ---------------------------------------------------------------------------
# statistical_analysis
# ---------------------------------------------------------------------------
def test_descriptive_stats_known_values():
    s = pd.Series([1, 2, 3, 4, 100])
    d = sa.descriptive_stats(s)
    assert d["median"] == 3 and d["mean"] == 22
    assert d["iqr"] == 2.0 and d["skewness"] > 1.5  # heavy right skew detected
    five = sa.five_number_summary(s)
    assert five["min"] == 1 and five["max"] == 100 and five["median"] == 3


def test_correlation_perfect_and_sign():
    x = pd.Series([1, 10, 100, 1000])
    y = pd.Series([2, 20, 200, 2000])
    res = sa.correlation_analysis(pd.DataFrame({"x": x, "y": y}),
                                  ["x", "y"], use_log=True)
    assert res["pearson"].iloc[0, 1] == pytest.approx(1.0, abs=1e-6)


def test_linear_fit_recovers_slope():
    rng = np.random.default_rng(42)
    x = rng.uniform(1, 1000, 500)
    y = 2 * x + rng.normal(0, 0.01, 500)  # log10 slope -> 1 on log-log for y=k*x
    fit = sa.linear_fit(pd.Series(x), pd.Series(y), log10=True)
    assert fit["slope"] == pytest.approx(1.0, abs=1e-3)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-3)


def test_iqr_outliers_flags_upper_tail():
    s = pd.Series([10.0] * 50 + [1e9])
    res = sa.iqr_outliers(s)
    assert res["n_outliers"] >= 1 and res["pct_outliers"] > 0


def test_chi2_and_cramers_v():
    ct = pd.DataFrame([[80, 20], [40, 60]])
    res = sa.chi2_test(ct)
    assert res["significant_5pct"] and 0 < res["cramers_v"] <= 1


def test_concentration_profile():
    s = pd.Series([1.0] * 99 + [10_000.0])  # one whale
    c = sa.concentration_profile(s, top_pct=1.0)
    assert c["top_share_pct"] == pytest.approx(10_000 / 10_099 * 100, abs=0.5)
    assert 0 <= c["gini"] <= 1 and c["gini"] > 0.5


def test_trend_summary_excludes_partial_year():
    yrs = pd.Series({2015: 100, 2020: 400, 2025: 1600, 2026: 50})
    t = sa.trend_summary(yrs, recent_from=2015, end_year=2025)
    assert t["peak_year"] == 2025 and t["cagr_2015_to_2025_pct"] is not None
    assert t["cagr_end_year"] == 2025


# ---------------------------------------------------------------------------
# insights
# ---------------------------------------------------------------------------
def test_inr_formatting():
    assert ins.inr(1e5) == "₹1.00 L"
    assert ins.inr(2.5e7) == "₹2.50 Cr"
    assert ins.inr(999) == "₹999"
    assert ins.inr(None) == "NA"


def test_build_insights_returns_evidence():
    R = {
        "overview": {"n_rows": 100, "n_cols": 5},
        "quality": {"n_exact_duplicates": 0, "n_duplicate_cin": 0,
                    "n_block_missing": 23, "pct_block_missing": 23.0,
                    "n_zero_paidup": 5},
        "descriptives": {"AuthorizedCapital": {"median": 1e5, "mean": 1e7,
                                               "skewness": 50, "cv_pct": 900,
                                               "max": 1e12},
                         "PaidupCapital": {"median": 1e5, "max": 1e11}},
        "concentration": {"top_pct": 1.0, "top_share_pct": 90.0, "gini": 0.95, "n": 100},
        "correlation": {"pearson_r": 0.7, "spearman_r": 0.7},
        "fit": {"slope": 0.87, "r2": 0.6, "n": 100},
        "crosstab_class_status": {"chi2": 100.0, "p_value": 1e-9, "cramers_v": 0.1},
        "trend": {"peak_year": 2025, "peak_n": 22407},
        "trend_recent_base": 5000,
        "yearly_counts": {"2025": 22407, "2026": 12261},
        "monthly_counts": {"3": 21134, "11": 14972},
        "outliers_iqr": {"n_outliers": 10, "pct_outliers": 10.0},
    }
    insights = ins.build_insights(R)
    assert len(insights) >= 8
    assert all(i["evidence"] for i in insights)
    assert all(i["category"] and i["title"] and i["detail"] for i in insights)
