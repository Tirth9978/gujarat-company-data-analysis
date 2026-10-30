"""run_full_analysis.py - Execute the complete EDA pipeline ONCE.

Produces (single source of truth for notebook / website / report / PPT):
  data/processed/company_master_cleaned.csv   - cleaned + feature-enriched data
  data/processed/analysis_results.json        - every headline number
  reports/figures/*.png                       - all static figures
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1] / "download" / "EDA_Project"
sys.path.insert(0, str(PROJECT))

from src import data_loader as dl
from src import data_cleaning as dc
from src import statistical_analysis as sa
from src import visualization as vz
from src import insights as ins

pd.set_option("display.width", 200)

OUT_JSON = dl.PROCESSED_DIR / "analysis_results.json"
OUT_CSV = dl.PROCESSED_DIR / "company_master_cleaned.csv"


def convert(o):
    """JSON-safe conversion for numpy/pandas types."""
    if isinstance(o, dict):
        return {str(k): convert(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [convert(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if np.isnan(f) else f
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def downsample_lorenz(c: dict, n: int = 250) -> dict:
    """Keep the Lorenz curve readable in JSON without 160k points."""
    c = dict(c)
    x, y = np.array(c["lorenz_x"]), np.array(c["lorenz_y"])
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    c["lorenz_x"], c["lorenz_y"] = x[idx].tolist(), y[idx].tolist()
    return c


def main() -> dict:
    print("[1/8] Loading raw data (unchanged) ...")
    raw = dl.load_raw()
    overview = dl.basic_overview(raw)
    inventory = dl.column_inventory(raw)
    print("   ", overview)

    print("[2/8] Data-quality register ...")
    quality = dc.data_quality_register(raw)

    print("[3/8] Cleaning pipeline (keep & flag) ...")
    clean, clog = dc.clean_pipeline(raw)
    print("    cleaning log:", json.dumps(convert(clog), indent=2)[:600])

    # analysis-ready subset (details known) used for capital statistics
    known = clean.loc[~clean["block_missing"]].copy()
    print(f"    analysis-ready rows with full details: {len(known):,}")

    print("[4/8] Descriptive statistics ...")
    descriptives = {
        col: sa.descriptive_stats(known[col]) for col in ["AuthorizedCapital", "PaidupCapital"]
    }
    five_auth = sa.five_number_summary(known["AuthorizedCapital"])
    log_stats = {
        "log10_AuthorizedCapital": sa.descriptive_stats(np.log10(known.loc[known["AuthorizedCapital"] > 0, "AuthorizedCapital"])),
        "log10_PaidupCapital": sa.descriptive_stats(np.log10(known.loc[known["PaidupCapital"] > 0, "PaidupCapital"])),
    }

    print("[5/8] Distributions, crosstabs, correlation, fit ...")
    class_dist = sa.categorical_distribution(known["CompanyClass"])
    status_dist = sa.categorical_distribution(clean["CompanyStatusGroup"])
    status_raw_dist = sa.categorical_distribution(clean["CompanyStatus"])
    listing_dist = sa.categorical_distribution(clean["Listingstatus"])
    origin_dist = sa.categorical_distribution(clean["CompanyIndian/Foreign Company"], top=12)

    ct = sa.crosstab_analysis(known, "CompanyClass", "CompanyStatusGroup")
    ct_pretty = {str(k): {str(c): int(v) for c, v in row.items()} for k, row in ct["table"].to_dict(orient="index").items()}

    corr = sa.correlation_analysis(known, ["AuthorizedCapital", "PaidupCapital"], use_log=True)
    fit = sa.linear_fit(known["AuthorizedCapital"], known["PaidupCapital"], log10=True)

    print("[6/8] Outliers, concentration, time series ...")
    out_iqr = sa.iqr_outliers(known["PaidupCapital"])
    out_z = sa.zscore_outliers(known["PaidupCapital"])
    conc = sa.concentration_profile(known["PaidupCapital"], top_pct=1.0)
    conc5 = sa.concentration_profile(known["PaidupCapital"], top_pct=5.0)

    years = sa.yearly_registrations(clean)
    months = sa.monthly_registrations(clean)
    trend = sa.trend_summary(years, recent_from=2015, end_year=2025)  # 2026 is partial (snapshot in June)
    season_months = {int(k): int(v) for k, v in months.items()}

    # NIC sections
    known_nic = clean.loc[clean["NICSection"].notna()]
    sec_counts = known_nic["NICSection"].value_counts()
    sec_labels = {k: dc.NIC_SECTIONS.get(k, (None, None, k))[2] for k in sec_counts.index}
    sections_top = {k: {"count": int(v), "label": sec_labels.get(k, k)}
                    for k, v in sec_counts.head(10).items()}
    n_nic_classified = int(clean["NICSection"].notna().sum())

    print("[7/8] Missingness mechanism evidence ...")
    mech = sa.missingness_mechanism_report(clean)
    status_missing_tbl = sa.missingness_by_group(clean, "CompanyStatusGroup")

    print("[8/8] Building results + figures ...")
    R = {
        "overview": overview,
        "inventory": inventory.to_dict(orient="records"),
        "quality": quality,
        "cleaning_log": convert(clog),
        "analysis_ready_rows": int(len(known)),
        "descriptives": descriptives,
        "five_number_authorized": five_auth,
        "log_descriptives": log_stats,
        "class_dist": class_dist,
        "status_dist": status_dist,
        "status_raw_dist": status_raw_dist,
        "listing_dist": listing_dist,
        "origin_dist": origin_dist,
        "crosstab_class_status": {
            "chi2": ct["chi2"]["chi2"], "p_value": ct["chi2"]["p_value"],
            "dof": ct["chi2"]["dof"], "cramers_v": ct["chi2"]["cramers_v"],
            "table": ct_pretty},
        "correlation": {"pearson_r": float(corr["pearson"].iloc[0, 1]),
                        "spearman_r": float(corr["spearman"].iloc[0, 1]),
                        "n_used": corr["n_used"],
                        "pearson_matrix": corr["pearson"].to_dict(),
                        "spearman_matrix": corr["spearman"].to_dict()},
        "fit": fit,
        "outliers_iqr": out_iqr,
        "outliers_zscore": out_z,
        "concentration": downsample_lorenz(conc),
        "concentration_5pct": downsample_lorenz(conc5),
        "trend": trend,
        "trend_recent_base": int(years.get(2015, 0)),
        "yearly_counts": {int(k): int(v) for k, v in years.items()},
        "monthly_counts": season_months,
        "sections_top": sections_top,
        "n_nic_classified": n_nic_classified,
        "missingness_mechanism": {
            "chi2_era": convert(mech["chi2_era"]),
            "chi2_status": convert(mech["chi2_status"]),
            "by_era": convert(mech["by_era"]),
            "by_status": convert(mech["by_status"])},
    }

    # ---------------- figures ----------------
    figs = {}
    miss_tbl = dc.profile_missingness(raw)
    figs["missing_values"] = vz.plot_missing_values(miss_tbl)
    figs["missing_by_status"] = vz.plot_missingness_by_group(status_missing_tbl)
    figs["class_bar"] = vz.plot_category_bar(class_dist, "fig_class_distribution.png",
                                             "Company class mix (companies with full details)")
    figs["status_bar"] = vz.plot_category_bar(status_dist, "fig_status_distribution.png",
                                              "Status groups (all 222,915 companies)", color=vz.PALETTE["teal"])
    figs["cap_hist"] = vz.plot_capital_histograms(known)
    figs["box_class"] = vz.plot_box_by_group(known, "AuthorizedCapital", "CompanyClass")
    figs["scatter_fit"] = vz.plot_scatter_fit(known["AuthorizedCapital"], known["PaidupCapital"], fit)
    figs["corr_heatmap"] = vz.plot_corr_heatmap(corr["pearson"])
    figs["year_trend"] = vz.plot_yearly_trend(years)
    figs["season"] = vz.plot_month_seasonality(months)
    figs["nic_sections"] = vz.plot_nic_sections(sections_top)
    figs["lorenz"] = vz.plot_lorenz(conc)
    top15 = known.nlargest(15, "PaidupCapital")[["CompanyName", "PaidupCapital"]]
    figs["top_paid"] = vz.plot_top_companies(top15)
    figs["status_era"] = vz.plot_age_vs_survival(clean)
    R["figures"] = {k: Path(v).name for k, v in figs.items()}

    # ---------------- insights ----------------
    I = ins.build_insights(R)
    R["insights"] = I

    # ---------------- persist ----------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    keep = [c for c in clean.columns]
    clean[keep].to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(convert(R), indent=2))
    print(f"saved: {OUT_CSV.name} ({OUT_CSV.stat().st_size/1e6:.1f} MB), {OUT_JSON.name}")
    for k, v in figs.items():
        print(f"  figure {k}: {v}")
    return R


if __name__ == "__main__":
    main()
