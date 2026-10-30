"""Interactive EDA Dashboard - Gujarat Company Master Data (IT462 project).

Streamlit + Plotly dashboard for live demonstration. The ORIGINAL dataset is
loaded read-only and never modified: every cleaning option is applied to a
working copy, and downloads always generate fresh files from the current view.

Run from the project root:   streamlit run website/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- project imports (app lives in website/) --------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data_cleaning as dc  # noqa: E402
from src import data_loader as dl  # noqa: E402
from src import insights as ins  # noqa: E402
from src import statistical_analysis as sa  # noqa: E402

st.set_page_config(page_title="EDA · Gujarat Company Master", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

NAVY, TEAL, AMBER, RED = "#0F2A43", "#2A9D8F", "#E8A33D", "#C4453C"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# =============================================================================
# Cached data layer (original file is NEVER written to)
# =============================================================================
@st.cache_resource(show_spinner="Loading raw dataset (63 MB, one-time)...")
def get_raw() -> pd.DataFrame:
    return dl.load_raw()


@st.cache_resource(show_spinner="Loading pipeline results...")
def get_results() -> dict:
    return json.loads(dl.RESULTS_JSON.read_text())


@st.cache_data(show_spinner="Applying cleaning options...")
def get_clean(_raw: pd.DataFrame, drop_block: bool, fix_country: bool,
              drop_dups: bool) -> tuple[pd.DataFrame, dict]:
    clean, log = dc.clean_pipeline(_raw, drop_block_missing=drop_block,
                                   fix_country=fix_country,
                                   drop_duplicates=drop_dups)
    return clean, log


R = get_results()
raw = get_raw()

# =============================================================================
# Sidebar - global filters + cleaning options
# =============================================================================
with st.sidebar:
    st.markdown(f"# 📊 EDA Dashboard")
    st.caption("IT462 · Exploratory Data Analysis · Gujarat company master, 1880–2026")
    page = st.radio("Section", ["Overview", "Data Quality & Cleaning", "Univariate",
                                "Bivariate", "Correlation & Fit", "Time Trends",
                                "Outliers & Concentration", "Insights & Conclusions",
                                "Downloads"], label_visibility="collapsed")

    st.divider()
    st.markdown("### 🧹 Interactive cleaning")
    opt_block = st.checkbox("Drop rows with ALL detail fields missing", value=False,
                            help="Default: keep & flag. 51,773 rows (23.2%) affected.")
    opt_country = st.checkbox("Standardise country names (91 → India)", value=True)
    opt_dups = st.checkbox("Remove duplicate CINs", value=True)

    clean, clog = get_clean(raw, opt_block, opt_country, opt_dups)

    st.divider()
    st.markdown("### 🔎 Global filters")
    yrs = clean["RegistrationYear"].dropna().astype(int)
    yr_range = st.slider("Registration year", int(yrs.min()), int(yrs.max()),
                         (int(yrs.min()), int(yrs.max())))
    class_opts = ["(any)"] + sorted(clean["CompanyClass"].dropna().unique())
    f_class = st.selectbox("Company class", class_opts)
    status_opts = ["(any)"] + sorted(clean["CompanyStatusGroup"].dropna().unique())
    f_status = st.selectbox("Status group", status_opts)
    f_listed = st.selectbox("Listing status", ["(any)", "Listed", "Unlisted"])
    f_foreign = st.selectbox("Origin", ["(any)", "Indian", "Foreign"])

    view = clean.copy()
    # rows with unparseable dates (NaT year) are always kept visible — they are
    # part of the data-quality story and must not vanish behind the year filter
    view = view[view["RegistrationYear"].between(yr_range[0], yr_range[1], inclusive="both")
                | view["RegistrationYear"].isna()]
    if f_class != "(any)":
        view = view[view["CompanyClass"] == f_class]
    if f_status != "(any)":
        view = view[view["CompanyStatusGroup"] == f_status]
    if f_listed != "(any)":
        view = view[view["Listingstatus"] == f_listed]
    if f_foreign == "Indian":
        view = view[view["CompanyIndian/Foreign Company"].eq("India")]
    elif f_foreign == "Foreign":
        view = view[view["IsForeign"].fillna(False)]

    st.divider()
    st.caption(f"Working set: **{len(view):,}** of {len(clean):,} cleaned rows "
               f"(raw file untouched: {len(raw):,})")

PAL_SEQ = [NAVY, TEAL, AMBER, RED, "#5B7DA3", "#8FD3C7", "#F4C67A", "#E2907E",
           "#94A3B8", "#3E5C76", "#B9C2CE", "#D98E5F", "#7B6D8D", "#4F8A8B"]


def style_fig(fig, title: str):
    fig.update_layout(title=dict(text=title, font=dict(size=18, color=NAVY)),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color="#22303F"), margin=dict(t=56, l=10, r=10, b=10),
                      colorway=PAL_SEQ)
    fig.update_xaxes(gridcolor="#E3E8EF", zerolinecolor="#E3E8EF")
    fig.update_yaxes(gridcolor="#E3E8EF", zerolinecolor="#E3E8EF")
    return fig


def kpi(label: str, value: str, help: str = ""):
    st.metric(label, value, help=help)


# =============================================================================
# Page 1 - Overview
# =============================================================================
if page == "Overview":
    st.title("Dataset overview")
    st.markdown("MCA company master extract — every company registered with **ROC Ahmedabad "
                "(state: Gujarat)** between **1880 and 2026**. Raw file loaded read-only.")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Rows", f"{R['overview']['n_rows']:,}")
    with c2:
        kpi("Columns", R["overview"]["n_cols"])
    with c3:
        kpi("Duplicate rows", "0", "identity keys are pristine")
    with c4:
        kpi("Missing cells", f"{R['overview']['pct_missing_cells']}%",
            "414,070 cells, concentrated in 8 detail fields")
    with c5:
        kpi("Working set", f"{len(view):,}", "after sidebar filters")

    st.subheader("Preview")
    st.dataframe(view.head(25), use_container_width=True, height=340)

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Column inventory")
        inv = pd.DataFrame(R["inventory"])[["column", "dtype", "measurement_scale",
                                            "n_unique", "pct_missing"]]
        st.dataframe(inv, use_container_width=True, height=520)
    with right:
        st.subheader("Summary statistics (numeric, filtered view)")
        num = view[["AuthorizedCapital", "PaidupCapital", "CapitalRatio"]].describe().T
        num["skew"] = view[["AuthorizedCapital", "PaidupCapital", "CapitalRatio"]].skew()
        st.dataframe(num, use_container_width=True)
        st.info("Capitals are extremely right-skewed (skewness ≈ 163–202). "
                "Interpret medians, not means — see Univariate section.", icon="💡")

# =============================================================================
# Page 2 - Data quality & cleaning
# =============================================================================
elif page == "Data Quality & Cleaning":
    st.title("Data quality & interactive cleaning")
    q = R["quality"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Duplicate rows / CINs", f"{q['n_exact_duplicates']} / {q['n_duplicate_cin']}")
    with c2:
        kpi("Sentinel dates (1111-*)", q["invalid_dates"], "set to NaT by pipeline")
    with c3:
        kpi("Paid-up > Authorised", q["capital_violations"], "impossible under Companies Act")
    with c4:
        kpi("Country coded '91'", f"{q['n_country_code_91']:,}", "calling code stored as country")

    st.subheader("Missing values — a *block* pattern")
    miss = dc.profile_missingness(raw)
    miss = miss[miss["n_missing"] > 0].reset_index().rename(columns={"index": "column"})
    fig = px.bar(miss, x="n_missing", y="column", orientation="h",
                 text=miss["pct_missing"].map(lambda v: f"{v:.2f}%"),
                 color_discrete_sequence=[AMBER])
    fig = style_fig(fig, "Missing values per column (raw data)")
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**{q['n_block_missing']:,} rows ({q['pct_block_missing']}%) miss every "
                "detail field at once** (category, class, subcategory, capitals, listing, "
                "NIC) while identifier fields are 100% populated → missingness happens in "
                "blocks, not per-cell at random.")

    st.subheader("Is it MCAR, MAR or MNAR?")
    mech = R["missingness_mechanism"]
    left, right = st.columns(2)
    with left:
        era_df = pd.DataFrame(mech["by_era"]).T.reset_index().rename(
            columns={"index": "era", "pct_block_missing": "% block-missing", "n": "n"})
        fig = px.bar(era_df, x="era", y="% block-missing", color="% block-missing",
                     color_continuous_scale=[TEAL, RED], text=era_df["% block-missing"].map(lambda v: f"{v:.1f}%"))
        fig = style_fig(fig, "Block-missingness by registration era (χ² V = 0.384)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        stat_df = pd.DataFrame(mech["by_status"]).T.reset_index().rename(
            columns={"index": "status", "pct_block_missing": "% block-missing", "n": "n"})
        fig = px.bar(stat_df, x="status", y="% block-missing",
                     color_discrete_sequence=[TEAL], text=stat_df["% block-missing"].map(lambda v: f"{v:.1f}%"))
        fig = style_fig(fig, "Block-missingness by status (χ² V = 0.219)")
        st.plotly_chart(fig, use_container_width=True)
    st.success("Not MCAR: rates range from ~0% (pre-2001) to 36.1% (2016–2026). Missingness "
               "depends on observed variables (MAR) with an administrative cause — recent "
               "registrations lack populated details in this extract. The pipeline keeps and "
               "flags these rows so filtering stays reproducible.", icon="✅")

    st.subheader("Cleaning log (current options)")
    st.json(clog["steps"])
    if opt_block:
        st.warning("Block-missing rows are DROPPED in the working set "
                   f"({clog.get('n_block_missing', 0):,} rows). Turn off to keep & flag.",
                   icon="⚠️")

# =============================================================================
# Page 3 - Univariate
# =============================================================================
elif page == "Univariate":
    st.title("Univariate exploration")
    tab_cat, tab_num = st.tabs(["Categorical", "Numeric"])

    with tab_cat:
        col = st.selectbox("Choose a categorical column",
                           ["CompanyClass", "CompanyStatusGroup", "Listingstatus",
                            "CompanyIndian/Foreign Company", "NICSection",
                            "CapitalBucket", "CompanySubCategory"])
        top_n = st.slider("Top N categories", 5, 25, 10)
        vc = view[col].value_counts(dropna=True).head(top_n)
        pct = (vc / vc.sum() * 100).round(2)
        t1, t2 = st.columns([3, 2])
        with t1:
            fig = px.bar(x=vc.values, y=vc.index.astype(str), orientation="h",
                         text=[f"{v:,} ({p}%)" for v, p in zip(vc.values, pct.values)],
                         color_discrete_sequence=[NAVY])
            fig = style_fig(fig, f"{col} — distribution (filtered view)")
            fig.update_yaxes(categoryorder="total ascending")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            fig = px.pie(names=vc.index.astype(str), values=vc.values, hole=0.45,
                         color_discrete_sequence=PAL_SEQ)
            fig = style_fig(fig, f"{col} — share")
            fig.update_traces(textinfo="percent", textposition="inside")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame({"count": vc, "pct": pct}))

    with tab_num:
        col = st.selectbox("Choose a numeric column",
                           ["AuthorizedCapital", "PaidupCapital", "CapitalRatio"])
        use_log = st.checkbox("log10 transform (recommended for capitals)", value=True)
        bins = st.slider("Bins", 20, 120, 60)
        d = view[[col]].dropna()
        if use_log and col != "CapitalRatio":
            d = d[d[col] > 0]
            plot_vals = np.log10(d[col])
            xlab = f"log10({col})"
        else:
            plot_vals = d[col]
            xlab = col
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(x=plot_vals, nbins=bins, color_discrete_sequence=[TEAL])
            fig = style_fig(fig, f"{col} — histogram{' (log10)' if use_log else ''}")
            fig.update_layout(showlegend=False, xaxis_title=xlab, yaxis_title="Companies")
            fig.add_vline(x=float(plot_vals.median()), line_color=AMBER, line_width=3,
                          annotation_text=f"median = {plot_vals.median():,.2f}",
                          annotation_position="top")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.box(x=plot_vals, color_discrete_sequence=[NAVY])
            fig = style_fig(fig, f"{col} — boxplot{' (log10)' if use_log else ''}")
            fig.update_layout(showlegend=False, xaxis_title=xlab)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.Series(sa.descriptive_stats(d[col])).to_frame("value")
                     .drop(index=["mode"]), use_container_width=True, height=560)

# =============================================================================
# Page 4 - Bivariate
# =============================================================================
elif page == "Bivariate":
    st.title("Bivariate exploration")
    mode = st.radio("Relationship type", ["Categorical × Categorical",
                                          "Numeric × Categorical",
                                          "Numeric × Numeric"], horizontal=True)

    if mode == "Categorical × Categorical":
        c1, c2 = st.columns(2)
        with c1:
            x = st.selectbox("Rows", ["CompanyClass", "CompanyStatusGroup", "Listingstatus",
                                      "NICSection", "CapitalBucket"], key="cx")
        with c2:
            y = st.selectbox("Columns", ["CompanyStatusGroup", "CompanyClass",
                                         "Listingstatus", "NICSection"], index=0, key="cy")
        sub = view[[x, y]].dropna()
        ct = pd.crosstab(sub[x], sub[y])
        rowpct = pd.crosstab(sub[x], sub[y], normalize="index").mul(100).round(1)
        t1, t2 = st.columns(2)
        with t1:
            fig = px.imshow(ct, text_auto=True, color_continuous_scale="Blues",
                            aspect="auto")
            fig = style_fig(fig, f"Crosstab counts: {x} × {y}")
            st.plotly_chart(fig, use_container_width=True)
        with t2:
            fig = px.imshow(rowpct, text_auto=".1f", color_continuous_scale="Teal",
                            aspect="auto")
            fig = style_fig(fig, f"Row %: {x} × {y}")
            st.plotly_chart(fig, use_container_width=True)
        res = sa.chi2_test(ct)
        st.markdown(f"**Chi-square test of independence:** χ² = {res['chi2']:,.1f}, "
                    f" dof = {res['dof']}, p = {res['p_value']:.2e}, "
                    f" Cramér's V = {res['cramers_v']:.3f} → "
                    f"{'significant association' if res['significant_5pct'] else 'no significant association'} at 5%.")

    elif mode == "Numeric × Categorical":
        c1, c2, c3 = st.columns(3)
        with c1:
            numc = st.selectbox("Numeric", ["AuthorizedCapital", "PaidupCapital"])
        with c2:
            catc = st.selectbox("Category", ["CompanyClass", "CompanyStatusGroup",
                                             "Listingstatus", "NICSection"])
        with c3:
            logv = st.checkbox("log10 numeric", value=True)
        sub = view[[numc, catc]].dropna()
        if logv:
            sub = sub[sub[numc] > 0]
            sub = sub.assign(plot=np.log10(sub[numc]))
            ylab = f"log10({numc})"
        else:
            sub = sub.assign(plot=sub[numc])
            ylab = numc
        fig = px.box(sub, x=catc, y="plot", color=catc,
                     color_discrete_sequence=PAL_SEQ, points="outliers")
        fig = style_fig(fig, f"{numc} by {catc}{' (log10)' if logv else ''}")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(sa.group_statistics(sub.assign(v=sub["plot"]), catc, "v"),
                     use_container_width=True)

    else:
        c1, c2 = st.columns(2)
        with c1:
            x = st.selectbox("X", ["AuthorizedCapital", "PaidupCapital"])
        with c2:
            y = st.selectbox("Y", ["PaidupCapital", "AuthorizedCapital"], index=0)
        logv = st.checkbox("log10 both axes (recommended)", value=True)
        sub = view[[x, y]].dropna()
        if logv:
            sub = sub[(sub[x] > 0) & (sub[y] > 0)]
            sub = sub.assign(xl=np.log10(sub[x]), yl=np.log10(sub[y]))
            cols = ["xl", "yl"]
        else:
            sub = sub.assign(xl=sub[x], yl=sub[y])
            cols = ["xl", "yl"]
        if len(sub) > 20000:
            sub_s = sub[cols].sample(20000, random_state=42)
        else:
            sub_s = sub[cols]
        fig = px.scatter(sub_s, x="xl", y="yl", opacity=0.35,
                         color_discrete_sequence=[NAVY])
        if len(sub_s) > 2:  # OLS overlay from the shared statistical module
            f = sa.linear_fit(sub[x], sub[y], log10=logv)
            xs = np.linspace(sub_s["xl"].min(), sub_s["xl"].max(), 100)
            fig.add_scatter(x=xs, y=f["intercept"] + f["slope"] * xs, mode="lines",
                            name=f"OLS: slope {f['slope']:.2f}, R² {f['r2']:.2f}",
                            line=dict(color=AMBER, width=4))
            fig.update_layout(legend=dict(orientation="h", y=1.06))
        fig = style_fig(fig, f"{x} vs {y}{' (log10)' if logv else ''} — {len(sub):,} rows")
        fig.update_xaxes(title="log10(X)" if logv else x)
        fig.update_yaxes(title="log10(Y)" if logv else y)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Random sample of ≤ 20,000 points is plotted for responsiveness; "
                   "the OLS line uses ALL available pairs.")

# =============================================================================
# Page 5 - Correlation & fit
# =============================================================================
elif page == "Correlation & Fit":
    st.title("Correlation analysis & data fitting")
    st.markdown("Course toolkit: **Pearson** (linear), **Spearman** (monotone, robust), "
                "correlation matrix and an OLS fit with goodness-of-fit statistics — "
                "computed on log10 capitals because raw values span seven orders of magnitude.")
    known = view.loc[~view["block_missing"] &
                     view["AuthorizedCapital"].notna() & view["PaidupCapital"].notna()]
    pos = known[(known["AuthorizedCapital"] > 0) & (known["PaidupCapital"] > 0)]
    log_pos = np.log10(pos[["AuthorizedCapital", "PaidupCapital"]])
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Pearson r (log10)", f"{log_pos['AuthorizedCapital'].corr(log_pos['PaidupCapital']):.4f}")
    with c2:
        kpi("Spearman ρ", f"{log_pos['AuthorizedCapital'].corr(log_pos['PaidupCapital'], method='spearman'):.4f}")
    with c3:
        kpi("n pairs", f"{len(log_pos):,}")

    corr_m = log_pos.corr(method="pearson").round(3)
    spear_m = log_pos.corr(method="spearman").round(3)
    t1, t2 = st.columns(2)
    with t1:
        fig = px.imshow(corr_m, text_auto=".3f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        fig = style_fig(fig, "Pearson correlation (log10 capitals)")
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        fig = px.imshow(spear_m, text_auto=".3f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        fig = style_fig(fig, "Spearman correlation (log10 capitals)")
        st.plotly_chart(fig, use_container_width=True)

    f = sa.linear_fit(known["AuthorizedCapital"], known["PaidupCapital"], log10=True)
    st.subheader("OLS fit: log10(Paid-up) ~ log10(Authorised)")
    st.dataframe(pd.Series(f).to_frame("statistic"), use_container_width=True)
    st.markdown(f"**Read:** slope {f['slope']:.3f} → a 10× increase in authorised capital is "
                f"associated with a ~{10 ** f['slope']:.1f}× increase in paid-up capital; "
                f"R² = {f['r2']:.3f} (RMSE {f['rmse_log10']:.2f} log-units). "
                "Authorisation is a commitment ceiling, not a capitalisation guarantee.")

# =============================================================================
# Page 6 - Time trends
# =============================================================================
elif page == "Time Trends":
    st.title("Time-series EDA — trend & seasonality")
    yr = view["RegistrationYear"].dropna().astype(int).value_counts().sort_index()
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Peak year", f"{yr.idxmax()}", f"{yr.max():,} registrations")
    with c2:
        kpi("2025 total", f"{R['trend']['peak_n']:,}", "last complete year")
    with c3:
        kpi("CAGR 2015→2025", f"{R['trend']['cagr_2015_to_2025_pct']}%", "full years only (2026 partial)")

    fig = px.area(x=yr.index, y=yr.values, color_discrete_sequence=[NAVY])
    fig = style_fig(fig, f"Registrations per year — filtered view ({len(view):,} rows)")
    fig.update_xaxes(title="Year of registration")
    fig.update_yaxes(title="New companies")
    st.plotly_chart(fig, use_container_width=True)

    mo = view["RegistrationMonth"].dropna().astype(int).value_counts().sort_index()
    mo = mo.reindex(range(1, 13), fill_value=0)
    fig = px.bar(x=MONTHS, y=mo.values, color_discrete_sequence=[TEAL])
    fig = style_fig(fig, "Seasonality — registrations by calendar month")
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="New companies")
    fig.add_hline(y=float(mo.mean()), line_color=RED, line_dash="dash",
                  annotation_text=f"mean {mo.mean():,.0f}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Status mix by era (survivorship warning)")
    era = pd.cut(view["RegistrationYear"], bins=[1850, 1990, 2000, 2010, 2027],
                 labels=["1850–1990", "1991–2000", "2001–2010", "2011–2026"])
    era_ct = pd.crosstab(era, view["CompanyStatusGroup"], normalize="index").mul(100).round(1)
    fig = px.bar(era_ct, barmode="stack", color_discrete_map={
        "Active": TEAL, "Strike Off / in process": RED,
        "Converted / dissolved / liquidation": "#8A94A6",
        "Inactive / dormant": AMBER, "Other / NA": "#B9C2CE"})
    fig = style_fig(fig, "Status-group composition per registration era (%)")
    fig.update_xaxes(title="Era")
    fig.update_yaxes(title="% of era")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Older cohorts carry far more strike-offs — current status mix is not "
               "comparable across eras without age adjustment (survivorship bias).")

# =============================================================================
# Page 7 - Outliers & concentration
# =============================================================================
elif page == "Outliers & Concentration":
    st.title("Outlier detection & capital concentration")
    col = st.selectbox("Capital column", ["PaidupCapital", "AuthorizedCapital"])
    sub = view.loc[~view["block_missing"], col].dropna()
    sub = sub[sub > 0]

    t1, t2 = st.columns(2)
    with t1:
        res_iqr = sa.iqr_outliers(sub)
        st.markdown(f"#### Tukey fences (1.5×IQR, log10 space)\n"
                    f"- lower fence: **₹{res_iqr['lower_fence']:,.0f}**\n"
                    f"- upper fence: **₹{res_iqr['upper_fence']:,.0f}**\n"
                    f"- outliers: **{res_iqr['n_outliers']:,}** ({res_iqr['pct_outliers']}%)")
    with t2:
        res_z = sa.zscore_outliers(sub)
        st.markdown(f"#### Z-score (|z| > 3, log10 space)\n"
                    f"- outliers: **{res_z['n_outliers']:,}** ({res_z['pct_outliers']}%)\n"
                    f"- consistent with the IQR view: log-space high tail is long but thin")

    topn = st.slider("Top N companies", 5, 30, 15)
    top = view.loc[~view["block_missing"]].nlargest(topn, col)[
        ["CompanyName", "CompanyClass", col]]
    top = top.assign(**{f"{col} (₹B)": (top[col] / 1e9).round(2)}).drop(columns=[col])
    st.dataframe(top, use_container_width=True)

    conc = sa.concentration_profile(sub, top_pct=1.0)
    c1, c2 = st.columns([2, 3])
    with c1:
        kpi("Top 1% capital share", f"{conc['top_share_pct']}%", f"Gini = {conc['gini']}")
        kpi("Top 5% capital share", f"{R['concentration_5pct']['top_share_pct']}%")
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=np.linspace(0, 1, 100), y=np.linspace(0, 1, 100),
                        mode="lines", name="perfect equality",
                        line=dict(color="#8A94A6", dash="dash"))
        fig.add_scatter(x=conc["lorenz_x"], y=conc["lorenz_y"], mode="lines",
                        name=f"Lorenz (Gini {conc['gini']:.3f})",
                        line=dict(color=NAVY, width=3), fill="tonexty",
                        fillcolor="rgba(15,42,67,0.08)")
        fig = style_fig(fig, "Lorenz curve — paid-up capital concentration")
        fig.update_xaxes(title="Cumulative share of companies")
        fig.update_yaxes(title="Cumulative share of capital", range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Outliers are genuine giants (Reliance Jio ₹1,531 B paid-up, Vodafone Idea, "
               "Sardar Sarovar) — kept, and handled with log transforms and robust stats. "
               "10,925 companies with ₹0 paid-up form the low-side special case (flagged).")

# =============================================================================
# Page 8 - Insights
# =============================================================================
elif page == "Insights & Conclusions":
    st.title("Key insights & conclusions")
    st.caption("Every insight is generated from the same pipeline results "
               "(data/processed/analysis_results.json) — identical numbers in the notebook, "
               "report and presentation.")
    for i, item in enumerate(R["insights"], 1):
        with st.container(border=True):
            st.markdown(f"#### {i}. {item['title']}")
            st.markdown(f"`{item['category']}` &nbsp; {item['detail']}")
            st.caption(f"Evidence: {item['evidence']}")

    st.subheader("Limitations")
    st.markdown(
        "1. **Registry, not economy** — no turnover/employment; capital ≠ activity.\n"
        "2. **23.2% block-missing details** — capital stats describe the 171,142 detail-known companies.\n"
        "3. **Snapshot censoring** — extract ends June 2026; 2026 excluded from CAGR.\n"
        "4. **Status ≠ survival time** — no closure dates available.\n"
        "5. **Zero-capital entries & 74 rule violations** stay flagged, never silently 'fixed'.")

# =============================================================================
# Page 9 - Downloads
# =============================================================================
else:
    st.title("Downloads")
    st.markdown("Files are generated **fresh from the current filters & cleaning options**. "
                "The raw dataset on disk is never modified.")
    dl1, dl2, dl3, dl4 = st.columns(4)

    with dl1:
        st.markdown("#### Cleaned data (CSV)")
        csv_bytes = view.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download working set", csv_bytes,
                           f"gujarat_companies_cleaned_{len(view)}.csv", "text/csv",
                           use_container_width=True)
        st.caption(f"{len(view):,} rows × {len(view.columns)} cols, current filters applied")

    with dl2:
        st.markdown("#### Filtered subset (Parquet)")
        try:
            pq = view.to_parquet(index=False)
            st.download_button("⬇️ Download Parquet", pq,
                               f"companies_filtered_{len(view)}.parquet",
                               "application/octet-stream", use_container_width=True)
            st.caption("compact binary, fast reload")
        except Exception as e:  # pragma: no cover
            st.caption(f"Parquet engine unavailable: {e}")

    with dl3:
        st.markdown("#### Pipeline results (JSON)")
        st.download_button("⬇️ Download results JSON", dl.RESULTS_JSON.read_bytes(),
                           "analysis_results.json", "application/json",
                           use_container_width=True)
        st.caption("all headline statistics (52 KB)")

    with dl4:
        st.markdown("#### Static figures (PNG)")
        figdir = ROOT / "reports" / "figures"
        figfiles = sorted(figdir.glob("*.png"))
        sel = st.selectbox("Figure", [f.name for f in figfiles])
        st.download_button("⬇️ Download figure", (figdir / sel).read_bytes(),
                           sel, "image/png", use_container_width=True)

    st.divider()
    st.markdown("#### Full cleaned pipeline output")
    proc = dl.PROCESSED_DIR / "company_master_cleaned.csv"
    st.caption(f"`{proc.relative_to(ROOT)}` — {proc.stat().st_size/1e6:.0f} MB, "
               "all 222,915 rows, cleaned + flagged + feature-enriched (generated by "
               "scripts/run_full_analysis.py). Also see `reports/analysis_report.md`.")

    st.markdown("#### Reproduce everything")
    st.code("streamlit run website/app.py          # this dashboard\n"
            "jupyter lab notebooks/EDA_Demo.ipynb   # live-demo notebook\n"
            "pytest tests/ -q                       # test suite\n"
            "python scripts/run_full_analysis.py    # rebuild results + figures",
            language="bash")

st.sidebar.divider()
st.sidebar.caption("Data: MCA company master extract (data.gov.in) · "
                   "Built with Streamlit + Plotly · IT462 EDA project")
