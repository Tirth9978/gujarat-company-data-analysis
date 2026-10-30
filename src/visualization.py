"""visualization.py - Static (matplotlib/seaborn) EDA figures.

Every figure function saves a PNG under reports/figures/ and returns the
matplotlib Axes so notebooks/website can reuse the same logic. The website
additionally builds interactive Plotly charts in app.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Font fallback chain (Latin-first; Noto Sans SC catches CJK if ever needed)
for _f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
           "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf"]:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# A consistent, professional palette (navy / teal / amber)
PALETTE = {"primary": "#0F2A43", "accent": "#E8A33D", "teal": "#2A9D8F",
           "red": "#C4453C", "grey": "#8A94A6", "light": "#E9EEF4"}
SEQ_CMAP = sns.light_palette(PALETTE["primary"], as_cmap=True)


def set_plot_style() -> None:
    """Project-wide matplotlib/seaborn style (consistent across deliverables)."""
    sns.set_theme(context="talk", style="whitegrid",
                  rc={"axes.edgecolor": "#42536B", "axes.labelcolor": "#22303F",
                      "text.color": "#22303F", "xtick.color": "#42536B",
                      "ytick.color": "#42536B", "grid.color": "#D7DEE8",
                      "figure.facecolor": "white"})
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["axes.titleweight"] = "bold"


def _save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, bbox_inches=None)  # constrained_layout manages margins
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Individual figures
# ---------------------------------------------------------------------------
def plot_missing_values(miss_table: pd.DataFrame, name: str = "fig_missing_values.png"):
    """Horizontal bar chart of missing values per column."""
    set_plot_style()
    t = miss_table[miss_table["n_missing"] > 0].sort_values("n_missing")
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ax.barh(t.index, t["pct_missing"], color=PALETTE["accent"], edgecolor=PALETTE["primary"])
    for i, (n, p) in enumerate(zip(t["n_missing"], t["pct_missing"])):
        ax.text(p + 0.3, i, f"{p:.1f}%  (n={n:,})", va="center", fontsize=10)
    ax.set_xlim(0, max(t["pct_missing"]) * 1.35)
    ax.set_xlabel("Missing values (% of rows)")
    ax.set_title("Missing values by column - block pattern in company-detail fields")
    return _save(fig, name)


def plot_missingness_by_group(group_tbl: pd.DataFrame,
                              name: str = "fig_missingness_by_status.png",
                              title: str = "Block-missingness rate by company status group"):
    """Where do the 23% detail-missing rows live? (MAR/MNAR evidence)."""
    set_plot_style()
    t = group_tbl.sort_values("pct_block_missing")
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.barh(t.index.astype(str), t["pct_block_missing"],
            color=[PALETTE["red"] if v > 50 else PALETTE["teal"] for v in t["pct_block_missing"]])
    for i, v in enumerate(t["pct_block_missing"]):
        ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("% of group with all detail fields missing")
    ax.set_title(title)
    return _save(fig, name)


def plot_category_bar(dist: dict, name: str, title: str,
                      xlabel: str = "Companies", color: str | None = None):
    """Bar chart from a categorical distribution dict."""
    set_plot_style()
    labels = list(dist.keys())
    counts = [v["count"] if isinstance(v, dict) else v for v in dist.values()]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bars = ax.barh(labels[::-1], counts[::-1],
                   color=color or PALETTE["primary"], edgecolor="white")
    total = sum(counts)
    for b, v in zip(bars, counts[::-1]):
        ax.text(b.get_width() + total * 0.008, b.get_y() + b.get_height() / 2,
                f"{v:,} ({v / total * 100:.1f}%)", va="center", fontsize=10)
    ax.set_xlim(0, max(counts) * 1.28)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return _save(fig, name)


def plot_capital_histograms(df: pd.DataFrame, name: str = "fig_capital_hist_log.png"):
    """Overlayed log10 histograms of authorised vs paid-up capital."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True, sharex=True)
    for ax, col, color, label in [
        (axes[0], "AuthorizedCapital", PALETTE["primary"], "Authorised capital"),
        (axes[1], "PaidupCapital", PALETTE["teal"], "Paid-up capital"),
    ]:
        vals = df[col].dropna()
        vals = vals[vals > 0]
        log_vals = np.log10(vals)
        ax.hist(log_vals, bins=60, color=color, edgecolor="white", linewidth=.3)
        ax.axvline(log_vals.median(), color=PALETTE["accent"], lw=2.2,
                   label=f"median = ₹{vals.median():,.0f}")
        ax.set_title(f"{label} (log$_{{10}}$ scale)")
        ax.set_xlabel("log$_{{10}}$(INR)")
        ax.set_ylabel("Companies")
        ax.legend(frameon=True)
    return _save(fig, name)


def plot_box_by_group(df: pd.DataFrame, value_col: str, group_col: str,
                      name: str = "fig_capital_box_by_class.png",
                      title: str = "Authorised capital by company class (log scale)"):
    """Boxplot of a log-capital by a categorical group, ordered by median."""
    set_plot_style()
    d = df[[value_col, group_col]].dropna()
    d = d[d[value_col] > 0]
    d = d.assign(logv=np.log10(d[value_col]))
    order = d.groupby(group_col)["logv"].median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    sns.boxplot(data=d, x=group_col, y="logv", order=order, ax=ax,
                hue=group_col, palette=[PALETTE["teal"], PALETTE["primary"],
                                        PALETTE["accent"]][:len(order)],
                width=.55, linewidth=1.4, showfliers=True,
                flierprops={"marker": ".", "markersize": 2, "alpha": .25})
    ax.set_xticklabels([f"{g}\n(n={int((d[group_col] == g).sum()):,})" for g in order])
    ax.set_xlabel("")
    ax.set_ylabel("log$_{10}$(Authorised capital, INR)")
    ax.set_title(title)
    return _save(fig, name)


def plot_scatter_fit(x, y, fit: dict, name: str = "fig_capital_scatter_fit.png",
                     xlabel: str = "log10 Authorised capital",
                     ylabel: str = "log10 Paid-up capital",
                     title: str = "Authorised vs paid-up capital with OLS fit"):
    """Log-log scatter with regression line and R-squared annotation."""
    set_plot_style()
    d = pd.concat([x, y], axis=1).dropna()
    d = d[(d.iloc[:, 0] > 0) & (d.iloc[:, 1] > 0)]
    lx, ly = np.log10(d.iloc[:, 0]), np.log10(d.iloc[:, 1])
    # hexbin keeps 170k points readable (log-binned colour for mid-density)
    fig, ax = plt.subplots(figsize=(8.6, 6), constrained_layout=True)
    cmap = sns.cubehelix_palette(start=.4, rot=-.35, gamma=.6, hue=.8, light=.95, dark=.25, as_cmap=True)
    hb = ax.hexbin(lx, ly, gridsize=60, cmap=cmap, mincnt=1, bins="log", linewidths=0.05)
    xs = np.linspace(lx.min(), lx.max(), 100)
    ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color=PALETTE["accent"],
            lw=2.6, label=f"OLS fit: slope={fit['slope']:.2f}, R$^2$={fit['r2']:.2f}")
    cb = fig.colorbar(hb, ax=ax, pad=.01)
    cb.set_label("Companies per hexbin (log scale)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=True, loc="upper left")
    return _save(fig, name)


def plot_corr_heatmap(corr: pd.DataFrame, name: str = "fig_corr_heatmap.png",
                      title: str = "Correlation matrix (log10 capitals)"):
    """Annotated Pearson/Spearman heatmap."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, linewidths=1, linecolor="white",
                cbar_kws={"shrink": .8}, ax=ax,
                annot_kws={"size": 13, "weight": "bold"})
    ax.set_title(title)
    return _save(fig, name)


def plot_yearly_trend(year_counts: pd.Series, name: str = "fig_yearly_trend.png",
                      title: str = "Company registrations in Gujarat per year, 1880-2026"):
    """Annual registration counts with decade shading (trend detection)."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.plot(year_counts.index, year_counts.values, color=PALETTE["primary"], lw=1.8)
    ax.fill_between(year_counts.index, year_counts.values,
                    color=PALETTE["primary"], alpha=.15)
    # annotate liberalisation & post-2020 surge
    ax.axvline(1991, color=PALETTE["grey"], ls="--", lw=1.2)
    ax.text(1992, ax.get_ylim()[1] * .92, "1991 liberalisation", fontsize=10, color=PALETTE["grey"])
    peak_year = int(year_counts.idxmax())
    ax.annotate(f"peak {peak_year}: {year_counts.max():,}",
                xy=(peak_year, year_counts.max()),
                xytext=(peak_year - 26, year_counts.max() * 0.9),
                arrowprops=dict(arrowstyle="->", color=PALETTE["red"]),
                color=PALETTE["red"], fontsize=11, weight="bold")
    ax.set_xlabel("Year of registration")
    ax.set_ylabel("New companies")
    ax.set_title(title)
    return _save(fig, name)


def plot_month_seasonality(month_counts: pd.Series,
                           name: str = "fig_month_seasonality.png",
                           title: str = "Registrations by calendar month (seasonality)"):
    """Average registrations per month - March year-end rush visible."""
    set_plot_style()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    vals = [month_counts.get(i, 0) for i in range(1, 13)]
    colors = [PALETTE["accent"] if v == max(vals) else PALETTE["primary"] for v in vals]
    fig, ax = plt.subplots(figsize=(9.5, 4.6), constrained_layout=True)
    ax.bar(months, vals, color=colors, edgecolor="white")
    mean_v = np.mean(vals)
    ax.axhline(mean_v, color=PALETTE["red"], ls="--", lw=1.4,
               label=f"monthly mean = {mean_v:,.0f}")
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * .015, f"{v / 1000:.1f}k", ha="center", fontsize=9)
    ax.set_ylabel("New companies")
    ax.set_title(title)
    ax.legend(frameon=True)
    return _save(fig, name)


def plot_nic_sections(section_counts: dict, name: str = "fig_nic_sections.png",
                      title: str = "Top NIC-2008 sections of Gujarat companies"):
    """Top economic sectors (NIC section) bar chart."""
    set_plot_style()
    labels = list(section_counts.keys())[:10][::-1]
    counts = [section_counts[k]["count"] for k in labels]
    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    bars = ax.barh(labels, counts, color=PALETTE["primary"], edgecolor="white")
    bars[-1].set_color(PALETTE["accent"])
    total = sum(section_counts[k]["count"] for k in section_counts)
    for b, v in zip(bars, counts):
        ax.text(b.get_width() + max(counts) * .01, b.get_y() + b.get_height() / 2,
                f"{v / total * 100:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Share of companies with known NIC (%)")
    ax.set_title(title)
    return _save(fig, name)


def plot_lorenz(conc: dict, name: str = "fig_lorenz_capital.png",
                title: str = "Lorenz curve - paid-up capital concentration"):
    """Lorenz curve with Gini annotation (inequality of capital)."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100),
            ls="--", color=PALETTE["grey"], lw=1.4, label="perfect equality")
    ax.plot(conc["lorenz_x"], conc["lorenz_y"], color=PALETTE["primary"], lw=2.4,
            label=f"paid-up capital (Gini = {conc['gini']:.3f})")
    ax.fill_between(conc["lorenz_x"], conc["lorenz_y"],
                    np.linspace(0, 1, len(conc["lorenz_y"])), alpha=.12,
                    color=PALETTE["primary"])
    ax.set_xlabel("Cumulative share of companies (poorest -> richest)")
    ax.set_ylabel("Cumulative share of paid-up capital")
    ax.set_title(title)
    ax.legend(frameon=True, loc="upper left")
    return _save(fig, name)


def plot_top_companies(top_df: pd.DataFrame, name: str = "fig_top_paidup.png",
                       title: str = "Top 15 companies by paid-up capital",
                       value_col: str = "PaidupCapital", label_col: str = "CompanyName"):
    """Horizontal bar chart of the largest companies (outlier inspection)."""
    set_plot_style()
    t = top_df.sort_values(value_col)
    labels = [s[:34] + ("…" if len(s) > 34 else "") for s in t[label_col]]
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    bars = ax.barh(labels, t[value_col] / 1e9, color=PALETTE["teal"], edgecolor="white")
    bars[-1].set_color(PALETTE["red"])
    for b, v in zip(bars, t[value_col] / 1e9):
        ax.text(b.get_width() * 1.06, b.get_y() + b.get_height() / 2,
                f"₹{v:,.1f} B", va="center", fontsize=9)
    ax.set_xlabel("Paid-up capital (₹ billion)")
    ax.set_xscale("log")
    ax.set_xlim(1e-1, t[value_col].max() / 1e9 * 4)
    ax.set_title(title)
    return _save(fig, name)


def plot_age_vs_survival(df: pd.DataFrame, name: str = "fig_status_by_era.png",
                         title: str = "Company status mix by registration era"):
    """100% stacked bars: status group composition per era (bias check)."""
    set_plot_style()
    era = pd.cut(df["RegistrationYear"], bins=[1850, 1990, 2000, 2010, 2027],
                 labels=["1850-1990", "1991-2000", "2001-2010", "2011-2026"])
    ct = pd.crosstab(era, df["CompanyStatusGroup"], normalize="index").mul(100)
    colors = {"Active": PALETTE["teal"], "Strike Off / in process": PALETTE["red"],
              "Converted / dissolved / liquidation": PALETTE["grey"],
              "Inactive / dormant": PALETTE["accent"], "Other / NA": "#B9C2CE"}
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bottom = np.zeros(len(ct))
    for col in ct.columns:
        ax.barh(ct.index.astype(str), ct[col], left=bottom, label=col,
                color=colors.get(col, "#B9C2CE"), edgecolor="white")
        for i, v in enumerate(ct[col]):
            if v > 6:
                ax.text(bottom[i] + v / 2, i, f"{v:.0f}%", va="center",
                        ha="center", fontsize=9, color="white", weight="bold")
        bottom += ct[col].values
    ax.set_xlabel("% of companies registered in era")
    ax.set_title(title)
    ax.legend(frameon=True, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    return _save(fig, name)
