# Exploratory Data Analysis Report
## Gujarat Company Master Data — 222,915 companies, 1880–2026

**Course:** IT462 · Exploratory Data Analysis · Autumn Semester (AY 2026-27) · Final Project
**Dataset:** `data/raw/EDA_Project_Dataset.csv` (63 MB, 222,915 × 16) — MCA company master extract, ROC Ahmedabad / state of Gujarat
**Companion artifacts:** `notebooks/EDA_Demo.ipynb` · `website/app.py` (Streamlit dashboard) · `reports/eda_presentation.pptx` · `data/processed/analysis_results.json` (single source of truth)
**Reproducibility:** every number in this report is computed by `src/` + `scripts/run_full_analysis.py`; no result is hard-coded.

---

## 1. Problem statement

The project answers three questions about Gujarat's formal corporate sector using a public registry extract (data.gov.in / MCA style), before any modelling takes place:

1. **What does the registry actually contain?** Structure, measurement scales, and — critically — *content quality*: duplicates, invalid entries, inconsistent formats, and missing-data mechanisms.
2. **What does corporate Gujarat look like?** Class/status/listing composition, capital distributions, sector mix, concentration, and the relationship between authorised and paid-up capital.
3. **How has registration evolved?** Long-run trend (1880–2026) and within-year seasonality.

The analysis is deliberately **descriptive-exploratory** (per the course brief): classical statistical EDA with Python, emphasising visualisation, robust statistics and transparent treatment of data-quality problems rather than predictive modelling.

## 2. Methodology

### 2.1 Workflow
```
raw CSV (read-only)
   → structure & measurement-scale inventory      (data_loader.py)
   → quality register: duplicates, invalid dates,
     rule violations, block missingness           (data_cleaning.py)
   → missingness mechanism tests (chi-square)     (statistical_analysis.py)
   → reproducible cleaning pipeline + feature
     engineering, logged step-by-step             (data_cleaning.py)
   → descriptive / bivariate / correlation / fit /
     outlier / concentration / time-series stats  (statistical_analysis.py)
   → static figures (matplotlib) + results JSON   (visualization.py, run_full_analysis.py)
   → insight generation with evidence strings     (insights.py)
```

### 2.2 Measurement scales (course module)
| Field | Scale | Role |
|---|---|---|
| CIN, names, ROC, status, class, listing, state, country, NIC | Nominal | identity / categories |
| AuthorizedCapital, PaidupCapital | Ratio (INR, true zero) | numeric analysis |
| CompanyRegistrationdate_date | Interval (datetime) | time-series EDA |

`nic_code` arrives as a float but is a **coded nominal** sector; the date column arrives as text; both are converted in the cleaning stage.

### 2.3 Cleaning decisions (logged, reversible)
| Issue | Evidence | Treatment |
|---|---|---|
| Duplicate rows / duplicate CINs | 0 / 0 | none needed (identity pristine) |
| Sentinel dates `1111-01-01`, `1111-11-11` | 16 rows | set to `NaT`, counted in `n_invalid_dates` |
| Country stored as calling code `91` | 39,392 rows | normalised to *India* (documented inference: +91); `USA`→`United States` etc. |
| Paid-up > Authorised capital | 74 rows | flagged (`CapitalViolation`), **not** deleted — a registry-reality check |
| Zero paid-up / zero authorised capital | 10,925 / 2,209 rows | kept, flagged; analysed as the low-side special case |
| Block-missing detail fields (7 columns) | 51,773 rows (23.23%) | kept + `block_missing` flag; capital statistics restricted to the 171,142 detail-known rows |

The cleaned, feature-enriched table is exported to `data/processed/company_master_cleaned.csv` (222,915 rows; original file untouched). Derived features: `RegistrationYear/Month`, `CompanyStatusGroup` (18 → 4 groups), `IsListed`, `IsForeign`, `CapitalRatio`, `NICSection` (5-digit NIC 2008 → 21 letter sections), `CapitalBucket`, `block_missing`.

## 3. Data quality findings

### 3.1 Missing data — a *block* pattern, not random noise
Missingness is concentrated in eight detail fields with **identical counts (51,773)** — rows either have *all* details or *none*, while identifier fields (CIN, name, date, state, status) are 100% populated. This block structure alone already rules out independent per-cell MCAR.

Chi-square tests against observed variables confirm dependence:
- **vs registration era:** χ² = 32,871, p < 0.001, **Cramér's V = 0.384**. Block-missingness is ~0% for companies registered before 2001 but **36.1% for the 2016–2026 cohort**.
- **vs status group:** χ² = 10,667, p < 0.001, V = 0.219. *Active* companies miss details most (28.6%); struck-off companies only 7.5%.

**Verdict: MAR/MNAR with an administrative cause** — recent registrations in this extract simply have not had their detail fields populated (the opposite of the "old records are incomplete" prior). Practical consequence: case-wise deletion would silently remove one-third of the newest cohort, so rows are flagged, and capital statistics use the detail-known subset (171,142 rows), stated wherever relevant.

### 3.2 Invalid entries & inconsistencies
- 16 sentinel dates (year `1111`) — impossible values that parse as valid ISO dates, caught by domain rule, not by the parser.
- 74 companies report **paid-up > authorised capital** — impossible under the Companies Act; a registry-quality signal worth auditing upstream.
- Country field mixes names, `NaN`, and the calling code `91` (39,392 rows); cosmetic duplicates (`USA` vs `United States`) unified.
- ROC anomaly: 118 companies registered to *ROC Delhi I* inside a Gujarat extract — small but real provenance noise.

## 4. Descriptive summary (detail-known subset, n = 171,142)

| Statistic | AuthorizedCapital (₹) | PaidupCapital (₹) |
|---|---|---|
| Median (Q2) | 100,000 | 100,000 |
| Mean | 111,647,082 | 58,892,976 |
| Std deviation | 8.30 × 10⁹ | 5.56 × 10⁹ |
| IQR (Q3−Q1) | 1,400,000 | 400,000 |
| MAD (about median) | 80,000 | 60,000 |
| Coefficient of variation | **7,432.7%** | 9,443.9% |
| Skewness | **163.3** | 201.5 |
| Min / Max | 0 / 1.9 × 10¹² | 0 / 1.53 × 10¹² |

Five-number summary (authorised): **min 0 · Q1 ₹1 L · median ₹1 L · Q3 ₹15 L · max ₹19,000 Cr**.

With skewness above 160 and CV above 7,000%, the mean is a misleading summary — all downstream analysis uses **log10 transforms and robust statistics** (median, IQR, MAD), as the course's descriptive-summary module prescribes. On the log10 scale the authorised-capital distribution is moderately skewed (skew 1.24, std 0.95), i.e. approximately log-normal in shape. The mode at ₹1 lakh is the statutory default capital — visible as spikes at 10⁵ in every histogram.

## 5. Univariate analysis

**Categorical (nominal) profiles.** Class: **Private 157,169 (91.8%)**, Public 10,641 (6.2%), One Person Company 3,332 (1.9%). Status groups: **Active 74.7%**, Strike Off / in process 22.3%, Converted/dissolved/liquidation 2.9%, Inactive/dormant 0.2%. Listing: **99.2% unlisted** (1,327 listed, 0.78%). Origin after normalisation: Indian 171,141, foreign 119 (≈ 0.07%, led by USA, Singapore, UK).

**Sector mix (NIC-2008 sections, n = 165,485 classified).** Manufacturing **C: 57,769 (34.9%)**, Professional & scientific **M: 29,682 (17.9%)**, Wholesale & retail trade **G: 17,276 (10.4%)**, Transport & storage H: 14,745 (8.9%), Information & communication J: 8,207 (5.0%), Financial & insurance K: 7,120 (4.3%). A textbook Gujarat signature — industry-led with a deep professional-services tail.

**Numeric (ratio-scale) distributions.** Both capitals are violently right-skewed on the raw scale; log10 histograms show the ₹1-lakh pile-up plus round-number preference (10⁶, 10⁷, …). Boxplots confirm medians at log10 = 5 with long upper whiskers.

## 6. Outliers and concentration

- **Tukey fences (1.5×IQR, log10 paid-up):** 18,050 outliers (11.27%), fences ≈ ₹4,419 – ₹1.81 Cr.
- **Z-score (|z| > 3, log10):** 1,479 outliers (0.92%).
- The high-side outliers are **genuine mega-corporates, not errors**: Reliance Jio Infocomm (₹1,531 B paid-up), Vodafone Idea (₹1,083 B), Jio Digital Fibre (₹787 B), Sardar Sarovar Narmada Nigam (₹735 B), Gujarat Urja Vikas Nigam (₹456 B), ArcelorMittal Nippon Steel (₹315 B)… They are retained and handled with log/robust methods.
- **Concentration is extreme.** Lorenz curve: **top 1% of companies hold 93.26% of all paid-up capital (Gini = 0.992)**; top 5% hold 98.0%. Corporate capital in Gujarat is effectively a two-tier system: a handful of giant groups vs a sea of ₹1-lakh shells.

## 7. Bivariate analysis

**Class × status (crosstab + χ²):** χ² = 2,499.1, dof 8, p < 0.001, **Cramér's V = 0.085** — a statistically significant but modest association (expected with n ≈ 171k). The informative pattern: **One Person Companies are 87.5% active** (post-2013 vehicles, little time to fail), Private 69.8% active, **Public only 58.6% active** with a 30.4% strike-off share — many old public companies are defunct shells.

**Capital × class (grouped medians, log boxplots):** public companies carry visibly higher authorised-capital medians than private/OPC — infrastructure and utility heavyweights (state PSUs, GMRC, energy) live in the public class.

## 8. Correlation analysis & data fitting

Computed on the detail-known subset with both capitals > 0 (n = 160,198 pairs), log10 space:

| Statistic | Value |
|---|---|
| Pearson r (log10) | **0.7697** |
| Spearman ρ | **0.7650** |
| OLS slope (log-log) | **0.8743** (SE 0.0018) |
| Intercept | 0.3881 |
| **R²** | **0.5925** |
| RMSE (log10 units) | 0.696 |
| p-value | < 0.001 |

Pearson ≈ Spearman tells us the association is monotone and not outlier-driven. The slope 0.874 means a 10× increase in authorised capital is associated with a **~7.5×** increase in paid-up capital. R² ≈ 0.59: authorised capital explains roughly three-fifths of the (log) variation in paid-up capital — authorisation is a *ceiling commitment*, not a capitalisation guarantee. The hexbin scatter also exposes the ₹1-lakh horizontal stripe (default paid-up) and zero-capital corner as structural artefacts.

## 9. Time-series EDA

**Trend.** Registrations per year: pre-1991 trickle (≤ ~1,000/yr), post-liberalisation lift with a 1994–97 mini-boom, and an **explosive post-2020 phase**: 5,263 (2015) → 13,607 (2021) → **22,407 (2025, peak)** — **CAGR 15.59% (2015→2025, full years)**. 2026 already records 12,261 by June (partial year, excluded from CAGR).

**Seasonality.** March is the strongest month (21,134 cumulative registrations — the Indian financial year-end rush), November the weakest (14,972); a real but moderate annual cycle (~±13% around the monthly mean).

**Survivorship warning.** Status composition differs sharply by era (e.g. 1850–1990 cohort is majority struck-off/converted; 2011–2026 cohort is ~86% active). Any "survival" comparison must adjust for company age — current status mix is not comparable across eras.

## 10. Key insights (evidence-attached)

1. **A 146-year registry of Gujarat corporate India** — 222,915 companies, 1880–2026, 100% state code `gujarat`; a census of one state, not a sample.
2. **Identity keys are pristine** — 0 duplicate rows, 0 duplicate CINs.
3. **23.23% of rows miss ALL detail fields together — and the pattern is NOT random** — χ² V = 0.384 (era), 36.1% missing for 2016–2026 vs ~0% pre-2001 → MAR/MNAR, administrative cause.
4. **Corporate capital is extremely right-skewed** — median ₹1 L vs mean ₹11.16 Cr (authorised), skewness 163, CV 7,433% → log/robust stats mandatory.
5. **Top 1% of companies hold 93.26% of paid-up capital** (Gini 0.992) — extreme concentration.
6. **Authorised and paid-up capital move together (R² = 0.5925)** — Pearson 0.770 / Spearman 0.765 (log10), slope 0.874 → 10× authorised ≈ 7.5× paid-up.
7. **Company class and legal status are significantly dependent** — χ² = 2,499, p < 0.001, V = 0.085; OPC 87.5% active vs Public 58.6%.
8. **Registrations exploded to a peak of 22,407 in 2025** — CAGR 15.59% (2015–2025); March peak / November trough seasonality.
9. **18,050 capital outliers on the log scale — and they are real giants** — IQR fences ₹4,419–₹1.81 Cr; Reliance Jio ₹1,531 B paid-up; 10,925 zero-paid-up entries need separate treatment.
10. **Manufacturing leads with 34.9% of classified companies** — then professional & scientific (17.9%) and trade (10.4%), of 165,485 NIC-classified firms.

## 11. Limitations

1. **Registry, not economy.** No turnover, employment or filing-frequency data; registered capital is a commitment, not activity.
2. **Detail-field missingness (23.2%)** induces a documented selection effect: capital statistics describe the 171,142 detail-known companies and under-represent the newest cohort (§3.1).
3. **Snapshot censoring.** Extract ends June 2026; 2026 counts partial, excluded from growth-rate claims.
4. **Status ≠ survival time.** No closure dates; era comparisons face survivorship bias (§9).
5. **No silent fixes.** 74 paid-up > authorised rows, 10,925 zero-paid-up rows and 16 sentinel dates are flagged, never altered or imputed — corrections belong upstream at the registry.
6. **Address parsing deferred.** Free-text addresses were not geocoded; district-level analysis is future work.

## 12. Conclusions

Gujarat's company registry is a **content-quality case study wrapped around a clean identity spine**: zero duplicates, but a structural, era-driven block of missing details that any naive pipeline would mishandle. Substantively, the state's corporate landscape is **private-dominated (91.8%), manufacturing-led (34.9% of classified firms), and extraordinarily concentrated** — the top 1% of companies hold 93% of paid-up capital while the median company sits at the ₹1-lakh statutory floor. Authorised and paid-up capital co-move strongly on the log scale (r ≈ 0.77, R² ≈ 0.59, slope 0.87), and registrations follow a structural upward march (22.4k in 2025, CAGR 15.6% since 2015) with a March financial-year seasonality. All findings are reproducible end-to-end via the `src/` package, demonstrable live in the Streamlit dashboard, and packaged in the accompanying notebook, presentation and test suite.

## 13. Suggested next steps

- Logistic / survival modelling of strike-off risk using age-adjusted cohorts.
- Join with MCA XBRL financial filings to move from registered capital to activity measures.
- Geocode registered-office addresses for district-level heatmaps on the dashboard.
- AROC-wise comparison by joining sibling state extracts from data.gov.in.
