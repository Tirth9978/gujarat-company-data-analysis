# EDA Project — Gujarat Company Master Data (IT462)

Exploratory Data Analysis of **222,915 companies** registered with the Registrar of
Companies (MCA master extract, state of Gujarat) between **1880 and 2026** — the final
project for **IT462 · Exploratory Data Analysis · Autumn Semester (AY 2026-27)**.

Four synchronised deliverables share one analysis pipeline, so every number matches
everywhere:

| Deliverable | Location | What it is |
|---|---|---|
| Live-demo notebook | `notebooks/EDA_Demo.ipynb` | Executable top-to-bottom, with explanations, charts and findings |
| Interactive dashboard | `website/app.py` | Streamlit + Plotly: filters, cleaning options, downloads |
| Analysis report | `reports/analysis_report.md` | Methodology, findings, limitations, conclusions |
| Presentation | `reports/eda_presentation.pptx` | 16-slide academic deck built from actual results |

## Project structure

```text
EDA_Project/
├── data/
│   ├── raw/EDA_Project_Dataset.csv          # original data — NEVER modified
│   └── processed/
│       ├── company_master_cleaned.csv       # cleaned + flagged + feature-enriched
│       └── analysis_results.json            # all headline statistics (single source of truth)
├── notebooks/EDA_Demo.ipynb                 # live academic demonstration
├── src/
│   ├── data_loader.py                       # import, overview, measurement-scale inventory
│   ├── data_cleaning.py                     # quality register, cleaning pipeline, features
│   ├── statistical_analysis.py              # descriptive, tests, correlation, fit, outliers
│   ├── visualization.py                     # consistent static figures
│   └── insights.py                          # evidence-attached insight generation
├── website/app.py                           # Streamlit + Plotly dashboard
├── reports/
│   ├── figures/                             # 14 publication-quality PNGs
│   ├── analysis_report.md
│   └── eda_presentation.pptx
├── tests/                                   # pytest suite for the src package
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1) environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) explore interactively
streamlit run website/app.py

# 3) walk the live-demo notebook
jupyter lab notebooks/EDA_Demo.ipynb                   # Kernel -> Restart & Run All

# 4) run the test suite
pytest tests/ -q

# 5) (optional) rebuild every statistic and figure from the raw data
python scripts/run_full_analysis.py                    # if scripts/ is present in your copy
```

> The raw CSV ships inside `data/raw/` so all three deliverables run out of the box.
> Every cleaning step operates on in-memory copies; the original file is read-only.

## Headline findings (all reproducible)

1. **Pristine identity spine** — 0 duplicate rows, 0 duplicate CINs in 222,915 records.
2. **Block missingness, not random** — 23.23% of rows miss *all* detail fields together;
   rate rises from ~0% (pre-2001) to **36.1% (2016–2026)** (χ² p < 0.001, Cramér's V = 0.384)
   → MAR/MNAR with an administrative cause; rows are flagged, never silently dropped.
3. **Extreme capital skew** — median authorised capital ₹1 lakh vs mean ₹11.2 crore
   (skewness 163, CV 7,433%); log/robust statistics throughout.
4. **Extreme concentration** — top 1% of companies hold **93.26%** of paid-up capital (Gini 0.992).
5. **Authorised ↔ paid-up capital** — Pearson r = 0.770 / Spearman 0.765 (log10);
   OLS slope 0.874, **R² = 0.5925** (n = 160,198).
6. **Registration boom** — 5,263 (2015) → **22,407 (2025)**, CAGR 15.59%; March peak seasonality.
7. **Sector signature** — Manufacturing 34.9% of 165,485 classified companies; professional
   services 17.9%; trade 10.4%.
8. **Genuine outliers kept** — Reliance Jio ₹1,531 B paid-up leads 18,050 log-scale outliers;
   10,925 zero-paid-up entries and 74 paid-up > authorised violations are flagged, not "fixed".

## Course-concept coverage (IT462)

Measurement scales · data import & inspection · duplicate detection · invalid entries ·
type conversion · encoding/format inconsistencies · missing-data mechanisms (MCAR/MAR/MNAR)
with chi-square evidence · five-number summary, IQR, MAD, CV · histograms, boxplots, bar
charts, crosstabs, heatmaps, scatter · Pearson & Spearman correlation · OLS fit with
goodness-of-fit · IQR/z-score outliers · Lorenz/Gini concentration · trend & seasonality ·
interactive dashboarding and data storytelling.

## Notes on data quality (honest treatment)

- 16 sentinel dates (`1111-01-01`) → set to `NaT`, never imputed.
- 39,392 rows store country as the calling code `91` → normalised to *India*
  (documented inference) alongside `USA`→`United States`-style fixes.
- 74 impossible `paid-up > authorised` rows and 10,925 zero-paid-up rows remain in the
  data with explicit flags — corrections belong to the registry, not the analyst.
- Capital statistics use the 171,142 detail-known rows; this selection is stated
  wherever relevant (missingness is era-driven — see report §3.1).

## Tech stack

Python 3.10+ · pandas · NumPy · SciPy · Matplotlib · Seaborn · Plotly · Streamlit ·
nbformat/nbclient · pytest

## Academic integrity

Prepared for IT462 coursework. AI assistance was used as a scaffold and is acknowledged
in the notebook; all quantitative results are computed live from the supplied dataset and
are reproducible from the code in this repository.
