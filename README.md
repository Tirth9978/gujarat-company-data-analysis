# EDA Project — Gujarat Company Master Data (IT462)

Exploratory Data Analysis of **222,915 companies** registered with the Registrar of Companies (MCA master extract, state of Gujarat) between **1880 and 2026** — the final project for **IT462 · Exploratory Data Analysis · Autumn Semester (AY 2026-27)**.

This project uses the official **Company Master Data** published by the Ministry of Corporate Affairs (MCA), Government of India, available through the [Open Government Data Platform](https://www.data.gov.in/catalog/company-master-data).

Four synchronised deliverables share one analysis pipeline, so every number matches everywhere:

| Deliverable | Location | What it is |
|---|---|---|
| Live-demo notebook | `notebooks/EDA_Demo.ipynb` | Executable top-to-bottom, with explanations, charts and findings |
| Interactive dashboard | `website/app.py` | Streamlit + Plotly: filters, cleaning options, downloads |
| Analysis report | `reports/analysis_report.md` | Methodology, findings, limitations, conclusions |
| Presentation | `reports/eda_presentation.pptx` | 16-slide academic deck built from actual results |

## Dataset and Data Setup

### Dataset Source

The project uses the **Company Master Data** dataset published by:

- **Publisher:** Ministry of Corporate Affairs (MCA), Government of India
- **Platform:** Open Government Data (OGD) Platform India
- **Official Dataset:** [Company Master Data — data.gov.in](https://www.data.gov.in/catalog/company-master-data)
- **Original Format:** CSV (available through ZIP download)

The original dataset contains company registration records. This project focuses specifically on companies registered in **Gujarat, India**.

### Download and Setup Instructions

The raw dataset is not included in this GitHub repository. Follow these steps to obtain it:

**Step 1 — Download the dataset**

1. Visit the [official Company Master Data portal](https://www.data.gov.in/catalog/company-master-data).
2. Navigate to the available dataset resources.
3. Download the dataset using the available ZIP Download option.
4. Extract the downloaded ZIP archive.

**Step 2 — Prepare the Gujarat dataset**

1. Locate the Company Master Data CSV file.
2. If the downloaded dataset contains records from multiple states, filter the records using the appropriate Registrar of Companies (ROC) or state-related field to retain Gujarat records.
3. Save the resulting Gujarat dataset as:

`EDA_Project_Dataset.csv`

**Step 3 — Place the dataset in the project directory**

Place the CSV file inside `data/raw/`:

```text
EDA_Project/
└── data/
    └── raw/
        └── EDA_Project_Dataset.csv
```

**Important:** The analysis in this repository is based on a specific dataset extract containing **222,915 Gujarat company records**. Downloading a newer or different version of the nationwide dataset may produce different record counts and statistical results. To reproduce the published findings exactly, use the same dataset extract and filtering criteria.

### Dataset Overview

| Attribute | Details |
|---|---|
| Dataset Name | Company Master Data |
| Publisher | Ministry of Corporate Affairs, Government of India |
| Source | [data.gov.in](https://www.data.gov.in/catalog/company-master-data) |
| Geographic Coverage | Gujarat, India |
| Records Analyzed | 222,915 |
| Registration Period | 1880–2026 |
| Original Format | CSV |
| Download Format | ZIP archive containing CSV data |

## Project Structure

```text
EDA_Project/
├── data/
│   ├── raw/
│   │   └── EDA_Project_Dataset.csv          # downloaded Gujarat dataset — NEVER modified
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

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/gujarat-company-data-analysis.git
cd gujarat-company-data-analysis
```

### 2. Download and prepare the dataset

Download the [Company Master Data](https://www.data.gov.in/catalog/company-master-data) dataset and follow the instructions in the **Dataset and Data Setup** section above.

Ensure that the Gujarat-filtered CSV is available at:

```text
data/raw/EDA_Project_Dataset.csv
```

### 3. Set up the Python environment

Python 3.10 or later is required.

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 4. Explore the interactive dashboard

```bash
streamlit run website/app.py
```

### 5. Run the live-demo notebook

```bash
jupyter lab notebooks/EDA_Demo.ipynb
```

Select **Kernel → Restart & Run All** to execute the complete analysis.

### 6. Run the test suite

```bash
pytest tests/ -q
```

### 7. Rebuild the analysis (optional)

If `scripts/run_full_analysis.py` is present in your copy:

```bash
python scripts/run_full_analysis.py
```

> **Data integrity:** The original CSV is never modified. Every cleaning and feature-engineering operation works on in-memory copies. Processed datasets, statistics, and visualizations are stored separately.

## Headline Findings (All Reproducible)

1. **Pristine identity spine** — 0 duplicate rows, 0 duplicate CINs in 222,915 records.

2. **Block missingness, not random** — 23.23% of rows miss *all* detail fields together; rate rises from ~0% (pre-2001) to **36.1% (2016–2026)** (χ² p < 0.001, Cramér's V = 0.384) → MAR/MNAR with an administrative cause; rows are flagged, never silently dropped.

3. **Extreme capital skew** — median authorised capital ₹1 lakh vs mean ₹11.2 crore (skewness 163, CV 7,433%); log/robust statistics throughout.

4. **Extreme concentration** — top 1% of companies hold **93.26%** of paid-up capital (Gini 0.992).

5. **Authorised ↔ paid-up capital** — Pearson r = 0.770 / Spearman 0.765 (log10); OLS slope 0.874, **R² = 0.5925** (n = 160,198).

6. **Registration boom** — 5,263 (2015) → **22,407 (2025)**, CAGR 15.59%; March peak seasonality.

7. **Sector signature** — Manufacturing 34.9% of 165,485 classified companies; professional services 17.9%; trade 10.4%.

8. **Genuine outliers kept** — Reliance Jio ₹1,531 B paid-up leads 18,050 log-scale outliers; 10,925 zero-paid-up entries and 74 paid-up > authorised violations are flagged, not "fixed".

## Course-Concept Coverage (IT462)

Measurement scales · data import & inspection · duplicate detection · invalid entries · type conversion · encoding/format inconsistencies · missing-data mechanisms (MCAR/MAR/MNAR) with chi-square evidence · five-number summary, IQR, MAD, CV · histograms, boxplots, bar charts, crosstabs, heatmaps, scatter · Pearson & Spearman correlation · OLS fit with goodness-of-fit · IQR/z-score outliers · Lorenz/Gini concentration · trend & seasonality · interactive dashboarding and data storytelling.

## Notes on Data Quality (Honest Treatment)

- 16 sentinel dates (`1111-01-01`) → set to `NaT`, never imputed.
- 39,392 rows store country as the calling code `91` → normalised to *India* (documented inference) alongside `USA` → `United States`-style fixes.
- 74 impossible `paid-up > authorised` rows and 10,925 zero-paid-up rows remain in the data with explicit flags — corrections belong to the registry, not the analyst.
- Capital statistics use the 171,142 detail-known rows; this selection is stated wherever relevant (missingness is era-driven — see report §3.1).
- The raw dataset is preserved as downloaded. Data cleaning is performed on separate in-memory copies.
- Results correspond to the specific Gujarat dataset extract used during this project. Different dataset versions may produce different results.

## Tech Stack

Python 3.10+ · pandas · NumPy · SciPy · Matplotlib · Seaborn · Plotly · Streamlit · nbformat/nbclient · pytest.

## Repository Data Policy

The original dataset is obtained from the official [data.gov.in Company Master Data catalog](https://www.data.gov.in/catalog/company-master-data).

To keep this repository lightweight:

- The raw CSV is not committed to GitHub.
- Users must download the dataset independently from the official source.
- The raw dataset should be placed in `data/raw/EDA_Project_Dataset.csv`.
- Processed datasets and analysis outputs are generated separately.
- Please refer to the official portal for applicable dataset licensing and redistribution terms.

## Academic Integrity

Prepared for **IT462 · Exploratory Data Analysis · Autumn Semester (AY 2026–27)**.

AI assistance was used as a scaffold and is acknowledged in the notebook. All quantitative results are computed from the supplied dataset and are reproducible from the code in this repository.
