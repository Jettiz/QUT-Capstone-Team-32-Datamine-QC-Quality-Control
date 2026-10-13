# QUT-Capstone-Team-32-Datamine-QC-Quality-Control
The repo for the Datamine QC Quality Control AI Solution

# Datamine QC Quality Control

An automated quality-control (QC) anomaly detection system for laboratory analytical chemistry results, developed as a QUT capstone project with Datamine.

## About the project

Analytical laboratories run several kinds of quality-control samples alongside every batch of real client samples, to catch instrument drift, contamination, and measurement error before results are reported. Reviewing these QC results manually is slow and inconsistent. This project analyses real laboratory exports (CCLAS-style `ResultSet.csv` exports) and automatically flags QC results that look anomalous, so a chemist can focus review effort on what actually needs attention instead of re-checking everything.

### What it analyses

A lab export contains several distinct QC sample types, identified by the `ANALYTICAL_TYPE` column (plus supporting fields like `STD_CODE`/`SCHEME_CODE`):

| Sample type | Purpose |
|---|---|
| **Blank** | A sample with no analyte added — checks for contamination or carryover from a previous sample. |
| **Control / LCS** (Laboratory Control Sample) | A reference material with a known concentration, run repeatedly over time — checks instrument accuracy and long-term drift. |
| **SRM** (reference material, e.g. OREAS-branded standards) | Also a known reference material, assessed against its own historical result distribution. |
| **Replicate** / **Duplicate** | The same sample measured twice — checks measurement precision (how well the two results agree). |
| **Matrix Spike** | A real sample with a known amount of analyte added — checks recovery accuracy in the presence of the sample's own matrix. |

### Methods applied

- **Control (LCS) drift detection** (`src/detectors/control_detector.py`, reference: `notebooks/LCS_drift_detection_historic.ipynb`) — analyses Standard rows whose `STD_LOT_CODE` is `Sample` (every other Standard row goes to SRM; the split lives in `src/data_validator.py`). Jobs are processed in date order against a per-analyte rolling history (up to 30 observations, de-duplicated) and each result is scaled against its own target/limits (target → 0, failure limit → ±1). Each analyte is classified on two axes: has it **already breached** a limit (point check, always applied), and is a robust trend (Theil-Sen slope, resistant to single outliers) over recent history **heading toward** a limit. The outcome maps to three states: **Fail** (outside the failure limits), **Warning** (in the warning band or trending toward a limit) and **Pass**.
- **SRM anomaly detection** (`src/detectors/srms_detector.py`) — extracts SRM-candidate results, computes statistical features (deviation %, robust Z-score, rolling drift, distance to limits), applies explainable rule-based flags, and combines them with an unsupervised Isolation Forest model into a prioritised `Low / Medium / High / Critical` risk score.
- **Blank, Duplicate, Replicate, Matrix Spike** — data validation (required columns present, correctly typed, complete) is implemented for all four in `src/data_validator.py`. Anomaly-detection logic for these is planned but not yet implemented (see Project status).

### Project status

| Sample type | Data validation | Anomaly detection | Chart generation |
|---|---|---|---|
| Control (LCS) | ✅ | ✅ (in the POC) | ✅ |
| SRM | ✅ | ✅ | — |
| Blank | ✅ | 🔲 planned | — |
| Duplicate | ✅ | 🔲 planned | — |
| Replicate | ✅ | 🔲 planned | — |
| Matrix Spike | ✅ | 🔲 planned | — |

*(Matrix Spike Duplicate was evaluated and removed from project scope.)*

## Solution structure

```
├── config/            # Per-sample-type YAML configuration (thresholds, windows) +
│                       # column_config.csv (raw column -> internal column mapping)
├── data/
│   ├── raw/            # Raw CCLAS exports (kept local only, see Setup below)
│   ├── processed/       # Derived/persisted state, e.g. the LCS rolling history CSV
│   └── samples/         # Small scenario-based fixtures (Pass/Warn/Fail/Ignored per type)
├── notebooks/          # Exploratory analysis + reusable test-harness notebooks
├── poc/                # Client-side proof-of-concept report (see below)
├── docs/               # QC_INTEGRATION_GUIDE.md: how to add a QC method to the POC
├── src/
│   ├── data_loader.py    # Loads a raw export and maps it to the internal column names
│   ├── data_validator.py # Validates each sample type's data + routes Standard rows (LCS vs SRM)
│   ├── qc_status.py      # The three shared warning states: FAIL / WARNING / PASS
│   ├── qc_report/        # Result contract + registry + per-method adapters used by the POC
│   ├── detectors/        # One anomaly detector per sample type
│   └── visualisers/      # One chart generator per sample type
├── tests/              # pytest suite
└── requirements.txt
```

### Proof-of-concept report (`poc/`)

A small local web app: plain HTML/CSS/JS pages served by a standard-library Python server (`poc/server.py`, no extra dependencies) that runs the real analysis from `src/`. Control (LCS) shows real results; the other detectors show placeholder data until they are integrated (see `docs/QC_INTEGRATION_GUIDE.md`).

1. **Load a sample** (`index.html`) — pick a CSV export and choose **Use history** (the older half of the file's jobs becomes the history, the newer half is analysed; the history store is rebuilt each run) or **No history** (nothing stored; each job is only compared with earlier jobs in the same file). The server loads, validates and analyses the file.
2. **Detector overview** (`summary.html`) — one card per detector (real Fail/Warning/Pass counts for Control), a per-job table with **Job** and **Instrument**, and a real-data table of the file's rows.
3. **Detail** (`detail.html`) — one job at a time (job selector when the file has several): Job and Instrument, one button per `ANALYTE_CODE` coloured by its state, scheme tabs when an analyte was measured under two schemes, the reason for the state, the supporting numbers, and the LCS control chart.

#### Running the proof of concept

```bash
pip install -r requirements.txt
python poc/server.py          # then open http://localhost:8000
```

Choose a CSV (e.g. `data/raw/ResultSet.csv` or a file in `data/samples/`), click **Analyse sample**, **Continue to overview**, then **View details**. Opening the HTML files directly from disk only shows a notice: the analysis needs the server.

## Setup (for running the Python pipeline / notebooks / tests)

Requires Python 3.10–3.13.

```bash
pip install -r requirements.txt
```

Raw data files (`*.csv`, `*.xlsx`) are not committed to the repo (see `.gitignore`) — place your own CCLAS export at `data/raw/ResultSet.csv` (and, for Matrix Spike, `data/raw/QC_Anomaly_Training_Data_v2.xlsx`) before running the loader, detectors, or notebooks against real data.

Run the test suite with:

```bash
pytest
```
