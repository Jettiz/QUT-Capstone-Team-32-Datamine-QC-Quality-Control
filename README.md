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

- **Control (LCS) drift detection** (`src/detectors/control_detector.py`) — maintains a persistent, per-analyte rolling history (up to 30 past observations) and scales each new result against its own target/limits (target → 0, failure limit → ±1). Every result is classified on two axes: has it **already breached** a limit right now (a direct point check), and is a robust trend (Theil-Sen slope, resistant to single outliers) over recent history **heading toward** a limit (a predictive check). The two combine into one severity ladder: `Critical > High > Medium > None`.
- **SRM anomaly detection** (`src/detectors/srms_detector.py`) — extracts SRM-candidate results, computes statistical features (deviation %, robust Z-score, rolling drift, distance to limits), applies explainable rule-based flags, and combines them with an unsupervised Isolation Forest model into a prioritised `Low / Medium / High / Critical` risk score.
- **Blank, Duplicate, Replicate, Matrix Spike** — data validation (required columns present, correctly typed, complete) is implemented for all four in `src/data_validator.py`. Anomaly-detection logic for these is planned but not yet implemented (see Project status).

### Project status

| Sample type | Data validation | Anomaly detection | Chart generation |
|---|---|---|---|
| Control (LCS) | ✅ | ✅ | ✅ |
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
├── src/
│   ├── data_loader.py    # Loads a raw export and normalises it to one internal schema
│   ├── data_validator.py # Validates each sample type's data ahead of detection
│   ├── detectors/        # One anomaly detector per sample type
│   └── visualisers/      # One chart generator per sample type
├── tests/              # pytest suite
└── requirements.txt
```

### Proof-of-concept report (`poc/`)

A static, browser-only preview of the intended report experience — plain HTML/CSS/JS, no backend, no build step. It demonstrates the full intended flow end-to-end using a mix of real and dummy data:

1. **Load a sample** (`index.html`) — pick a real local CSV; the browser checks (client-side) which detectors have the columns and row types they need to run against that file.
2. **Detector overview** (`summary.html`) — one card per sample type showing whether it's usable against the loaded file, plus a real-data table (filterable by sample type, detected-issue severity, and analyte) so the underlying rows can be visually sanity-checked.
3. **Detail** (`detail.html`) — a per-sample-type anomaly viewer. Its sample-type filter is permanently locked (not just disabled) so there's no way to accidentally view another sample type's data while reviewing one. Control's page shows **real** generated charts and real detector output for 5 analytes; every other sample type currently shows dummy placeholder data pending its detector's implementation.

#### Running the proof of concept

No installation needed — it's static files.

1. Open `poc/index.html` directly in a browser (double-click it, or use an editor's "Open with Live Server").
2. On the Load page, choose a CSV file to try — for example one of the fixtures in `data/samples/`, or a real export if you have one locally.
3. Click **Analyse sample**, then **Continue to overview** to see which detectors are usable and browse the real data table.
4. Click **View details** on any usable detector card to open its detail page.

## Setup (for running the Python pipeline / notebooks / tests)

```bash
pip install -r requirements.txt
```

Raw data files (`*.csv`, `*.xlsx`) are not committed to the repo (see `.gitignore`) — place your own CCLAS export at `data/raw/ResultSet.csv` (and, for Matrix Spike, `data/raw/QC_Anomaly_Training_Data_v2.xlsx`) before running the loader, detectors, or notebooks against real data.

Run the test suite with:

```bash
pytest
```
