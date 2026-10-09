# Integrating a QC method into the proof of concept

This guide explains how to connect your QC detector (Blank, Duplicate, Replicate, SRM or Matrix Spike) to the proof of concept (POC) in `poc/`, so it shows **real** results instead of placeholder data. Control (LCS) is already connected and is used as the worked example throughout.

You do **not** need to write any JavaScript. You add one Python adapter and one registration line; the server and the three POC pages pick your method up automatically.

---

## 1. How the pieces fit together

```
 Browser (poc/*.html, *.js)                       Python (poc/server.py + src/)
 ──────────────────────────                       ─────────────────────────────
 Step 1  index.html ── upload CSV + History ──►  POST /api/analyse
                                                    │
                                                    ├─ src/data_loader.load_qc_data()      raw -> internal columns
                                                    │
                                                    └─ for every method in src/qc_report/registry.py:
                                                          method.run(df, use_history)
                                                            ├─ validate     (src/data_validator.validate_*_data)
                                                            ├─ select rows  (src/data_validator.select_*_rows)
                                                            ├─ analyse      (src/detectors/<your>_detector.py)
                                                            └─ map result → QCMethodResult  (src/qc_report/contract.py)
 Step 2  summary.html ◄── run summary (counts, jobs) ──┘
 Step 3  detail.html  ◄── GET /api/runs/<run>/methods/<id>/jobs/<n>     one job's items
                      ◄── GET /api/runs/<run>/methods/<id>/charts/<item>.png
                                                    └─ method.render_chart(item) (src/visualisers/<your>_visualize.py)
```

Everything left of the arrow is shared and method-agnostic. Everything your method needs to provide sits in the right-hand column.

---

## 2. Files: what you add, what you change, what you leave alone

| File | Action | LCS example |
|---|---|---|
| `src/detectors/<your>_detector.py` | Your detector (exists already). Make it accept the **internal** column names (section 3). | `src/detectors/control_detector.py` |
| `src/visualisers/<your>_visualize.py` | A function that draws **one item** as a PNG from data you already computed. | `plot_evaluation()` in `src/visualisers/control_visualize.py` |
| `src/data_validator.py` | Add/adjust `select_<your>_rows(df)` (routing) and make `validate_<your>_data(df)` use it. | `select_lcs_rows()`, `validate_lcs_data()` |
| `src/qc_report/<your>_method.py` | **New.** The adapter: validate → select → run detector → map to the contract. | `src/qc_report/lcs_method.py` |
| `src/qc_report/registry.py` | Add **one line**: `register(YourMethod())`. | `register(LCSMethod())` |
| `poc/data.js` | In your detector's entry, set `serverAnalysis: true` and delete its placeholder `ITEMS`. | the `control` entry |
| `config/<your>_config.yaml` | Your thresholds (optional). | `config/lcs_config.yaml` |
| `tests/test_<your>_*.py` | Tests for your adapter (copy `tests/test_qc_report.py`). | `tests/test_qc_report.py` |

**Do not modify** (shared by every method): `poc/server.py`, `poc/api.js`, `poc/app.js`, `poc/render.js`, `poc/ranking.js`, `poc/summary.js`, `poc/load.js`, `src/qc_report/contract.py`, `src/qc_status.py`, `src/data_loader.py` (except a new row in `config/column_config.csv`, see section 3). If you think one of them needs a change, raise it with the team first: a change there affects every method.

---

## 3. Input: what your detector receives

Your adapter's `run(df, use_history)` receives the **whole uploaded file** as returned by `src/data_loader.load_qc_data()`:

- **Internal, lowercase column names** from `config/column_config.csv`, never the raw export names. For example `analytical_type`, `std_lot_code`, `std_code`, `scheme_code`, `job_code`, `instrument_id`, `analyte_code`, `analysed_date`, `measured_value` (= `NUMERIC_FINAL_VALUE`), `target_value`, `limit_min`, `limit_max`, `limit_min_warning`, `limit_max_warning`, `limit_*_inclusive`, `unit_code`, `standard_status`, `precision_status`, `parent_value`, plus the derived `rpd` and `mean_conc` (Duplicate/Replicate).
- **Typed** already: dates are datetimes, numbers are floats, text columns are strings with missing values as `NA`.
- **All sample types**, so your adapter must select its own rows (section 4).

**Aliases.** Different exports name the same field differently. The loader merges them into one internal column, preferring the row listed first in `config/column_config.csv`:

| Internal column | Source columns, in priority order |
|---|---|
| `job_code` | `JOB_NAME_ANON`, then `JOB_CODE` |
| `instrument_id` | `INSTRUMENT_CODE`, then `INSTRUMENT_ID` |

If your detector needs a column that isn't mapped yet, add a row to `config/column_config.csv` (`source_column,internal_column,dtype,required`). Use `required=false` unless every supported export has it.

> **Detector still on raw names?** (This currently applies to `srms_detector.py`.) Either migrate it to the internal names, or have your adapter rename the selected rows back before calling it. The `source_column → internal_column` pairs in `config/column_config.csv` give you that mapping. Keep the rename inside the adapter, not in shared code.

---

## 4. Sample selection (routing): where QC-specific filtering happens

**Which rows belong to which QC method is decided in `src/data_validator.py`, and nowhere else.** Your validator, your detector and your adapter all call the same `select_<your>_rows(df)`, so the rule exists exactly once.

Current routing:

| Method | Rows | Function |
|---|---|---|
| Control (LCS) | `analytical_type == "Standard"` **and** `std_lot_code == "Sample"` (case-insensitive) | `select_lcs_rows()` |
| SRM | every other `Standard` row | `select_srm_rows()` |
| Blank / Duplicate / Replicate / Spike | add `select_blank_rows()` etc. following the same pattern | to add |

Rules:

- Compare text case-insensitively and trimmed, using the existing `_normalised()` helper.
- If two methods share an `analytical_type`, make their selections **complementary** (as LCS and SRM are) and add a test proving they don't overlap (see `test_lcs_and_srm_partition_the_standard_rows` in `tests/test_data_validator.py`).
- Data-quality filtering that is specific to your *analysis* (e.g. dropping rows with missing limits) belongs in your detector, and should be **counted and reported** rather than silently dropped. LCS returns a `validation_report` with these counts, and the adapter turns it into a notice ("Some LCS rows were skipped: …").
- Mirror your rule in `poc/rows.js` → `classifyRowType()` **only** if the browser preview must tell your rows apart (it does this for LCS vs SRM). The Python function stays the source of truth.

---

## 5. The adapter: the contract your method must satisfy

The server only calls three methods (see the `QCMethod` Protocol in `src/qc_report/registry.py`):

```python
class QCMethod(Protocol):
    id: str      # the POC detector id from poc/data.js, e.g. "duplicate"
    label: str   # display name, e.g. "Duplicate"

    def validate(self, df) -> dict: ...                       # {"usable", "n_rows", "missing_columns", "notes", ...}
    def run(self, df, *, use_history: bool) -> QCMethodResult: ...
    def render_chart(self, item: QCItem, output_path: Path) -> Path: ...
```

A minimal skeleton, structured exactly like `src/qc_report/lcs_method.py`:

```python
# src/qc_report/duplicate_method.py
from pathlib import Path
from ..data_validator import select_duplicate_rows, validate_duplicate_data
from ..detectors.duplicate_detector import DuplicateDetector
from ..qc_status import FAIL, PASS, WARNING
from ..visualisers.duplicate_visualize import plot_item          # your one-item chart
from .contract import QCItem, QCJob, QCMethodResult, display_instrument, display_instruments, display_job


class DuplicateMethod:
    id = "duplicate"          # must equal the id in poc/data.js
    label = "Duplicate"

    def validate(self, df):
        report = validate_duplicate_data(select_duplicate_rows(df))
        usable = not report["missing_columns"] and report["n_rows"] > 0
        return {**report, "usable": usable,
                "notes": [] if usable else ["Why it can't run, in plain language."]}

    def run(self, df, *, use_history):
        validation = self.validate(df)
        if not validation["usable"]:          # never raise for "cannot run"
            return QCMethodResult(self.id, self.label, False, "on" if use_history else "off",
                                  validation, validation["notes"], [])

        rows = select_duplicate_rows(df)
        result = DuplicateDetector().detect(rows)          # your analysis, unchanged

        jobs = []
        for n, (job_code, job_rows) in enumerate(...):      # group your results per job
            items = [self._to_item(f"{n}-{i}", r) for i, r in enumerate(job_rows)]
            jobs.append(QCJob(key=str(n), job=display_job(job_code),
                              instruments=display_instruments(...), first_date="...", items=items))
        return QCMethodResult(self.id, self.label, True, "on" if use_history else "off",
                              validation, ["One-line explanation of what was analysed."], jobs)

    def _to_item(self, item_id, r):
        return QCItem(
            id=item_id,                   # unique within this method's result
            code=r["analyte_code"],       # what the user selects by
            variant="",                   # e.g. scheme when one analyte has several results per job
            state=FAIL if ... else WARNING if ... else PASS,   # YOUR rule (section 6)
            status_detail="RPD_ABOVE_Q95",                     # your own finer outcome name
            reason="RPD 18.5% exceeds the historical Q95 (12.1%) for this concentration.",
            magnitude=abs(r["rpd"]),      # tie-break between items with the same state
            metrics=[{"label": "RPD", "value": "18.5%"}, ...], # already-formatted text
            job=display_job(r.get("job_code")),
            instrument=display_instrument(r.get("instrument_id")),
            has_chart=True,
            chart_payload=r,              # whatever render_chart needs; stays on the server
        )

    def render_chart(self, item, output_path):
        return plot_item(item.chart_payload, output_path)
```

Then register it:

```python
# src/qc_report/registry.py  (bottom of the file)
register(LCSMethod())
register(DuplicateMethod())
```

### Result format (what the browser receives)

`QCMethodResult.to_summary()` (Step 2), and `QCJob.to_dict()` (Step 3, one job at a time). This is a real LCS item from `ResultSet.csv`:

```json
{
  "id": "1956",
  "code": "PB",
  "variant": "GE_ICP40Q12",
  "state": "FAIL",
  "status_detail": "UPPER_FAILURE",
  "reason": "Latest result has already breached the upper failure limit.",
  "magnitude": 7.09,
  "metrics": [
    {"label": "Result", "value": "75.65 mg/kg"},
    {"label": "Target", "value": "23.5 mg/kg"},
    {"label": "Failure limits", "value": "16.15 – 30.85 mg/kg"},
    {"label": "Offset", "value": "+7.095 (target = 0, failure limits = ±1)"},
    {"label": "Company status (STANDARD_STATUS)", "value": "IgnoredUpperFailure"}
  ],
  "job": "TSV_LB0015957009",
  "instrument": "Instrument unknown",
  "has_chart": true
}
```

| Object | Fields | Notes |
|---|---|---|
| `QCMethodResult` | `id, label, usable, history_mode, validation, notices, jobs` (+ `counts` in the summary) | `notices`: short sentences shown on Steps 2 and 3 (what was analysed, rows skipped and why). |
| `QCJob` | `key, job, instruments, first_date, items` (+ `counts`, `n_items`) | `key` = position as a string (`"0"`, `"1"`, …) in date order. |
| `QCItem` | see the table in `contract.py` | `metrics` values are **finished strings** (units, rounding, wording). The browser prints them as-is. |

The server serialises with `allow_nan=False`, so a NaN anywhere in your output fails loudly. Format or omit missing numbers in the adapter; never pass NaN/None through.

---

## 6. Warning states

Every item has exactly one of three states (`src/qc_status.py`):

| State | Meaning | LCS rule (`WARNING_STATE_BY_DRIFT_STATUS` in `control_detector.py`) |
|---|---|---|
| `FAIL` | breached an acceptance limit; action required | latest result outside the failure limits |
| `WARNING` | within limits, but something needs review | in the warning band, or a consistent trend toward a limit |
| `PASS` | no issue found | within the warning limits and no consistent trend (also: too little history for a trend, explained in the reason) |

- **You decide the rule** for your method, and you define it *next to your analysis logic* (in your detector or a small mapping dict), not in the browser.
- Keep your finer outcome in `status_detail` (LCS: `UPPER_FAILURE_DRIFT`, `LOWER_WARNING`, …) and explain it in `reason`. That way nothing is lost by reducing to three states.
- The colours, ordering (`FAIL` → `WARNING` → `PASS`, then larger `magnitude` first) and counts are shared. Don't add a fourth state.

---

## 7. Selecting results (`ANALYTE_CODE`) and jobs

- The detail page shows **one job at a time** with a job selector, so group your items per job (`QCJob`). If your method has no meaningful job, return one `QCJob` with `job=display_job(None)` ("Job unknown").
- Inside a job, the page shows **one button per `code`** (LCS: `ANALYTE_CODE`), coloured by the worst state of that code's items.
- If one code has several results in a job (LCS: the same analyte under two schemes), give them distinct `variant` labels. The page then shows small tabs, worst first. Leave `variant` as `""` when there's only ever one.
- Switching code, variant or job never re-runs the analysis. The page only fetches that job's items or that item's chart.

---

## 8. Visualisation

- Implement `render_chart(item, output_path)` by calling your visualiser with `item.chart_payload`, i.e. data your detector **already computed** (LCS passes the evaluation dict, which includes its point-in-time `history_window`). Don't re-run detection or re-read files inside the visualiser.
- Draw with matplotlib's object API (`from matplotlib.figure import Figure`, no `pyplot`), as `control_visualize.py` does. The server renders charts on demand in background threads; it serialises the rendering, and the object API keeps it free of global state.
- Show what explains the state: the result, the limits or thresholds it was judged against, and (if your method uses history) the history and trend. Use the shared state colours (`FAIL #d03b3b`, `WARNING #fab219`, `PASS #0ca30c`) only for state, never for an ordinary data series.
- Charts are rendered lazily (first view, then cached per run) and pre-rendered in the background when a job is opened. Aim for about 0.1–0.3 s per chart.
- If an item has no sensible chart, set `has_chart=False`; the page shows the text panel only.

---

## 9. Job and Instrument

- **Never** read `JOB_NAME_ANON`, `JOB_CODE`, `INSTRUMENT_CODE` or `INSTRUMENT_ID` yourself. Use the internal `job_code` and `instrument_id`; the loader has already applied the preferences (section 3).
- **Always** turn them into display text with `display_job()`, `display_instrument()` and `display_instruments()` from `src/qc_report/contract.py`. These return "Job unknown" / "Instrument unknown" for `None`, `NaN`, `NaT`, blanks, `"nan"` and `"None"`.
- Several instruments in one job: `QCJob.instruments` lists them all (`display_instruments()` sorts them and adds "Instrument unknown" last if some rows have none). Each item's own `instrument` is the instrument of the observation that item is about.

---

## 10. History (if your method uses history)

The POC's **History** toggle reaches you as `use_history`:

- `True`: "use the older 50% of the jobs in this file as history, analyse the newer 50%". LCS does this with `LCSDetector.build_history(older_jobs)`, then `detect(newer_jobs, history=seed, persist=True)`. The store is written to a **separate demo path** (`demo_history_path` in `config/lcs_config.yaml`) so the POC never overwrites a real history store.
- `False`: nothing is read from or written to disk. LCS analyses all jobs in date order with an in-memory history (`persist=False`).
- Single-job file with history on: there is no older half. Analyse without history and add a notice saying so.

If your method uses a fixed, pre-built reference model (e.g. Duplicate's reference curves), keep it read-only, ignore the toggle, and say so in a notice.

---

## 11. Not breaking other methods

- One adapter per method, one `register()` line. The server calls each method separately, and `QC_METHODS` ids must be unique (`register()` refuses duplicates).
- Return `usable=False` with a notice instead of raising when your method can't run on a file. An unexpected exception fails the whole upload for **everyone**.
- Your routing must not overlap with another method's (section 4).
- Run the full suite before pushing: `python -m pytest`. The LCS tests (`tests/test_control_detector.py`, `tests/test_qc_report.py`, `tests/test_poc_server.py`, `tests/test_data_validator.py`) must stay green.

---

## 12. Checklist

- [ ] My detector accepts the **internal** column names from `load_qc_data()` (or my adapter renames, see section 3); any new column is in `config/column_config.csv`.
- [ ] `select_<method>_rows(df)` exists in `src/data_validator.py`, is used by my validator, detector and adapter, and has tests (including no overlap with methods that share my `analytical_type`).
- [ ] My detector reports rows it skips (counts + reason), and my adapter turns them into a notice.
- [ ] `src/qc_report/<method>_method.py` implements `id`, `label`, `validate`, `run`, `render_chart`; `id` equals my detector's id in `poc/data.js`.
- [ ] Every item has a `state` of FAIL / WARNING / PASS, decided by a documented rule next to my analysis logic, plus `status_detail` and a plain-language `reason`.
- [ ] Items are grouped per job; `code` is what the user selects by; `variant` is set when one code has several results in a job.
- [ ] Job and Instrument go through `display_job()` / `display_instrument()` / `display_instruments()`.
- [ ] `metrics` values are formatted strings; no NaN/None reaches the JSON.
- [ ] `render_chart` draws one item from `chart_payload` with the matplotlib object API, using state colours only for state.
- [ ] `register(MyMethod())` is added in `src/qc_report/registry.py`; in `poc/data.js` my entry has `serverAnalysis: true` and my placeholder `ITEMS` are removed.
- [ ] Tested end to end: `python poc/server.py`, load a file with and without history, check Steps 2 and 3, switch jobs and codes.
- [ ] `python -m pytest` passes (apart from the known, pre-existing SRMS/Matrix Spike failures).
