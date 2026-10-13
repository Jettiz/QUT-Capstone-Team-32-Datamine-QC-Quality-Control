/**
 * data.js — detector definitions for the browser-side file check, plus the
 * PLACEHOLDER items for detectors that are not yet connected to the server.
 *
 * Real results: any detector whose id is registered in
 * src/qc_report/registry.py (currently only "control" = LCS) is analysed by
 * poc/server.py and its real results replace everything below for that
 * detector (see app.js / summary.js). The ITEMS placeholders only exist for
 * detectors that still need to be integrated -- see
 * docs/QC_INTEGRATION_GUIDE.md.
 *
 * Two kinds of data live here:
 *
 * 1. DETECTORS — one entry per sample type the "Load a sample" page
 *    (index.html) previews a user-chosen file against, before it is sent to
 *    the server. `requiredColumns` mirrors the column lists in
 *    src/data_validator.py, expressed as the RAW uppercase export names a
 *    file actually has (the server maps them to internal names via
 *    config/column_config.csv). "A|B" means either column is accepted
 *    (aliases, e.g. JOB_NAME_ANON|JOB_CODE). `serverAnalysis: true` marks a
 *    detector whose real results come from poc/server.py. `rowType` is the rows.js
 *    canonical type the detector analyses -- note "LCS" vs "SRM": both are
 *    Standard rows, split on STD_LOT_CODE == "Sample" (source of truth:
 *    data_validator.select_lcs_rows / select_srm_rows).
 *
 *    Duplicate/Replicate special case -- rpd/mean_conc are DERIVED, not
 *    sourced: data_loader._derive_precision_metrics() computes both from
 *    NUMERIC_FINAL_VALUE/PARENT_NUMERIC_FINAL_VALUE, so THOSE are what this
 *    page checks for instead.
 *
 *    Matrix Spike Duplicate is deliberately NOT listed (removed from scope).
 *
 * 2. ITEMS — placeholder records for detectors without a server-side
 *    analysis yet, in the SAME shape the server returns for real items
 *    (src/qc_report/contract.py's QCItem): code, variant, state
 *    (FAIL | WARNING | PASS), status_detail, reason, magnitude (tie-break),
 *    metrics [{label, value}], job, instrument. No chart: render.js draws a
 *    generated placeholder instead.
 */
window.LCSPoc = window.LCSPoc || {};

window.LCSPoc.data = (function () {
  "use strict";

  const DETECTORS = [
    {
      id: "blank",
      label: "Blank",
      validatorImplemented: true, // validate_blank_data()
      rowType: "Blank",
      requiredColumns: [
        "ANALYTICAL_TYPE", "STD_LOT_CODE", "STD_CODE", "SCHEME_CODE", "JOB_NAME_ANON|JOB_CODE",
        "ANALYTE_CODE", "ANALYSED_DATE", "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE",
        "INTERNAL_MIN_VALUE", "INTERNAL_MAX_VALUE", "UNIT_CODE",
      ],
    },
    {
      id: "control",
      label: "Control (LCS)",
      validatorImplemented: true, // validate_lcs_data()
      serverAnalysis: true, // registered in src/qc_report/registry.py -- real results come from poc/server.py
      rowType: "LCS",
      requiredColumns: [
        "ANALYTICAL_TYPE", "STD_LOT_CODE", "STD_CODE", "SCHEME_CODE", "JOB_NAME_ANON|JOB_CODE",
        "ANALYTE_CODE", "ANALYSED_DATE", "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE",
        "INTERNAL_MAX_VALUE", "INTERNAL_MIN_VALUE",
        "INTERNAL_MAX_WARNING_VALUE", "INTERNAL_MIN_WARNING_VALUE",
      ],
    },
    {
      id: "duplicate",
      label: "Duplicate",
      validatorImplemented: true, // validate_duplicate_data()
      rowType: "Duplicate",
      // NUMERIC_FINAL_VALUE + PARENT_NUMERIC_FINAL_VALUE, not RPD/MEAN_CONC
      // themselves -- those are derived downstream, see module docstring.
      requiredColumns: [
        "ANALYTE_CODE", "NUMERIC_FINAL_VALUE", "PARENT_NUMERIC_FINAL_VALUE",
        "PRECISION_STATUS", "STAT_DL_DUP_VALUE", "LIM_REP_DUP_VALUE",
      ],
    },
    {
      id: "replicate",
      label: "Replicate",
      validatorImplemented: true, // validate_replicate_data()
      rowType: "Replicate",
      requiredColumns: [
        "ANALYTE_CODE", "NUMERIC_FINAL_VALUE", "PARENT_NUMERIC_FINAL_VALUE",
        "PRECISION_STATUS", "STAT_DL_VALUE", "LIM_REP_VALUE",
      ],
    },
    {
      id: "srm",
      label: "SRM",
      validatorImplemented: true, // validate_srm_data()
      rowType: "SRM",
      requiredColumns: [
        "ANALYTICAL_TYPE", "STD_LOT_CODE", "STD_CODE", "JOB_NAME_ANON|JOB_CODE", "NUMERIC_FINAL_VALUE",
        "ANALYSED_DATE", "SCHEME_CODE", "ANALYTE_CODE", "INTERNAL_MIN_VALUE", "INTERNAL_MAX_VALUE",
        "INTERNAL_MIN_INCLUSIVE", "INTERNAL_MAX_INCLUSIVE", "INTERNAL_MAX_WARNING_VALUE",
        "INTERNAL_MIN_WARNING_VALUE", "INTERNAL_MIN_WARNING_INCLUSIVE", "INTERNAL_MAX_WARNING_INCLUSIVE",
        "INTERNAL_TARGET_VALUE", "UNIT_CODE", "SPECIFICATION_CODE",
      ],
    },
    {
      id: "matrix_spike",
      label: "Matrix Spike",
      validatorImplemented: true, // validate_matrix_spike_data()
      rowType: "Spike",
      requiredColumns: [
        "ANALYTICAL_TYPE", "STD_LOT_CODE", "STD_CODE", "JOB_NAME_ANON|JOB_CODE", "SCHEME_CODE",
        "ANALYTE_CODE", "ANALYSED_DATE", "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE",
        "INTERNAL_MIN_VALUE", "INTERNAL_MAX_VALUE", "INTERNAL_MIN_INCLUSIVE", "INTERNAL_MAX_INCLUSIVE",
        "INTERNAL_MAX_WARNING_VALUE", "INTERNAL_MIN_WARNING_VALUE",
        "INTERNAL_MIN_WARNING_INCLUSIVE", "INTERNAL_MAX_WARNING_INCLUSIVE",
        "UNIT_CODE", "SPECIFICATION_CODE",
      ],
    },
  ];

  function getDetectors() {
    return DETECTORS;
  }

  function getDetector(detectorId) {
    return DETECTORS.find(function (d) { return d.id === detectorId; });
  }

  // Placeholder items (no real analysis behind them yet).
  function placeholder(code, label, state, magnitude, metrics) {
    return {
      id: code, code: code, variant: "", label: label, state: state, status_detail: "PLACEHOLDER",
      reason: metrics[metrics.length - 1].value, magnitude: magnitude,
      metrics: metrics.slice(0, -1), job: "Job unknown", instrument: "Instrument unknown",
      has_chart: false,
    };
  }

  const ITEMS = {
    duplicate: [
      placeholder("Cu", "Copper", "FAIL", 18.5, [
        { label: "RPD", value: "18.5%" },
        { label: "Mean concentration", value: "245.3 mg/kg" },
        { label: "Precision status", value: "Fail" },
        { label: "Reason", value: "RPD exceeds the allowable limit for this concentration." },
      ]),
      placeholder("Ni", "Nickel", "WARNING", 12.0, [
        { label: "RPD", value: "12.0%" },
        { label: "Mean concentration", value: "58.9 mg/kg" },
        { label: "Precision status", value: "Fail" },
        { label: "Reason", value: "RPD is close to the allowable limit for this concentration." },
      ]),
      placeholder("Zn", "Zinc", "WARNING", 7.4, [
        { label: "RPD", value: "7.4%" },
        { label: "Mean concentration", value: "110.2 mg/kg" },
        { label: "Precision status", value: "Pass" },
        { label: "Reason", value: "RPD is elevated but still within the allowable limit." },
      ]),
      placeholder("Pb", "Lead", "PASS", 3.1, [
        { label: "RPD", value: "3.1%" },
        { label: "Mean concentration", value: "89.6 mg/kg" },
        { label: "Precision status", value: "Pass" },
        { label: "Reason", value: "RPD is well within the allowable limit." },
      ]),
    ],

    replicate: [
      placeholder("As", "Arsenic", "WARNING", 15.2, [
        { label: "RPD", value: "15.2%" },
        { label: "Mean concentration", value: "34.7 mg/kg" },
        { label: "Precision status", value: "Fail" },
        { label: "Reason", value: "RPD exceeds the allowable limit for this concentration." },
      ]),
      placeholder("Cd", "Cadmium", "WARNING", 8.9, [
        { label: "RPD", value: "8.9%" },
        { label: "Mean concentration", value: "5.4 mg/kg" },
        { label: "Precision status", value: "Pass" },
        { label: "Reason", value: "RPD is elevated but still within the allowable limit." },
      ]),
      placeholder("Cr", "Chromium", "PASS", 2.0, [
        { label: "RPD", value: "2.0%" },
        { label: "Mean concentration", value: "67.1 mg/kg" },
        { label: "Precision status", value: "Pass" },
        { label: "Reason", value: "RPD is well within the allowable limit." },
      ]),
    ],

    srm: [
      placeholder("Au", "Gold", "FAIL", 1.8, [
        { label: "Risk level", value: "Critical" },
        { label: "Result", value: "1.8 g/t outside acceptance range" },
        { label: "Drift flagged", value: "Yes" },
        { label: "Reason", value: "Result and historical drift both breach the reference material's limits." },
      ]),
      placeholder("Ag", "Silver", "WARNING", 0.6, [
        { label: "Risk level", value: "Medium" },
        { label: "Result", value: "Within warning band" },
        { label: "Drift flagged", value: "No" },
        { label: "Reason", value: "Result sits inside the warning band but has not failed." },
      ]),
      placeholder("Pt", "Platinum", "PASS", 0.05, [
        { label: "Risk level", value: "Low" },
        { label: "Result", value: "On target" },
        { label: "Drift flagged", value: "No" },
        { label: "Reason", value: "Result is close to target with no flagged risk." },
      ]),
    ],

    blank: [
      placeholder("Pb", "Lead", "WARNING", 3.2, [
        { label: "Detected value", value: "3.2x LOD" },
        { label: "Limit of detection (LOD)", value: "0.5 mg/kg" },
        { label: "Status", value: "Fail" },
        { label: "Reason", value: "Blank shows carryover well above the detection limit." },
      ]),
      placeholder("Cd", "Cadmium", "WARNING", 1.4, [
        { label: "Detected value", value: "1.4x LOD" },
        { label: "Limit of detection (LOD)", value: "0.1 mg/kg" },
        { label: "Status", value: "Warning" },
        { label: "Reason", value: "Blank shows a small amount of carryover just above the detection limit." },
      ]),
      placeholder("As", "Arsenic", "PASS", 0.0, [
        { label: "Detected value", value: "Not detected" },
        { label: "Limit of detection (LOD)", value: "0.2 mg/kg" },
        { label: "Status", value: "Pass" },
        { label: "Reason", value: "No carryover detected." },
      ]),
    ],

    matrix_spike: [
      placeholder("Cu", "Copper", "FAIL", 45, [
        { label: "Spike recovery", value: "45%" },
        { label: "Acceptance range", value: "75% - 125%" },
        { label: "Status", value: "Fail" },
        { label: "Reason", value: "Recovery is far below the acceptable range -- possible matrix interference." },
      ]),
      placeholder("Zn", "Zinc", "WARNING", 128, [
        { label: "Spike recovery", value: "128%" },
        { label: "Acceptance range", value: "75% - 125%" },
        { label: "Status", value: "Warning" },
        { label: "Reason", value: "Recovery is just above the acceptable range." },
      ]),
      placeholder("Ni", "Nickel", "PASS", 102, [
        { label: "Spike recovery", value: "102%" },
        { label: "Acceptance range", value: "75% - 125%" },
        { label: "Status", value: "Pass" },
        { label: "Reason", value: "Recovery is well within the acceptable range." },
      ]),
    ],
  };

  /**
   * Placeholder items for a detector that has no server-side analysis yet
   * ([] for detectors that do -- their items come from the server).
   * @param {string} detectorId
   * @returns {Array<Object>}
   */
  function getPlaceholderItems(detectorId) {
    return ITEMS[detectorId] || [];
  }

  return {
    getDetectors: getDetectors,
    getDetector: getDetector,
    getPlaceholderItems: getPlaceholderItems,
  };
})();
