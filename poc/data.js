/**
 * data.js — the ONLY file in this POC that contains dummy data.
 *
 * PHASE 2 INTEGRATION POINT
 * -------------------------
 * When real detector output is available, this is the file that changes:
 * getDetectors() would map from the real detector/validator registry, and
 * getItems(detectorId) would map from that detector's real anomaly results
 * (e.g. LCSAnomalyResult.details["results"] for "control", generated via
 * src/detectors/control_detector.py + src/visualisers/control_visualize.py).
 * ranking.js, render.js, app.js, load.js, summary.js and every .html file
 * stay untouched as long as the new data matches the shapes below.
 *
 * Two kinds of data live here:
 *
 * 1. DETECTORS — one entry per sample type this POC's "Load a sample" page
 *    (index.html) checks a real, user-chosen CSV against. `requiredColumns`
 *    mirrors the real column lists in src/data_validator.py
 *    (_LCS_REQUIRED_COLUMNS, _BLANK_REQUIRED_COLUMNS, _DUP_REQUIRED_COLUMNS,
 *    _REP_REQUIRED_COLUMNS, _SRM_REQUIRED_COLUMNS, _MS_REQUIRED_COLUMNS),
 *    ALL of which are now real, implemented validators -- expressed here as
 *    the raw UPPERCASE CCLAS column names a user's CSV would actually have
 *    (this page checks a file before it's ever run through
 *    data_loader.load_qc_data()'s lowercase renaming, so it has to look for
 *    the raw names, same as validate_lcs_data()/validate_srm_data() do).
 *    `analyticalTypeFilter` mirrors each validator's own
 *    ANALYTICAL_TYPE == "..." row filter (validate_lcs_data() and
 *    validate_srm_data() both filter to "Standard";
 *    validate_matrix_spike_data() to "Spike"; validate_blank_data() to
 *    "blank", case-insensitive). validate_duplicate_data()/
 *    validate_replicate_data() don't filter internally (they expect
 *    pre-filtered input), but this page still uses "Duplicate"/"Replicate"
 *    here to judge whether the loaded file HAS any such rows at all.
 *
 *    Duplicate/Replicate special case -- rpd/mean_conc are DERIVED, not
 *    sourced: neither exists as a literal column in any raw CCLAS export.
 *    data_loader._derive_precision_metrics() computes both from
 *    NUMERIC_FINAL_VALUE/PARENT_NUMERIC_FINAL_VALUE, so THOSE are what this
 *    page checks for instead of RPD/MEAN_CONC themselves -- checking for
 *    the literal derived names would report every real file as unusable.
 *
 *    Matrix Spike Duplicate is deliberately NOT listed: its detector,
 *    visualiser, config, and validator stub were all removed from the
 *    project (see src/detectors/base_detector.py, src/data_validator.py,
 *    src/data_loader.py) — Matrix Spike itself stays.
 *
 * 2. ITEMS — per detector, a small set of dummy "anomaly" records driving
 *    the detail page (detail.html). ONE shared shape across every detector
 *    type, so ranking.js/render.js never branch on detector type:
 *      code      - short identifier (e.g. analyte code)
 *      label     - display name
 *      severity  - "CRITICAL" | "HIGH" | "MEDIUM" | "NONE"
 *      magnitude - a signed number, same-severity tie-breaker only
 *                  (mirrors control_detector.py's abs(OFFSET) tie-break)
 *      metrics   - ordered [{label, value}] shown in the info panel
 *      imagePath - relative path to a real dummy image, or null to fall
 *                  back to render.js's generated placeholder chart
 *                  (only "control" has real hand-drawn SVGs so far)
 *      imageAlt  - alt text (ignored when imagePath is null)
 */
window.LCSPoc = window.LCSPoc || {};

window.LCSPoc.data = (function () {
  "use strict";

  const DETECTORS = [
    {
      id: "blank",
      label: "Blank",
      validatorImplemented: true, // validate_blank_data()
      analyticalTypeFilter: "Blank",
      requiredColumns: [
        "ANALYTICAL_TYPE", "STD_LOT_CODE", "STD_CODE", "SCHEME_CODE", "JOB_CODE",
        "ANALYTE_CODE", "ANALYSED_DATE", "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE",
        "INTERNAL_MIN_VALUE", "INTERNAL_MAX_VALUE", "UNIT_CODE",
      ],
    },
    {
      id: "control",
      label: "Control (LCS)",
      validatorImplemented: true, // validate_lcs_data()
      analyticalTypeFilter: "Standard",
      requiredColumns: [
        "ANALYTICAL_TYPE", "STD_LOT_CODE", "STD_CODE", "SCHEME_CODE", "JOB_CODE",
        "ANALYTE_CODE", "ANALYSED_DATE", "NUMERIC_FINAL_VALUE",
        "INTERNAL_TARGET_VALUE", "INTERNAL_MAX_WARNING_VALUE", "INTERNAL_MIN_WARNING_VALUE",
      ],
    },
    {
      id: "duplicate",
      label: "Duplicate",
      validatorImplemented: true, // validate_duplicate_data()
      analyticalTypeFilter: "Duplicate",
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
      analyticalTypeFilter: "Replicate",
      requiredColumns: [
        "ANALYTE_CODE", "NUMERIC_FINAL_VALUE", "PARENT_NUMERIC_FINAL_VALUE",
        "PRECISION_STATUS", "STAT_DL_VALUE", "LIM_REP_VALUE",
      ],
    },
    {
      id: "srm",
      label: "SRM",
      validatorImplemented: true, // validate_srm_data()
      analyticalTypeFilter: "Standard",
      requiredColumns: [
        "ANALYTICAL_TYPE", "STD_LOT_CODE", "STD_CODE", "JOB_CODE", "NUMERIC_FINAL_VALUE",
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
      analyticalTypeFilter: "Spike",
      requiredColumns: [
        "ANALYTICAL_TYPE", "STD_LOT_CODE", "STD_CODE", "JOB_CODE", "SCHEME_CODE",
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

  // Control's 5 items are backed by REAL charts + REAL LCSDetector output
  // (unlike every other detector's items below, still dummy) -- generated
  // by notebooks/LCS_new_sample_test_harness.ipynb's "Generate real
  // Control-analyte demo charts" section, which seeds 5 ANALYTE_CODEs
  // (Cu/Zn/Pb/Ni/Fe) with a crafted baseline + current sample and runs
  // them through the real detector + src/visualisers/control_visualize.py.
  // OFFSET/CONFIDENCE/Reason below are copied verbatim from that run's
  // printed LCSDetector output, not invented.
  const ITEMS = {
    control: [
      {
        code: "Cu", label: "Copper", severity: "CRITICAL", magnitude: 1.38,
        metrics: [
          { label: "Offset", value: "+1.38" },
          { label: "Confidence", value: "100%" },
          { label: "Error type", value: "Upper drift" },
          { label: "Reason", value: "Latest result has already breached the upper failure limit." },
        ],
        imagePath: "images/control_CU_OREAS_502C_GE_ICP40Q12.png",
        imageAlt: "Copper control sample drift chart showing an upper failure breach",
      },
      {
        code: "Zn", label: "Zinc", severity: "HIGH", magnitude: 0.5506,
        metrics: [
          { label: "Offset", value: "+0.5506" },
          { label: "Confidence", value: "85%" },
          { label: "Error type", value: "Upper drift" },
          { label: "Reason", value: "Latest result is within limits, but the last 10 results show a consistent upward trend (slope +0.0523/result, strength 1.00) toward the upper failure boundary." },
        ],
        imagePath: "images/control_ZN_OREAS_502C_GE_ICP40Q12.png",
        imageAlt: "Zinc control sample drift chart showing a consistent upward trend",
      },
      {
        code: "Pb", label: "Lead", severity: "HIGH", magnitude: -0.5506,
        metrics: [
          { label: "Offset", value: "-0.5506" },
          { label: "Confidence", value: "85%" },
          { label: "Error type", value: "Lower drift" },
          { label: "Reason", value: "Latest result is within limits, but the last 10 results show a consistent downward trend (slope -0.0523/result, strength 1.00) toward the lower failure boundary." },
        ],
        imagePath: "images/control_PB_OREAS_502C_GE_ICP40Q12.png",
        imageAlt: "Lead control sample drift chart showing a consistent downward trend",
      },
      {
        code: "Ni", label: "Nickel", severity: "MEDIUM", magnitude: 0.7265,
        metrics: [
          { label: "Offset", value: "+0.7265" },
          { label: "Confidence", value: "65%" },
          { label: "Error type", value: "Upper drift" },
          { label: "Reason", value: "Latest result is currently in the upper warning band." },
        ],
        imagePath: "images/control_NI_OREAS_502C_GE_ICP40Q12.png",
        imageAlt: "Nickel control sample drift chart showing an upper warning band result",
      },
      {
        code: "Fe", label: "Iron", severity: "NONE", magnitude: 0.0191,
        metrics: [
          { label: "Offset", value: "+0.0191" },
          { label: "Confidence", value: "95%" },
          { label: "Error type", value: "No drift" },
          { label: "Reason", value: "Latest result is within limits and no consistent trend toward a boundary is evident." },
        ],
        imagePath: "images/control_FE_OREAS_502C_GE_ICP40Q12.png",
        imageAlt: "Iron control sample drift chart showing a stable, healthy result",
      },
    ],

    duplicate: [
      {
        code: "Cu", label: "Copper", severity: "CRITICAL", magnitude: 18.5,
        metrics: [
          { label: "RPD", value: "18.5%" },
          { label: "Mean concentration", value: "245.3 mg/kg" },
          { label: "Precision status", value: "Fail" },
          { label: "Reason", value: "RPD exceeds the allowable limit for this concentration." },
        ],
        imagePath: null, imageAlt: "Copper duplicate-pair RPD placeholder",
      },
      {
        code: "Ni", label: "Nickel", severity: "HIGH", magnitude: 12.0,
        metrics: [
          { label: "RPD", value: "12.0%" },
          { label: "Mean concentration", value: "58.9 mg/kg" },
          { label: "Precision status", value: "Fail" },
          { label: "Reason", value: "RPD is close to the allowable limit for this concentration." },
        ],
        imagePath: null, imageAlt: "Nickel duplicate-pair RPD placeholder",
      },
      {
        code: "Zn", label: "Zinc", severity: "MEDIUM", magnitude: 7.4,
        metrics: [
          { label: "RPD", value: "7.4%" },
          { label: "Mean concentration", value: "110.2 mg/kg" },
          { label: "Precision status", value: "Pass" },
          { label: "Reason", value: "RPD is elevated but still within the allowable limit." },
        ],
        imagePath: null, imageAlt: "Zinc duplicate-pair RPD placeholder",
      },
      {
        code: "Pb", label: "Lead", severity: "NONE", magnitude: 3.1,
        metrics: [
          { label: "RPD", value: "3.1%" },
          { label: "Mean concentration", value: "89.6 mg/kg" },
          { label: "Precision status", value: "Pass" },
          { label: "Reason", value: "RPD is well within the allowable limit." },
        ],
        imagePath: null, imageAlt: "Lead duplicate-pair RPD placeholder",
      },
    ],

    replicate: [
      {
        code: "As", label: "Arsenic", severity: "HIGH", magnitude: 15.2,
        metrics: [
          { label: "RPD", value: "15.2%" },
          { label: "Mean concentration", value: "34.7 mg/kg" },
          { label: "Precision status", value: "Fail" },
          { label: "Reason", value: "RPD exceeds the allowable limit for this concentration." },
        ],
        imagePath: null, imageAlt: "Arsenic replicate-pair RPD placeholder",
      },
      {
        code: "Cd", label: "Cadmium", severity: "MEDIUM", magnitude: 8.9,
        metrics: [
          { label: "RPD", value: "8.9%" },
          { label: "Mean concentration", value: "5.4 mg/kg" },
          { label: "Precision status", value: "Pass" },
          { label: "Reason", value: "RPD is elevated but still within the allowable limit." },
        ],
        imagePath: null, imageAlt: "Cadmium replicate-pair RPD placeholder",
      },
      {
        code: "Cr", label: "Chromium", severity: "NONE", magnitude: 2.0,
        metrics: [
          { label: "RPD", value: "2.0%" },
          { label: "Mean concentration", value: "67.1 mg/kg" },
          { label: "Precision status", value: "Pass" },
          { label: "Reason", value: "RPD is well within the allowable limit." },
        ],
        imagePath: null, imageAlt: "Chromium replicate-pair RPD placeholder",
      },
    ],

    srm: [
      {
        code: "Au", label: "Gold", severity: "CRITICAL", magnitude: 1.8,
        metrics: [
          { label: "Risk level", value: "Critical" },
          { label: "Result", value: "1.8 g/t outside acceptance range" },
          { label: "Drift flagged", value: "Yes" },
          { label: "Reason", value: "Result and historical drift both breach the reference material's limits." },
        ],
        imagePath: null, imageAlt: "Gold SRM placeholder chart",
      },
      {
        code: "Ag", label: "Silver", severity: "MEDIUM", magnitude: 0.6,
        metrics: [
          { label: "Risk level", value: "Medium" },
          { label: "Result", value: "Within warning band" },
          { label: "Drift flagged", value: "No" },
          { label: "Reason", value: "Result sits inside the warning band but has not failed." },
        ],
        imagePath: null, imageAlt: "Silver SRM placeholder chart",
      },
      {
        code: "Pt", label: "Platinum", severity: "NONE", magnitude: 0.05,
        metrics: [
          { label: "Risk level", value: "Low" },
          { label: "Result", value: "On target" },
          { label: "Drift flagged", value: "No" },
          { label: "Reason", value: "Result is close to target with no flagged risk." },
        ],
        imagePath: null, imageAlt: "Platinum SRM placeholder chart",
      },
    ],

    blank: [
      {
        code: "Pb", label: "Lead", severity: "HIGH", magnitude: 3.2,
        metrics: [
          { label: "Detected value", value: "3.2x LOD" },
          { label: "Limit of detection (LOD)", value: "0.5 mg/kg" },
          { label: "Status", value: "Fail" },
          { label: "Reason", value: "Blank shows carryover well above the detection limit." },
        ],
        imagePath: null, imageAlt: "Lead blank placeholder chart",
      },
      {
        code: "Cd", label: "Cadmium", severity: "MEDIUM", magnitude: 1.4,
        metrics: [
          { label: "Detected value", value: "1.4x LOD" },
          { label: "Limit of detection (LOD)", value: "0.1 mg/kg" },
          { label: "Status", value: "Warning" },
          { label: "Reason", value: "Blank shows a small amount of carryover just above the detection limit." },
        ],
        imagePath: null, imageAlt: "Cadmium blank placeholder chart",
      },
      {
        code: "As", label: "Arsenic", severity: "NONE", magnitude: 0.0,
        metrics: [
          { label: "Detected value", value: "Not detected" },
          { label: "Limit of detection (LOD)", value: "0.2 mg/kg" },
          { label: "Status", value: "Pass" },
          { label: "Reason", value: "No carryover detected." },
        ],
        imagePath: null, imageAlt: "Arsenic blank placeholder chart",
      },
    ],

    matrix_spike: [
      {
        code: "Cu", label: "Copper", severity: "CRITICAL", magnitude: 45,
        metrics: [
          { label: "Spike recovery", value: "45%" },
          { label: "Acceptance range", value: "75% - 125%" },
          { label: "Status", value: "Fail" },
          { label: "Reason", value: "Recovery is far below the acceptable range -- possible matrix interference." },
        ],
        imagePath: null, imageAlt: "Copper matrix spike recovery placeholder",
      },
      {
        code: "Zn", label: "Zinc", severity: "MEDIUM", magnitude: 128,
        metrics: [
          { label: "Spike recovery", value: "128%" },
          { label: "Acceptance range", value: "75% - 125%" },
          { label: "Status", value: "Warning" },
          { label: "Reason", value: "Recovery is just above the acceptable range." },
        ],
        imagePath: null, imageAlt: "Zinc matrix spike recovery placeholder",
      },
      {
        code: "Ni", label: "Nickel", severity: "NONE", magnitude: 102,
        metrics: [
          { label: "Spike recovery", value: "102%" },
          { label: "Acceptance range", value: "75% - 125%" },
          { label: "Status", value: "Pass" },
          { label: "Reason", value: "Recovery is well within the acceptable range." },
        ],
        imagePath: null, imageAlt: "Nickel matrix spike recovery placeholder",
      },
    ],
  };

  /**
   * Returns the current set of dummy anomaly items for one detector.
   * All consumers should call this function rather than touching ITEMS
   * directly -- that indirection is what lets Phase 2 swap in real,
   * computed results without changing any call site.
   * @param {string} detectorId
   * @returns {Array<Object>}
   */
  function getItems(detectorId) {
    return ITEMS[detectorId] || [];
  }

  return {
    getDetectors: getDetectors,
    getDetector: getDetector,
    getItems: getItems,
  };
})();
