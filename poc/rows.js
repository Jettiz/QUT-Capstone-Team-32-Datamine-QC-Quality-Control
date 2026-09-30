/**
 * rows.js — pure row-classification and bounded-sampling logic for the real,
 * user-uploaded file (as opposed to ranking.js/data.js's dummy per-analyte
 * mock items). No DOM access, no dummy data — mirrors ranking.js's style.
 *
 * Real CCLAS data has two "status" columns that are perfectly complementary
 * by sample type and never both populated on the same row (confirmed against
 * data/raw/ResultSet.csv, 99,999 rows, and the SPK(MS) Assessment sheet of
 * data/raw/QC_Anomaly_Training_Data_v2.xlsx, 6,958 rows):
 *   STANDARD_STATUS  -> populates Blank / Standard / Spike rows
 *   PRECISION_STATUS -> populates Replicate / Duplicate rows
 * Both map onto the same small vocabulary (confirmed exhaustively against the
 * real data -- these are ALL the distinct non-null values that exist):
 *   Pass                                    -> Pass
 *   UpperWarning / LowerWarning / Warning    -> Medium
 *   UpperFailure / LowerFailure / Failure    -> Critical
 *   Ignored{Upper,Lower}Failure / IgnoreFailure -> Critical (a chemist
 *     reviewed and dispositioned it, but it's still a real limit breach)
 * ~0.035% of real rows have neither column populated -- these (and any
 * unrecognised status string) get a 5th tag, UNKNOWN, ranked worst-last.
 *
 * NOTE: with this mapping, "HIGH" is never actually produced by real data
 * (only CRITICAL/MEDIUM/PASS are reachable, plus UNKNOWN for the gap). It's
 * kept in the tag vocabulary anyway for consistency with the rest of the
 * app (ranking.js's SEVERITY_RANK also has CRITICAL/HIGH/MEDIUM), but a
 * "High" filter will currently always show "no rows match" against real data.
 *
 * Rows are bucketed three-deep: type -> tag -> analyte code. This is what
 * lets "top 15" stay correct no matter which of the three dimensions a user
 * fixes (e.g. type=Duplicate + analyte=Cu with tag=ALL still needs to merge
 * across tags correctly) -- see getTopRows()'s docstring for the exact
 * guarantee. Analyte sub-buckets are created lazily (unlike type/tag, the
 * set of analyte codes isn't known up front), and a row with no analyte
 * code goes in a reserved UNSPECIFIED_ANALYTE bucket that's counted but
 * never offered as a dropdown option.
 */
window.LCSPoc = window.LCSPoc || {};

window.LCSPoc.rows = (function () {
  "use strict";

  // Single source of truth for these raw column names -- avoids the
  // "ANALYTICAL_TYPE" string being duplicated/drifting across load.js.
  const COLUMN_NAMES = {
    analyticalType: "ANALYTICAL_TYPE",
    standardStatus: "STANDARD_STATUS",
    precisionStatus: "PRECISION_STATUS",
    analyteCode: "ANALYTE_CODE",
  };

  // Worst-first order -- this exact array drives both table sort order and
  // dropdown option order everywhere.
  const TAGS = ["CRITICAL", "HIGH", "MEDIUM", "PASS", "UNKNOWN"];
  const TAG_LABELS = { CRITICAL: "Critical", HIGH: "High", MEDIUM: "Medium", PASS: "Pass", UNKNOWN: "Unknown" };

  // Matches data.js's DETECTORS[*].analyticalTypeFilter values exactly.
  const KNOWN_TYPES = ["Blank", "Standard", "Replicate", "Duplicate", "Spike"];
  const TYPE_LABELS = { Blank: "Blank", Standard: "Standard (Control/SRM)", Replicate: "Replicate", Duplicate: "Duplicate", Spike: "Matrix Spike" };
  // OTHER catches rows whose ANALYTICAL_TYPE is missing/blank/unrecognised,
  // including every row when there's no ANALYTICAL_TYPE column at all. Never
  // offered as a dropdown option; only swept in when the type filter is ALL.
  const ALL_TYPES = KNOWN_TYPES.concat(["OTHER"]);

  // Reserved analyte-bucket key for a row with no (or blank) ANALYTE_CODE.
  // Its rows are still counted/sampled (never silently dropped), but this
  // key is never surfaced as a selectable dropdown option.
  const UNSPECIFIED_ANALYTE = "__UNSPECIFIED__";

  // A bucket capped at ROWS_PER_BUCKET_CAP either holds every real row
  // (when the true count is <= cap) or is only ever asked for <= cap rows
  // anyway (since TABLE_DISPLAY_LIMIT never asks for more) -- so these two
  // must stay equal for the "top 15 is always correct" guarantee to hold.
  const ROWS_PER_BUCKET_CAP = 15;
  const TABLE_DISPLAY_LIMIT = 15;

  const STATUS_VALUE_TAG_MAP = {
    PASS: "PASS",
    UPPERWARNING: "MEDIUM",
    LOWERWARNING: "MEDIUM",
    WARNING: "MEDIUM",
    UPPERFAILURE: "CRITICAL",
    LOWERFAILURE: "CRITICAL",
    FAILURE: "CRITICAL", // PRECISION_STATUS's plain "Failure" (no Upper/Lower split, unlike STANDARD_STATUS)
    IGNOREDUPPERFAILURE: "CRITICAL",
    IGNOREDLOWERFAILURE: "CRITICAL",
    IGNOREFAILURE: "CRITICAL",
  };

  /**
   * Maps a raw ANALYTICAL_TYPE string to one of KNOWN_TYPES, or "OTHER" if
   * missing/blank/unrecognised.
   * @param {string} rawAnalyticalType
   * @returns {string}
   */
  function classifyRowType(rawAnalyticalType) {
    const normalised = String(rawAnalyticalType || "").trim().toUpperCase();
    const match = KNOWN_TYPES.find(function (t) { return t.toUpperCase() === normalised; });
    return match || "OTHER";
  }

  /**
   * Normalises a raw ANALYTE_CODE value to a bucket key: the trimmed value
   * as-is (case preserved -- real analyte codes are already consistently
   * cased in CCLAS data), or the UNSPECIFIED_ANALYTE sentinel if blank.
   * @param {string} rawAnalyteCode
   * @returns {string}
   */
  function classifyAnalyteCode(rawAnalyteCode) {
    const trimmed = String(rawAnalyteCode || "").trim();
    return trimmed || UNSPECIFIED_ANALYTE;
  }

  /**
   * Picks which raw status value is relevant for a given canonical type.
   * Blank/Standard/Spike -> STANDARD_STATUS; Replicate/Duplicate ->
   * PRECISION_STATUS; OTHER (undefined by the business rule) defensively
   * falls back to whichever of the two is actually non-empty.
   * @param {string} canonicalType
   * @param {string} standardStatusRaw
   * @param {string} precisionStatusRaw
   * @returns {string}
   */
  function pickStatusColumnRaw(canonicalType, standardStatusRaw, precisionStatusRaw) {
    if (canonicalType === "Replicate" || canonicalType === "Duplicate") {
      return precisionStatusRaw || "";
    }
    if (canonicalType === "Blank" || canonicalType === "Standard" || canonicalType === "Spike") {
      return standardStatusRaw || "";
    }
    return standardStatusRaw || precisionStatusRaw || "";
  }

  /**
   * Classifies one row's Detected-issue tag from its type and the two raw
   * status column values (whichever one is relevant is picked internally).
   * @param {string} canonicalType
   * @param {string} standardStatusRaw
   * @param {string} precisionStatusRaw
   * @returns {string} one of TAGS
   */
  function classifyRowTag(canonicalType, standardStatusRaw, precisionStatusRaw) {
    const raw = pickStatusColumnRaw(canonicalType, standardStatusRaw, precisionStatusRaw);
    const trimmed = String(raw || "").trim();
    if (!trimmed) {
      return "UNKNOWN";
    }
    const mapped = STATUS_VALUE_TAG_MAP[trimmed.toUpperCase()];
    return mapped || "UNKNOWN";
  }

  /**
   * Factory for the bounded type -> tag -> analyte row sampler used while
   * scanning a file's data lines. Keeps up to `bucketCap` raw rows per
   * (type, tag, analyte) combination while still counting every row that
   * belongs to it. Also tracks, per type, the set of distinct real analyte
   * codes seen (cheap -- just strings, independent of `bucketCap`) so
   * dropdowns can be populated without scanning the samples themselves.
   * @param {number} bucketCap
   */
  function createRowSampler(bucketCap) {
    const buckets = {}; // buckets[type][tag][analyteKey] = {totalCount, rows}
    const analyteCodesByType = {};
    ALL_TYPES.forEach(function (type) {
      buckets[type] = {};
      TAGS.forEach(function (tag) {
        buckets[type][tag] = {}; // analyte sub-buckets created lazily below
      });
      analyteCodesByType[type] = {}; // used as a string Set (object keys)
    });

    let rowsWithNeitherStatusColumnPopulated = 0;
    let rowsWithUnrecognizedStatusValue = 0;

    return {
      /**
       * @param {string} canonicalType one of ALL_TYPES
       * @param {string} tag one of TAGS
       * @param {string} analyteCodeRaw raw ANALYTE_CODE value for this row
       *   (may be blank/undefined)
       * @param {Array<string>} fields raw field values for this row,
       *   positionally aligned to the file's header
       * @param {string} statusRawForDiagnostics the raw status value that
       *   was actually used to classify this row (for diagnostics only)
       */
      addRow: function (canonicalType, tag, analyteCodeRaw, fields, statusRawForDiagnostics) {
        const analyteKey = classifyAnalyteCode(analyteCodeRaw);
        const tagBucket = buckets[canonicalType][tag];
        if (!tagBucket[analyteKey]) {
          tagBucket[analyteKey] = { totalCount: 0, rows: [] };
        }
        const bucket = tagBucket[analyteKey];
        bucket.totalCount += 1;
        if (bucket.rows.length < bucketCap) {
          bucket.rows.push(fields);
        }

        if (analyteKey !== UNSPECIFIED_ANALYTE) {
          analyteCodesByType[canonicalType][analyteKey] = true;
        }

        if (tag === "UNKNOWN") {
          if (!statusRawForDiagnostics) {
            rowsWithNeitherStatusColumnPopulated += 1;
          } else {
            rowsWithUnrecognizedStatusValue += 1;
          }
        }
      },
      build: function () {
        const analyteCodesByTypeSorted = {};
        const allAnalyteCodesSet = {};
        ALL_TYPES.forEach(function (type) {
          const codes = Object.keys(analyteCodesByType[type]).sort();
          analyteCodesByTypeSorted[type] = codes;
          codes.forEach(function (code) { allAnalyteCodesSet[code] = true; });
        });

        return {
          bucketCap: bucketCap,
          tags: TAGS.slice(),
          types: ALL_TYPES.slice(),
          buckets: buckets,
          analyteCodesByType: analyteCodesByTypeSorted,
          analyteCodesAll: Object.keys(allAnalyteCodesSet).sort(),
          diagnostics: {
            rowsWithNeitherStatusColumnPopulated: rowsWithNeitherStatusColumnPopulated,
            rowsWithUnrecognizedStatusValue: rowsWithUnrecognizedStatusValue,
          },
        };
      },
    };
  }

  /**
   * The one shared query summary.js and the detail page's controller both
   * call: returns up to `limit` real sampled rows for a (type, tag, analyte)
   * filter combination (any of the three may be "ALL"), worst-tag-first,
   * plus the true total count of rows matching that filter (which may
   * exceed the sample size).
   *
   * Correctness guarantee: since buckets are kept at the full (type, tag,
   * analyte) granularity, every one of the 6x6x(analytes+1) selectable
   * filter combinations reduces to summing/concatenating a set of these
   * leaf buckets -- each of which independently holds every real row when
   * its true count is <= bucketCap, and is never asked for more than
   * TABLE_DISPLAY_LIMIT (<= bucketCap) rows regardless of how many leaf
   * buckets are merged under an "ALL" filter. So the rows returned are
   * always genuinely among the worst-first real rows for that filter,
   * never fabricated, never wrongly short of what actually exists (up to
   * the display limit).
   * @param {Object} rowSample the object built by createRowSampler().build()
   * @param {string} typeFilter "ALL" or one of KNOWN_TYPES
   * @param {string} tagFilter "ALL" or one of TAGS
   * @param {string} analyteFilter "ALL" or a real analyte code
   * @param {number} limit
   * @returns {{rows: Array<{type: string, tag: string, fields: Array<string>}>, totalMatching: number}}
   */
  function getTopRows(rowSample, typeFilter, tagFilter, analyteFilter, limit) {
    const tagsToConsider = tagFilter === "ALL" ? rowSample.tags : [tagFilter];
    // "ALL" sweeps every bucket type INCLUDING "OTHER" so nothing silently
    // disappears; a named type filter never includes OTHER.
    const typesToConsider = typeFilter === "ALL" ? rowSample.types : [typeFilter];

    const result = [];
    let totalMatching = 0;

    tagsToConsider.forEach(function (tag) {
      typesToConsider.forEach(function (type) {
        const tagBucket = rowSample.buckets[type] && rowSample.buckets[type][tag];
        if (!tagBucket) {
          return;
        }
        // "ALL" sweeps every analyte sub-bucket, including UNSPECIFIED_ANALYTE
        // (so rows with no analyte code are still counted/shown under "All").
        const analyteKeysToConsider = analyteFilter === "ALL" ? Object.keys(tagBucket) : [analyteFilter];
        analyteKeysToConsider.forEach(function (analyteKey) {
          const bucket = tagBucket[analyteKey];
          if (!bucket) {
            return;
          }
          totalMatching += bucket.totalCount;
          bucket.rows.forEach(function (fields) {
            if (result.length < limit) {
              result.push({ type: type, tag: tag, fields: fields });
            }
          });
        });
      });
    });

    return { rows: result, totalMatching: totalMatching };
  }

  return {
    COLUMN_NAMES: COLUMN_NAMES,
    TAGS: TAGS,
    TAG_LABELS: TAG_LABELS,
    KNOWN_TYPES: KNOWN_TYPES,
    TYPE_LABELS: TYPE_LABELS,
    ALL_TYPES: ALL_TYPES,
    UNSPECIFIED_ANALYTE: UNSPECIFIED_ANALYTE,
    ROWS_PER_BUCKET_CAP: ROWS_PER_BUCKET_CAP,
    TABLE_DISPLAY_LIMIT: TABLE_DISPLAY_LIMIT,
    STATUS_VALUE_TAG_MAP: STATUS_VALUE_TAG_MAP,
    classifyRowType: classifyRowType,
    classifyAnalyteCode: classifyAnalyteCode,
    pickStatusColumnRaw: pickStatusColumnRaw,
    classifyRowTag: classifyRowTag,
    createRowSampler: createRowSampler,
    getTopRows: getTopRows,
  };
})();
