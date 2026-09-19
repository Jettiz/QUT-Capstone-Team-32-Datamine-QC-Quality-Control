/**
 * ranking.js — pure severity-ranking logic. No DOM access, no dummy data.
 *
 * Mirrors the real LCSDetector's severity ordering (_SEVERITY_RANK in
 * src/detectors/control_detector.py: CRITICAL=0, HIGH=1, MEDIUM=2, NONE=3,
 * lower = worse) with the same tie-break the detector's own detect_drift()
 * wrapper uses (severity_score = abs(OFFSET)).
 *
 * These functions operate on plain arrays/objects shaped like the records
 * from data.js (every detector's items share the same `severity`/
 * `magnitude` fields) -- they don't know or care which detector or
 * whether the data is dummy or real, which is what keeps this file
 * reusable across every detector type and across Phase 1 and Phase 2.
 */
window.LCSPoc = window.LCSPoc || {};

window.LCSPoc.ranking = (function () {
  "use strict";

  /** Lower rank number = more severe. Matches DRIFT_SEVERITY strings exactly. */
  const SEVERITY_RANK = { CRITICAL: 0, HIGH: 1, MEDIUM: 2, NONE: 3 };

  /**
   * Compares two items for sorting worst-first.
   * Primary key: severity rank (lower number wins).
   * Secondary key (tie-break): larger absolute magnitude wins.
   * @param {Object} a
   * @param {Object} b
   * @returns {number} negative if a is worse (sorts first), positive if b is worse
   */
  function compareByPriority(a, b) {
    const rankA = SEVERITY_RANK.hasOwnProperty(a.severity) ? SEVERITY_RANK[a.severity] : 9;
    const rankB = SEVERITY_RANK.hasOwnProperty(b.severity) ? SEVERITY_RANK[b.severity] : 9;
    if (rankA !== rankB) {
      return rankA - rankB;
    }
    return Math.abs(b.magnitude) - Math.abs(a.magnitude);
  }

  /**
   * Returns a new array of items sorted worst (highest priority) first.
   * @param {Array<Object>} items
   * @returns {Array<Object>}
   */
  function sortByPriority(items) {
    return items.slice().sort(compareByPriority);
  }

  /**
   * Returns the single highest-priority (most severe) item, or undefined
   * if the input array is empty.
   * @param {Array<Object>} items
   * @returns {Object|undefined}
   */
  function getHighestPriority(items) {
    return sortByPriority(items)[0];
  }

  /**
   * Counts items by severity tier, e.g. {CRITICAL: 1, HIGH: 2, MEDIUM: 0,
   * NONE: 1}. Used by the detector-summary page to show a count per
   * detector without duplicating ranking logic there.
   * @param {Array<Object>} items
   * @returns {Object}
   */
  function countBySeverity(items) {
    const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, NONE: 0 };
    items.forEach(function (item) {
      if (counts.hasOwnProperty(item.severity)) {
        counts[item.severity] += 1;
      }
    });
    return counts;
  }

  return {
    SEVERITY_RANK: SEVERITY_RANK,
    compareByPriority: compareByPriority,
    sortByPriority: sortByPriority,
    getHighestPriority: getHighestPriority,
    countBySeverity: countBySeverity,
  };
})();
