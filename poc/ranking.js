/**
 * ranking.js — pure warning-state ranking logic. No DOM access, no data.
 *
 * Mirrors src/qc_status.py's STATE_RANK (FAIL=0, WARNING=1, PASS=2, lower =
 * worse) with the tie-break the server uses (larger `magnitude` first; for
 * LCS that is |offset|).
 *
 * Works on any items shaped like src/qc_report/contract.py's QCItem (real
 * server items and data.js placeholders alike), so it never needs to know
 * which QC method produced them.
 */
window.LCSPoc = window.LCSPoc || {};

window.LCSPoc.ranking = (function () {
  "use strict";

  const STATES = ["FAIL", "WARNING", "PASS"];
  const STATE_RANK = { FAIL: 0, WARNING: 1, PASS: 2 };
  const STATE_LABELS = { FAIL: "Fail", WARNING: "Warning", PASS: "Pass" };

  function rankOf(state) {
    return STATE_RANK.hasOwnProperty(state) ? STATE_RANK[state] : 9;
  }

  /**
   * Compares two items for sorting worst-first.
   * Primary key: state rank (lower number wins).
   * Secondary key (tie-break): larger absolute magnitude wins.
   */
  function compareByPriority(a, b) {
    const diff = rankOf(a.state) - rankOf(b.state);
    if (diff !== 0) {
      return diff;
    }
    return Math.abs(b.magnitude || 0) - Math.abs(a.magnitude || 0);
  }

  function sortByPriority(items) {
    return items.slice().sort(compareByPriority);
  }

  function getHighestPriority(items) {
    return sortByPriority(items)[0];
  }

  /** @returns {string} the worst state among `states` ("PASS" if empty) */
  function worstState(states) {
    return states.reduce(function (worst, s) { return rankOf(s) < rankOf(worst) ? s : worst; }, "PASS");
  }

  /**
   * Groups items by `code` (e.g. ANALYTE_CODE): one group per code, holding
   * all its variants (e.g. one per scheme), the group's worst state and
   * largest magnitude. Groups are returned worst-first.
   * @returns {Array<{code: string, state: string, magnitude: number, items: Array<Object>}>}
   */
  function groupByCode(items) {
    const byCode = {};
    const order = [];
    items.forEach(function (item) {
      if (!byCode[item.code]) {
        byCode[item.code] = [];
        order.push(item.code);
      }
      byCode[item.code].push(item);
    });
    const groups = order.map(function (code) {
      const groupItems = sortByPriority(byCode[code]);
      return {
        code: code,
        state: groupItems[0].state,
        magnitude: Math.abs(groupItems[0].magnitude || 0),
        items: groupItems,
      };
    });
    return groups.sort(compareByPriority);
  }

  /** Counts items per state, e.g. {FAIL: 1, WARNING: 2, PASS: 4}. */
  function countByState(items) {
    const counts = { FAIL: 0, WARNING: 0, PASS: 0 };
    items.forEach(function (item) {
      if (counts.hasOwnProperty(item.state)) {
        counts[item.state] += 1;
      }
    });
    return counts;
  }

  return {
    STATES: STATES,
    STATE_RANK: STATE_RANK,
    STATE_LABELS: STATE_LABELS,
    compareByPriority: compareByPriority,
    sortByPriority: sortByPriority,
    getHighestPriority: getHighestPriority,
    worstState: worstState,
    groupByCode: groupByCode,
    countByState: countByState,
  };
})();
