/**
 * report-store.js — the one place that reads/writes the Load page's report
 * object to sessionStorage. Extracted so load.js (writer), summary.js
 * (reader), and detail-rows.js (reader) don't each duplicate the same
 * JSON-guard logic.
 */
window.LCSPoc = window.LCSPoc || {};

window.LCSPoc.reportStore = (function () {
  "use strict";

  const STORAGE_KEY = "lcsPocLoadReport";

  /**
   * @param {Object} report
   * @throws if sessionStorage is unavailable or the report can't be serialised
   */
  function saveReport(report) {
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(report));
  }

  /**
   * @returns {Object|null} the stored report, or null if none exists, or
   *   sessionStorage is unavailable (e.g. some private-browsing modes), or
   *   the stored content is corrupted/unparsable.
   */
  function loadReport() {
    let raw;
    try {
      raw = sessionStorage.getItem(STORAGE_KEY);
    } catch (err) {
      return null;
    }
    if (!raw) {
      return null;
    }
    try {
      return JSON.parse(raw);
    } catch (err) {
      return null;
    }
  }

  return {
    STORAGE_KEY: STORAGE_KEY,
    saveReport: saveReport,
    loadReport: loadReport,
  };
})();
