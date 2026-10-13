/**
 * api.js — the only file that talks to poc/server.py.
 *
 * The real QC analysis runs in Python (src/qc_report, src/detectors); the
 * browser uploads the file once, keeps the returned run summary (via
 * report-store.js), and fetches one job's items / one chart at a time.
 * Result shapes are documented in src/qc_report/contract.py.
 */
window.LCSPoc = window.LCSPoc || {};

window.LCSPoc.api = (function () {
  "use strict";

  const NOT_SERVED_MESSAGE =
    "The real analysis needs the local server. Start it with `python poc/server.py` " +
    "and open http://localhost:8000 instead of opening this file directly.";

  /** @returns {boolean} true when the page was served by poc/server.py (not file://) */
  function isServed() {
    return window.location.protocol === "http:" || window.location.protocol === "https:";
  }

  function handle(response) {
    return response.json().catch(function () {
      return { error: "Unexpected response from the server (HTTP " + response.status + ")." };
    }).then(function (body) {
      if (!response.ok) {
        throw new Error(body.error || ("Request failed (HTTP " + response.status + ")."));
      }
      return body;
    });
  }

  function request(url, options) {
    if (!isServed()) {
      return Promise.reject(new Error(NOT_SERVED_MESSAGE));
    }
    return fetch(url, options).then(handle, function () {
      throw new Error("Could not reach the local server. Is `python poc/server.py` still running?");
    });
  }

  /**
   * Upload a file and run every registered QC method on it.
   * @param {File} file
   * @param {boolean} useHistory the Load page's History toggle
   * @returns {Promise<Object>} run summary {run_id, file_name, history_mode, n_rows, methods}
   */
  function analyse(file, useHistory) {
    const url = "/api/analyse?history=" + (useHistory ? "on" : "off") +
      "&filename=" + encodeURIComponent(file.name);
    return request(url, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
  }

  /** @returns {Promise<Object>} one job: {key, job, instruments, first_date, counts, items[]} */
  function getJob(runId, methodId, jobKey) {
    return request("/api/runs/" + encodeURIComponent(runId) + "/methods/" +
      encodeURIComponent(methodId) + "/jobs/" + encodeURIComponent(jobKey));
  }

  /** @returns {string} URL of an item's chart PNG (rendered on first request) */
  function chartUrl(runId, methodId, itemId) {
    return "/api/runs/" + encodeURIComponent(runId) + "/methods/" +
      encodeURIComponent(methodId) + "/charts/" + encodeURIComponent(itemId) + ".png";
  }

  /**
   * Fetch a chart PNG as a Blob. Pass an AbortController's signal so a
   * superseded chart request (the user picked another analyte) is cancelled
   * instead of tying up a browser connection.
   * @param {string} url from chartUrl()
   * @param {AbortSignal} signal
   * @returns {Promise<Blob>}
   */
  function fetchChart(url, signal) {
    return fetch(url, { signal: signal }).then(function (response) {
      if (!response.ok) {
        throw new Error("Chart request failed (HTTP " + response.status + ").");
      }
      return response.blob();
    });
  }

  return {
    NOT_SERVED_MESSAGE: NOT_SERVED_MESSAGE,
    isServed: isServed,
    analyse: analyse,
    getJob: getJob,
    chartUrl: chartUrl,
    fetchChart: fetchChart,
  };
})();
