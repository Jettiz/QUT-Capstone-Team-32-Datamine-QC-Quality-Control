/**
 * app.js — controller for the detail page (detail.html, Step 3).
 *
 * Reads ?detector=<id>&job=<key> from the URL. Two modes, one rendering path:
 *  - REAL: the detector is a QC method registered on the server
 *    (src/qc_report/registry.py) and the Load page stored its run summary.
 *    One job's items are fetched from poc/server.py; charts are PNG URLs
 *    rendered on demand by the server. A job <select> switches jobs without
 *    reloading the page.
 *  - PLACEHOLDER: any other detector -- data.js's placeholder items.
 * Both produce QCItem-shaped items (src/qc_report/contract.py), grouped by
 * `code` (ANALYTE_CODE for LCS) via ranking.groupByCode(); render.js paints
 * them. Selecting another code (or variant tab) only repaints the panel and
 * swaps the chart image -- nothing is re-analysed.
 *
 * After every selection change this fires a "lcspoc:selectionchange" event
 * on document so detail-rows.js can follow the selected analyte.
 */
(function () {
  "use strict";

  const DEFAULT_DETECTOR_ID = "control";
  const state = { detectorId: null, runId: null, groups: [], activeCode: null };

  function getParams() {
    return new URLSearchParams(window.location.search);
  }

  function notifySelectionChange() {
    document.dispatchEvent(new CustomEvent("lcspoc:selectionchange"));
  }

  function showMessage(message) {
    document.getElementById("item-controls").textContent = message;
    document.getElementById("variant-tabs").hidden = true;
    document.getElementById("viz-panel").hidden = true;
  }

  function renderNotices(notices) {
    const list = document.getElementById("detail-notices");
    list.textContent = "";
    (notices || []).forEach(function (text) {
      const li = document.createElement("li");
      li.textContent = text;
      list.appendChild(li);
    });
    list.hidden = !notices || notices.length === 0;
  }

  function findGroup(code) {
    return state.groups.find(function (g) { return g.code === code; });
  }

  function selectItem(group, itemId) {
    const item = group.items.find(function (i) { return i.id === itemId; }) || group.items[0];
    state.activeCode = group.code;
    window.LCSPoc.render.renderActivePanel(item);
    window.LCSPoc.render.setActiveControl(group.code);
    window.LCSPoc.render.renderVariantTabs(group, item.id);
    notifySelectionChange();
  }

  function selectCode(code) {
    const group = findGroup(code);
    if (group) {
      selectItem(group, group.items[0].id); // worst variant first
    }
  }

  function showItems(items) {
    if (!items || items.length === 0) {
      showMessage("No results to show for this selection.");
      return;
    }
    document.getElementById("viz-panel").hidden = false;
    state.groups = window.LCSPoc.ranking.groupByCode(items);
    window.LCSPoc.render.renderControls(state.groups, state.groups[0].code);
    selectCode(state.groups[0].code);
  }

  // ── REAL mode ─────────────────────────────────────────────────────────────

  function loadJob(jobKey, updateUrl) {
    document.getElementById("item-controls").textContent = "Loading results…";
    window.LCSPoc.api.getJob(state.runId, state.detectorId, jobKey).then(function (job) {
      window.LCSPoc.render.renderContext(job.job, (job.instruments || []).join(", "));
      const items = job.items.map(function (item) {
        return Object.assign({}, item, {
          imageUrl: item.has_chart ? window.LCSPoc.api.chartUrl(state.runId, state.detectorId, item.id) : null,
          imageAlt: item.code + " " + (item.variant || "") + " control chart: " + item.status_detail,
        });
      });
      showItems(items);
      if (updateUrl) {
        window.history.replaceState(null, "", "detail.html?detector=" + encodeURIComponent(state.detectorId) +
          "&job=" + encodeURIComponent(jobKey));
      }
    }).catch(function (err) {
      showMessage(err.message);
    });
  }

  function initReal(detector, run, method) {
    state.runId = run.run_id;
    document.getElementById("detector-subtitle").textContent =
      "Real analysis of " + run.file_name + " (history " + run.history_mode + "). " +
      "Select an analyte to see why it received its state.";
    renderNotices(method.notices);

    if (!method.usable) {
      showMessage((method.notices || []).join(" ") || (detector.label + " cannot run on the loaded file."));
      return;
    }
    if (!method.jobs || method.jobs.length === 0) {
      showMessage("No " + detector.label + " results were produced for this file.");
      return;
    }

    const requested = getParams().get("job");
    const jobKey = method.jobs.some(function (j) { return j.key === requested; }) ? requested : method.jobs[0].key;
    window.LCSPoc.render.renderJobSelect(method.jobs, jobKey);
    document.getElementById("job-select").addEventListener("change", function (event) {
      loadJob(event.target.value, true);
    });
    loadJob(jobKey, false);
  }

  // ── PLACEHOLDER mode ─────────────────────────────────────────────────────

  function initPlaceholder(detector) {
    document.getElementById("detector-subtitle").textContent =
      "Placeholder data: " + detector.label + " is not connected to a real analysis yet.";
    window.LCSPoc.render.renderContext("Job unknown", "Instrument unknown");
    window.LCSPoc.render.renderJobSelect([], "");
    renderNotices([]);
    showItems(window.LCSPoc.data.getPlaceholderItems(detector.id));
  }

  function init() {
    const detectorId = getParams().get("detector") || DEFAULT_DETECTOR_ID;
    const detector = window.LCSPoc.data.getDetector(detectorId);
    state.detectorId = detectorId;

    if (!detector) {
      document.getElementById("detector-title").textContent = "Unknown detector";
      document.getElementById("detector-subtitle").textContent =
        "\"" + detectorId + "\" is not a recognised detector type.";
      showMessage("");
      return;
    }

    document.title = detector.label + " — Detail (POC)";
    document.getElementById("detector-title").textContent = detector.label + " — detail";

    document.getElementById("item-controls").addEventListener("click", function (event) {
      const button = event.target.closest("button[data-item-code]");
      if (button) {
        selectCode(button.dataset.itemCode);
      }
    });
    document.getElementById("variant-tabs").addEventListener("click", function (event) {
      const tab = event.target.closest("button[data-item-id]");
      const group = findGroup(state.activeCode);
      if (tab && group) {
        selectItem(group, tab.dataset.itemId);
      }
    });

    const report = window.LCSPoc.reportStore.loadReport();
    const run = report && report.server;
    const method = run && run.methods && run.methods[detectorId];
    if (method) {
      initReal(detector, run, method);
    } else if (detector.serverAnalysis) {
      document.getElementById("detector-subtitle").textContent = "No real analysis is available.";
      window.LCSPoc.render.renderContext("Job unknown", "Instrument unknown");
      window.LCSPoc.render.renderJobSelect([], "");
      showMessage(detector.label + " results come from the local server. " + window.LCSPoc.api.NOT_SERVED_MESSAGE +
        " Then load the file again on step 1.");
    } else {
      initPlaceholder(detector);
    }
  }

  document.addEventListener("DOMContentLoaded", init);
})();
