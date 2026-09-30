/**
 * detail-rows.js — controller for the Detail page's real-data row table
 * (the new #row-sample-section in detail.html).
 *
 * Deliberately a SEPARATE, additive DOMContentLoaded listener from app.js:
 * it touches only the #row-sample-section subtree, never app.js's
 * #viz-panel/#controls-panel, so the existing dummy-item viewer keeps
 * working exactly as before regardless of anything in this file. It
 * duplicates app.js's small URL-parsing helper locally rather than reaching
 * into app.js's private IIFE, per this POC's existing "small, focused
 * files" convention.
 *
 * The sample-type filter is fixed to detector.analyticalTypeFilter -- the
 * exact same field load.js's own usability check keys off -- and rendered
 * as a plain label (table.js's renderLockedTypeLabel), never a <select>, so
 * there is no way to change or remove it here: the whole point is
 * preventing a user from accidentally viewing another sample type's rows
 * while looking at one detector's page.
 *
 * The Analyte filter's "Currently shown" / "Others" grouping tracks
 * whichever dummy item app.js currently has active in the visualisation
 * panel. Rather than reaching into app.js's private state, this reads the
 * DOM it already renders (#panel-code's textContent, written by
 * render.js's renderActivePanel()) and adds its OWN click listener on
 * #item-controls -- since script tags run in document order and
 * DOMContentLoaded listeners fire in registration order, app.js's listener
 * (registered first, in the earlier script block) always finishes updating
 * #panel-code before this file's listener (registered after) reads it, so
 * this never sees a stale value.
 */
(function () {
  "use strict";

  const DEFAULT_DETECTOR_ID = "control";
  let tagFilter = "ALL";
  let analyteFilter = "ALL";

  function getRequestedDetectorId() {
    const params = new URLSearchParams(window.location.search);
    return params.get("detector") || DEFAULT_DETECTOR_ID;
  }

  function showMessage(message) {
    document.getElementById("no-row-sample-message").hidden = false;
    document.getElementById("no-row-sample-message").textContent = message;
    document.getElementById("row-sample-content").hidden = true;
  }

  function renderRowSampleNote(report) {
    const noteEl = document.getElementById("row-sample-note");
    const notes = [];
    if (!report.rowSample.hasAnalyticalTypeColumn) {
      notes.push("No ANALYTICAL_TYPE column was found in the loaded file — the sample-type lock below has nothing to match against.");
    }
    if (!report.rowSample.hasStandardStatusColumn && !report.rowSample.hasPrecisionStatusColumn) {
      notes.push("Neither STANDARD_STATUS nor PRECISION_STATUS was found — every row is tagged \"Unknown\".");
    }
    if (!report.rowSample.hasAnalyteCodeColumn) {
      notes.push("No ANALYTE_CODE column was found — analyte filtering has nothing to go on.");
    }
    if (notes.length > 0) {
      noteEl.hidden = false;
      noteEl.textContent = notes.join(" ");
    } else {
      noteEl.hidden = true;
    }
  }

  /** @returns {string} the analyte code currently shown in the dummy viz panel */
  function getCurrentlyShownCode() {
    return document.getElementById("panel-code").textContent.trim();
  }

  function refreshAnalyteDropdown(report, lockedType) {
    const analyteSelect = document.getElementById("row-analyte-filter");
    const currentCode = getCurrentlyShownCode();
    const codesForType = report.rowSample.analyteCodesByType[lockedType] || [];
    window.LCSPoc.table.renderAnalyteOptionsGrouped(analyteSelect, currentCode, codesForType, analyteFilter);
    // renderAnalyteOptionsGrouped() may have reset the <select> to "ALL" if
    // the previous selection no longer exists as an option -- keep our own
    // state in sync with whatever it actually landed on.
    analyteFilter = analyteSelect.value;
  }

  function refresh(report, lockedType) {
    const result = window.LCSPoc.rows.getTopRows(report.rowSample, lockedType, tagFilter, analyteFilter, window.LCSPoc.rows.TABLE_DISPLAY_LIMIT);
    window.LCSPoc.table.renderRowTable(document.getElementById("row-sample-table"), report.columns, result.rows);
    window.LCSPoc.table.renderRowCountMessage(document.getElementById("row-count-message"), result.rows.length, result.totalMatching);
  }

  function init() {
    const detectorId = getRequestedDetectorId();
    const detector = window.LCSPoc.data.getDetector(detectorId);

    if (!detector) {
      showMessage("\"" + detectorId + "\" is not a recognised detector type — no real data to show.");
      return;
    }

    const report = window.LCSPoc.reportStore.loadReport();
    if (!report) {
      showMessage("No sample has been loaded yet. Go back and load a sample first, then come back here.");
      return;
    }

    const check = report.checks.find(function (c) { return c.id === detectorId; });
    if (!check) {
      showMessage("No data available for " + detector.label + " in the loaded file's analysis.");
      return;
    }
    if (!check.usable) {
      showMessage(
        detector.label + " was not usable against the loaded file, so no real rows can be shown here. " +
        "Go back to the overview page to see why."
      );
      return;
    }

    document.getElementById("no-row-sample-message").hidden = true;
    document.getElementById("row-sample-content").hidden = false;

    const lockedType = detector.analyticalTypeFilter;
    renderRowSampleNote(report);
    window.LCSPoc.table.renderLockedTypeLabel(document.getElementById("row-type-locked"), lockedType);

    const tagSelect = document.getElementById("row-tag-filter");
    window.LCSPoc.table.renderTagOptions(tagSelect, window.LCSPoc.rows.TAGS, tagFilter);
    tagSelect.addEventListener("change", function () {
      tagFilter = tagSelect.value;
      refresh(report, lockedType);
    });

    refreshAnalyteDropdown(report, lockedType);
    document.getElementById("row-analyte-filter").addEventListener("change", function (event) {
      analyteFilter = event.target.value;
      refresh(report, lockedType);
    });

    // Re-group "Currently shown"/"Others" whenever the dummy viz panel's
    // active item changes. Registered after app.js's own #item-controls
    // listener (this script loads later in detail.html), so #panel-code is
    // already up to date by the time this runs.
    const itemControls = document.getElementById("item-controls");
    if (itemControls) {
      itemControls.addEventListener("click", function (event) {
        if (!event.target.closest("button[data-item-code]")) {
          return;
        }
        refreshAnalyteDropdown(report, lockedType);
        refresh(report, lockedType);
      });
    }

    refresh(report, lockedType);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
