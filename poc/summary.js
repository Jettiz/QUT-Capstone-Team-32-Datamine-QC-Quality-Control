/**
 * summary.js — controller for the "Detector overview" page (summary.html).
 *
 * Reads the report load.js stored in sessionStorage (which detectors are
 * usable against the file the user chose on index.html), pairs each
 * detector with a dummy anomaly count from data.js (via
 * ranking.countBySeverity), and renders one card per detector. Only
 * detectors the load step found usable link through to their detail page
 * (detail.html?detector=<id>) — the same page built in Phase 1,
 * generalised in this step to work for any detector.
 *
 * Also renders a real-data table (rows.js/table.js) below the cards: up to
 * 15 real rows from the uploaded file, worst Detected-issue tag first,
 * filterable by sample type and tag. See rows.js's module docstring for
 * where the Detected-issue tag comes from and the storage/sampling design
 * that keeps this bounded regardless of source file size.
 *
 * PHASE 2 INTEGRATION POINT: countBySeverity(data.getItems(id)) below
 * would become a real per-detector anomaly count from that detector's
 * actual output, once each one exists — the card-rendering logic itself
 * would not need to change.
 */
(function () {
  "use strict";

  const rowFilterState = { type: "ALL", tag: "ALL", analyte: "ALL" };

  function summariseCounts(counts) {
    const total = counts.CRITICAL + counts.HIGH + counts.MEDIUM + counts.NONE;
    const flagged = counts.CRITICAL + counts.HIGH + counts.MEDIUM;
    if (total === 0) {
      return "No dummy items defined.";
    }
    if (flagged === 0) {
      return "No anomalies detected (" + total + " item(s) checked).";
    }
    const parts = [];
    if (counts.CRITICAL) parts.push(counts.CRITICAL + " Critical");
    if (counts.HIGH) parts.push(counts.HIGH + " High");
    if (counts.MEDIUM) parts.push(counts.MEDIUM + " Medium");
    return flagged + " anomal" + (flagged === 1 ? "y" : "ies") + " detected (" + parts.join(", ") + ")";
  }

  function renderCard(detector, check) {
    const card = document.createElement("article");
    card.className = "detector-card" + (check.usable ? "" : " detector-card-disabled");

    const heading = document.createElement("h3");
    heading.textContent = detector.label;
    card.appendChild(heading);

    const badge = document.createElement("span");
    badge.className = "usable-badge " + (check.usable ? "usable-yes" : "usable-no");
    badge.textContent = check.usable ? "Usable" : "Not usable";
    card.appendChild(badge);

    if (!check.validatorImplemented) {
      const note = document.createElement("p");
      note.className = "detector-card-note";
      note.textContent = "Backend validator not implemented yet — best-effort check only.";
      card.appendChild(note);
    }

    const counts = window.LCSPoc.ranking.countBySeverity(window.LCSPoc.data.getItems(detector.id));
    const countLine = document.createElement("p");
    countLine.className = "detector-card-count";
    countLine.textContent = summariseCounts(counts);
    card.appendChild(countLine);

    if (check.usable) {
      const link = document.createElement("a");
      link.className = "detector-card-button";
      link.href = "detail.html?detector=" + encodeURIComponent(detector.id);
      link.textContent = "View details ›";
      card.appendChild(link);
    } else {
      const reason = document.createElement("p");
      reason.className = "detector-card-reason";
      const bits = [];
      if (check.missingColumns.length > 0) {
        bits.push("Missing columns: " + check.missingColumns.join(", "));
      }
      if (check.matchingRowCount === 0) {
        bits.push("No \"" + check.analyticalTypeFilter + "\" rows found in the loaded file.");
      }
      reason.textContent = bits.join(" ") || "Cannot run against the loaded file.";
      card.appendChild(reason);

      const disabledButton = document.createElement("button");
      disabledButton.type = "button";
      disabledButton.className = "detector-card-button";
      disabledButton.disabled = true;
      disabledButton.textContent = "View details ›";
      card.appendChild(disabledButton);
    }

    return card;
  }

  function renderCards(report) {
    const grid = document.getElementById("detector-grid");
    grid.textContent = "";

    window.LCSPoc.data.getDetectors().forEach(function (detector) {
      const check = report.checks.find(function (c) { return c.id === detector.id; });
      if (!check) {
        return; // report predates a detector added later — skip rather than crash
      }
      grid.appendChild(renderCard(detector, check));
    });
  }

  function renderRowSampleNote(report) {
    const noteEl = document.getElementById("row-sample-note");
    const notes = [];
    if (!report.rowSample.hasAnalyticalTypeColumn) {
      notes.push("No ANALYTICAL_TYPE column was found — sample-type filtering has nothing to go on.");
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

  function refreshRowTable(report) {
    const result = window.LCSPoc.rows.getTopRows(
      report.rowSample, rowFilterState.type, rowFilterState.tag, rowFilterState.analyte,
      window.LCSPoc.rows.TABLE_DISPLAY_LIMIT
    );
    window.LCSPoc.table.renderRowTable(document.getElementById("row-sample-table"), report.columns, result.rows);
    window.LCSPoc.table.renderRowCountMessage(document.getElementById("row-count-message"), result.rows.length, result.totalMatching);
  }

  function initRowSampleSection(report) {
    renderRowSampleNote(report);

    const typeSelect = document.getElementById("row-type-filter");
    const tagSelect = document.getElementById("row-tag-filter");
    const analyteSelect = document.getElementById("row-analyte-filter");

    window.LCSPoc.table.renderTypeOptions(typeSelect, window.LCSPoc.rows.KNOWN_TYPES, rowFilterState.type);
    window.LCSPoc.table.renderTagOptions(tagSelect, window.LCSPoc.rows.TAGS, rowFilterState.tag);
    window.LCSPoc.table.renderAnalyteOptions(analyteSelect, report.rowSample.analyteCodesAll, rowFilterState.analyte);

    typeSelect.addEventListener("change", function () {
      rowFilterState.type = typeSelect.value;
      refreshRowTable(report);
    });
    tagSelect.addEventListener("change", function () {
      rowFilterState.tag = tagSelect.value;
      refreshRowTable(report);
    });
    analyteSelect.addEventListener("change", function () {
      rowFilterState.analyte = analyteSelect.value;
      refreshRowTable(report);
    });

    refreshRowTable(report);
  }

  function render(report) {
    document.getElementById("summary-file-name").textContent = report.fileName;
    renderCards(report);
    initRowSampleSection(report);
  }

  function init() {
    const report = window.LCSPoc.reportStore.loadReport();
    if (!report) {
      document.getElementById("no-report-message").hidden = false;
      document.getElementById("summary-content").hidden = true;
      return;
    }
    render(report);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
