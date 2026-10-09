/**
 * summary.js — controller for the "Detector overview" page (summary.html,
 * Step 2).
 *
 * Reads the report load.js stored (report-store.js) and renders:
 *  - one card per detector: real FAIL/WARNING/PASS counts for detectors
 *    analysed on the server (report.server.methods[id], shaped like
 *    src/qc_report/contract.py's QCMethodResult summary), placeholder
 *    counts for the rest;
 *  - per server-analysed method, a table of its jobs with Job, Instrument(s)
 *    and counts, each linking to that job on the detail page;
 *  - the real-row table (rows.js/table.js) filterable by sample type,
 *    company status and analyte.
 */
(function () {
  "use strict";

  const rowFilterState = { type: "ALL", tag: "ALL", analyte: "ALL" };

  function summariseCounts(counts) {
    const total = counts.FAIL + counts.WARNING + counts.PASS;
    if (total === 0) {
      return "No results.";
    }
    return counts.FAIL + " fail, " + counts.WARNING + " warning, " + counts.PASS + " pass";
  }

  function detailLink(detectorId, jobKey) {
    return "detail.html?detector=" + encodeURIComponent(detectorId) +
      (jobKey !== undefined ? "&job=" + encodeURIComponent(jobKey) : "");
  }

  function addText(parent, tag, className, text) {
    const el = document.createElement(tag);
    if (className) {
      el.className = className;
    }
    el.textContent = text;
    parent.appendChild(el);
    return el;
  }

  function renderCard(detector, check, method) {
    const card = document.createElement("article");
    const usable = method ? method.usable : check.usable;
    card.className = "detector-card" + (usable ? "" : " detector-card-disabled");

    addText(card, "h3", "", detector.label);
    addText(card, "span", "usable-badge " + (usable ? "usable-yes" : "usable-no"), usable ? "Usable" : "Not usable");

    if (method) {
      addText(card, "p", "detector-card-note", "Real analysis (history " + method.history_mode + ").");
      if (usable) {
        addText(card, "p", "detector-card-count",
          summariseCounts(method.counts) + " across " + method.jobs.length + " job(s).");
      }
    } else if (detector.serverAnalysis) {
      addText(card, "p", "detector-card-note",
        "Real analysis unavailable: the local server did not analyse this file.");
    } else {
      addText(card, "p", "detector-card-note", "Not connected yet: placeholder data.");
      addText(card, "p", "detector-card-count",
        summariseCounts(window.LCSPoc.ranking.countByState(window.LCSPoc.data.getPlaceholderItems(detector.id))));
    }

    if (usable && !(detector.serverAnalysis && !method)) {
      const link = document.createElement("a");
      link.className = "detector-card-button";
      link.href = detailLink(detector.id);
      link.textContent = "View details ›";
      card.appendChild(link);
    } else {
      const bits = [];
      if (method && method.notices && method.notices.length) {
        bits.push(method.notices.join(" "));
      } else {
        if (check.missingColumns.length > 0) {
          bits.push("Missing columns: " + check.missingColumns.join(", "));
        }
        if (check.matchingRowCount === 0) {
          bits.push("No rows of this sample type in the loaded file.");
        }
      }
      addText(card, "p", "detector-card-reason", bits.join(" ") || "Cannot run against the loaded file.");
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
    const methods = (report.server && report.server.methods) || {};

    window.LCSPoc.data.getDetectors().forEach(function (detector) {
      const check = report.checks.find(function (c) { return c.id === detector.id; });
      if (!check) {
        return; // report predates a detector added later — skip rather than crash
      }
      grid.appendChild(renderCard(detector, check, methods[detector.id]));
    });
  }

  /** Per-method job table: Job, Instrument(s), first analysed, counts, link. */
  function renderMethodResults(report) {
    const container = document.getElementById("method-results");
    container.textContent = "";
    const methods = (report.server && report.server.methods) || {};

    Object.keys(methods).forEach(function (methodId) {
      const method = methods[methodId];
      if (!method.usable || !method.jobs || method.jobs.length === 0) {
        return;
      }
      const section = document.createElement("section");
      section.className = "method-results";
      addText(section, "h2", "", method.label + " — results by job");

      const notices = document.createElement("ul");
      notices.className = "notice-list";
      (method.notices || []).forEach(function (text) { addText(notices, "li", "", text); });
      section.appendChild(notices);

      const scroll = document.createElement("div");
      scroll.className = "job-table-scroll";
      const table = document.createElement("table");
      table.className = "job-table";
      const headRow = document.createElement("tr");
      ["Job", "Instrument", "First analysed", "Fail", "Warning", "Pass", ""].forEach(function (h) {
        addText(headRow, "th", "", h);
      });
      const thead = document.createElement("thead");
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      method.jobs.forEach(function (job) {
        const tr = document.createElement("tr");
        addText(tr, "td", "", job.job);
        addText(tr, "td", "", (job.instruments || []).join(", ") || "Instrument unknown");
        addText(tr, "td", "", job.first_date || "");
        ["FAIL", "WARNING", "PASS"].forEach(function (state) {
          addText(tr, "td", "count-cell", String(job.counts[state] || 0));
        });
        const linkCell = document.createElement("td");
        const link = document.createElement("a");
        link.href = detailLink(methodId, job.key);
        link.textContent = "View ›";
        linkCell.appendChild(link);
        tr.appendChild(linkCell);
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      scroll.appendChild(table);
      section.appendChild(scroll);
      container.appendChild(section);
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
    noteEl.hidden = notes.length === 0;
    noteEl.textContent = notes.join(" ");
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
    document.getElementById("summary-history").textContent = report.server
      ? (report.server.history_mode === "on" ? "on" : "off")
      : "not analysed";
    renderCards(report);
    renderMethodResults(report);
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
