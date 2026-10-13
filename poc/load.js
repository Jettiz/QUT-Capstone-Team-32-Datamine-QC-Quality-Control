/**
 * load.js — controller for the "Load a sample" page (index.html, Step 1).
 *
 * On "Analyse sample" two things happen in parallel:
 *  1. REAL analysis: the file is uploaded to poc/server.py (api.js), which
 *     loads it with src/data_loader.py, validates it with
 *     src/data_validator.py and runs every QC method registered in
 *     src/qc_report/registry.py (currently LCS / Control), using the History
 *     toggle on this page. The returned run summary is stored with the
 *     report (report.server) for Steps 2 and 3.
 *  2. Browser-side PREVIEW: the file is read locally (File API) for the
 *     "which detectors can use this data?" table and the real-row tables.
 *     For detectors analysed on the server, the server's real validation
 *     replaces the preview's verdict; for the others the preview is a
 *     simplified stand-in for their validator (column presence + at least
 *     one row of the right type -- no null/dtype checks).
 *
 * Limitations of the preview, deliberately kept simple:
 * - CSV rows are split on a plain "," -- a field containing an embedded
 *   comma would misalign columns. None of the columns this reads are
 *   expected to contain one in real CCLAS exports.
 * - Only the first MAX_ROWS_SCANNED data rows are scanned, as a safety
 *   ceiling against a pathologically huge file freezing the tab. A real
 *   99,999-row/16.7MB export scans in well under a second; sample types are
 *   NOT evenly interleaved in real exports, so this must not be tuned down.
 */
(function () {
  "use strict";

  const MAX_ROWS_SCANNED = 250000;

  function splitLines(text) {
    const lines = text.split(/\r\n|\n|\r/);
    while (lines.length && lines[lines.length - 1].trim() === "") {
      lines.pop();
    }
    // Some real CCLAS exports (e.g. data/raw/ResultSet.csv) start with a
    // blank line before the real header -- skip any leading blank lines too.
    while (lines.length && lines[0].trim() === "") {
      lines.shift();
    }
    return lines;
  }

  // Simple, POC-level CSV field split -- see module docstring's Limitations.
  function splitCsvLine(line) {
    return line.split(",").map(function (field) {
      return field.trim().replace(/^"|"$/g, "");
    });
  }

  function parseHeader(firstLine) {
    return splitCsvLine(firstLine).map(function (c) { return c.toUpperCase(); });
  }

  function resolveColumnIndexes(header) {
    const names = window.LCSPoc.rows.COLUMN_NAMES;
    return {
      analyticalType: header.indexOf(names.analyticalType),
      stdLotCode: header.indexOf(names.stdLotCode),
      standardStatus: header.indexOf(names.standardStatus),
      precisionStatus: header.indexOf(names.precisionStatus),
      analyteCode: header.indexOf(names.analyteCode),
    };
  }

  /**
   * Single pass over the file's data lines: classifies each row once (type
   * incl. the LCS/SRM split, company-status tag, analyte), counts rows per
   * detector rowType and samples rows for the tables (rows.js).
   */
  function scanDataLines(header, dataLines, detectors) {
    const colIdx = resolveColumnIndexes(header);
    const hasAnalyticalTypeColumn = colIdx.analyticalType !== -1;

    const matchingCounts = {};
    detectors.forEach(function (d) {
      matchingCounts[d.id] = hasAnalyticalTypeColumn ? 0 : null;
    });

    const sampler = window.LCSPoc.rows.createRowSampler(window.LCSPoc.rows.ROWS_PER_BUCKET_CAP);
    const scanLimit = Math.min(dataLines.length, MAX_ROWS_SCANNED);

    for (let i = 0; i < scanLimit; i++) {
      const fields = splitCsvLine(dataLines[i]);

      const canonicalType = window.LCSPoc.rows.classifyRowType(
        fields[colIdx.analyticalType], colIdx.stdLotCode !== -1 ? fields[colIdx.stdLotCode] : ""
      );
      if (hasAnalyticalTypeColumn) {
        detectors.forEach(function (d) {
          if (canonicalType === d.rowType) {
            matchingCounts[d.id] += 1;
          }
        });
      }

      const standardStatusRaw = colIdx.standardStatus !== -1 ? fields[colIdx.standardStatus] : "";
      const precisionStatusRaw = colIdx.precisionStatus !== -1 ? fields[colIdx.precisionStatus] : "";
      const analyteCodeRaw = colIdx.analyteCode !== -1 ? fields[colIdx.analyteCode] : "";
      const tag = window.LCSPoc.rows.classifyRowTag(canonicalType, standardStatusRaw, precisionStatusRaw);
      const statusRawUsed = window.LCSPoc.rows.pickStatusColumnRaw(canonicalType, standardStatusRaw, precisionStatusRaw);
      sampler.addRow(canonicalType, tag, analyteCodeRaw, fields, statusRawUsed);
    }

    return {
      matchingCounts: matchingCounts,
      hasAnalyticalTypeColumn: hasAnalyticalTypeColumn,
      hasStandardStatusColumn: colIdx.standardStatus !== -1,
      hasPrecisionStatusColumn: colIdx.precisionStatus !== -1,
      hasAnalyteCodeColumn: colIdx.analyteCode !== -1,
      scanLimit: scanLimit,
      rowSample: sampler.build(),
    };
  }

  /**
   * Preview check for one detector: required columns present ("A|B" = either
   * alias) and at least one row of its rowType. No file scanning here.
   */
  function checkDetector(detector, header, matchingRowCount) {
    const missingColumns = detector.requiredColumns.filter(function (spec) {
      return !spec.split("|").some(function (col) { return header.indexOf(col) !== -1; });
    }).map(function (spec) { return spec.split("|").join(" or "); });

    return {
      id: detector.id,
      label: detector.label,
      validatorImplemented: detector.validatorImplemented,
      rowType: detector.rowType,
      missingColumns: missingColumns,
      matchingRowCount: matchingRowCount,
      usable: missingColumns.length === 0 && (matchingRowCount === null || matchingRowCount > 0),
      source: "preview",
      notes: [],
    };
  }

  /**
   * For detectors analysed on the server, its real validation decides
   * usability (src/data_validator.py on the loader's internal columns).
   */
  function applyServerValidation(checks, server) {
    if (!server || !server.methods) {
      return checks;
    }
    return checks.map(function (check) {
      const method = server.methods[check.id];
      if (!method) {
        return check;
      }
      const v = method.validation || {};
      return Object.assign({}, check, {
        usable: !!method.usable,
        missingColumns: v.missing_columns || [],
        matchingRowCount: typeof v.n_rows === "number" ? v.n_rows : check.matchingRowCount,
        source: "server",
        notes: method.notices || [],
      });
    });
  }

  function buildReport(fileName, text) {
    const lines = splitLines(text);
    if (lines.length === 0) {
      throw new Error("The selected file appears to be empty.");
    }

    const header = parseHeader(lines[0]);
    const dataLines = lines.slice(1);
    const detectors = window.LCSPoc.data.getDetectors();

    const scan = scanDataLines(header, dataLines, detectors);
    const checks = detectors.map(function (d) {
      return checkDetector(d, header, scan.matchingCounts[d.id]);
    });

    return {
      fileName: fileName,
      nRows: dataLines.length,
      rowsScannedForTypeCheck: scan.scanLimit,
      columns: header,
      checks: checks,
      analysedAt: new Date().toISOString(),
      server: null,
      rowSample: Object.assign(scan.rowSample, {
        hasAnalyticalTypeColumn: scan.hasAnalyticalTypeColumn,
        hasStandardStatusColumn: scan.hasStandardStatusColumn,
        hasPrecisionStatusColumn: scan.hasPrecisionStatusColumn,
        hasAnalyteCodeColumn: scan.hasAnalyteCodeColumn,
      }),
    };
  }

  function typeLabel(rowType) {
    return window.LCSPoc.rows.TYPE_LABELS[rowType] || rowType;
  }

  function renderReport(report) {
    const summary = document.getElementById("load-summary");
    summary.hidden = false;
    summary.querySelector("[data-field='file-name']").textContent = report.fileName;
    summary.querySelector("[data-field='row-count']").textContent = report.nRows.toLocaleString();
    summary.querySelector("[data-field='column-count']").textContent = report.columns.length;
    summary.querySelector("[data-field='history-mode']").textContent = report.server
      ? (report.server.history_mode === "on" ? "On (older half of the jobs used as history)" : "Off (no history stored)")
      : "Not analysed (server unavailable)";

    const tbody = document.getElementById("detector-check-body");
    tbody.textContent = "";

    report.checks.forEach(function (check) {
      const row = document.createElement("tr");

      const nameCell = document.createElement("td");
      nameCell.textContent = check.label;
      row.appendChild(nameCell);

      const statusCell = document.createElement("td");
      const badge = document.createElement("span");
      badge.textContent = check.usable ? "Usable" : "Not usable";
      badge.className = "usable-badge " + (check.usable ? "usable-yes" : "usable-no");
      statusCell.appendChild(badge);
      row.appendChild(statusCell);

      const detailCell = document.createElement("td");
      const notes = [];
      notes.push(check.source === "server"
        ? "Validated and analysed on the server."
        : "Browser preview only (placeholder results).");
      if (check.missingColumns.length > 0) {
        notes.push("Missing columns: " + check.missingColumns.join(", ") + ".");
      }
      if (check.matchingRowCount === null) {
        notes.push("Could not check row type (no ANALYTICAL_TYPE column found).");
      } else {
        notes.push(check.matchingRowCount.toLocaleString() + " row(s) of type \"" + typeLabel(check.rowType) + "\".");
      }
      detailCell.textContent = notes.join(" ");
      row.appendChild(detailCell);

      tbody.appendChild(row);
    });
  }

  function showError(message) {
    const errorBox = document.getElementById("load-error");
    errorBox.hidden = false;
    errorBox.textContent = message;
  }

  function clearError() {
    const errorBox = document.getElementById("load-error");
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function setStatus(message) {
    const el = document.getElementById("load-status");
    el.hidden = !message;
    el.textContent = message || "";
  }

  function readAsText(file) {
    return new Promise(function (resolve, reject) {
      const reader = new FileReader();
      reader.onerror = function () {
        reject(new Error("Could not read \"" + file.name + "\". Please choose a valid, readable file."));
      };
      reader.onload = function (event) { resolve(String(event.target.result || "")); };
      reader.readAsText(file);
    });
  }

  function useHistory() {
    const checked = document.querySelector("input[name='history-mode']:checked");
    return !checked || checked.value !== "off";
  }

  let lastReport = null;

  function handleAnalyseClick() {
    clearError();
    document.getElementById("load-summary").hidden = true;
    document.getElementById("continue-button").disabled = true;
    lastReport = null;

    const input = document.getElementById("sample-file");
    if (!input.files || input.files.length === 0) {
      showError("Please choose a file first.");
      return;
    }

    const file = input.files[0];
    if (!/\.csv$/i.test(file.name)) {
      showError("Please choose a CSV export (.csv).");
      return;
    }
    const analyseButton = document.getElementById("analyse-button");
    analyseButton.disabled = true;
    setStatus("Analysing " + file.name + "… (large files can take a few seconds)");

    // A server failure doesn't block the preview -- it is reported instead.
    const serverPromise = window.LCSPoc.api.analyse(file, useHistory()).then(
      function (summary) { return { summary: summary, error: null }; },
      function (err) { return { summary: null, error: err }; }
    );

    Promise.all([readAsText(file), serverPromise]).then(function (results) {
      const server = results[1];
      const report = buildReport(file.name, results[0]);
      report.server = server.summary;
      report.checks = applyServerValidation(report.checks, server.summary);
      lastReport = report;
      renderReport(report);
      if (server.error) {
        showError("Real analysis unavailable: " + server.error.message);
      }
      document.getElementById("continue-button").disabled = false;
    }).catch(function (err) {
      showError("Could not analyse \"" + file.name + "\": " + err.message);
    }).then(function () {
      analyseButton.disabled = false;
      setStatus("");
    });
  }

  function handleContinueClick() {
    if (!lastReport) {
      return;
    }
    try {
      window.LCSPoc.reportStore.saveReport(lastReport);
      window.location.href = "summary.html";
    } catch (err) {
      showError("Could not proceed to the overview: " + err.message);
    }
  }

  function init() {
    document.getElementById("server-banner").hidden = window.LCSPoc.api.isServed();
    document.getElementById("analyse-button").addEventListener("click", handleAnalyseClick);
    document.getElementById("continue-button").addEventListener("click", handleContinueClick);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
