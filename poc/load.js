/**
 * load.js — controller for the "Load a sample" page (index.html).
 *
 * Reads a REAL local CSV the user picks (File API, no backend, no upload
 * anywhere) and runs a simplified, client-side stand-in for what
 * src/data_loader.py + src/data_validator.py do on the real pipeline:
 *   - data_loader.load_qc_data() reads the file and normalises column
 *     names to uppercase (this file's parseHeader() mirrors that).
 *   - each validate_*_data(df) function in src/data_validator.py filters
 *     to that sample type's ANALYTICAL_TYPE and checks its required
 *     columns are present (this file's checkDetector() mirrors that, at
 *     header/row-count level only -- no null/dtype checks, unlike the
 *     real validators).
 * It also samples real rows (via rows.js) for the Overview/Detail pages'
 * data tables -- see scanDataLines()'s docstring for how that's kept
 * bounded regardless of file size.
 *
 * PHASE 2 INTEGRATION POINT: analyseFile()/checkDetector() below is what
 * would be replaced by an actual call into data_loader.py/data_validator.py
 * (via a real backend) if this ever stops being a client-only POC. The
 * report object shape this produces (see buildReport()) is deliberately
 * close to what those real functions return
 * (status/n_rows/missing_columns) so that swap would be small.
 *
 * Limitations, deliberately kept simple for a POC:
 * - CSV rows are split on a plain "," -- a field containing an embedded
 *   comma would misalign columns. None of the columns this check reads
 *   (ANALYTICAL_TYPE, and the required-column headers themselves) are
 *   expected to contain one in real CCLAS exports.
 * - Only the first MAX_ROWS_SCANNED data rows are scanned for the
 *   ANALYTICAL_TYPE row-count check and the row-sample table, capped at
 *   MAX_ROWS_SCANNED as a safety ceiling against a pathologically huge file
 *   freezing the browser tab -- NOT tuned down for "typical" files. A real
 *   99,999-row/16.7MB CCLAS export (data/raw/ResultSet.csv) scans in well
 *   under a second, and its sample types are NOT evenly interleaved
 *   (Duplicate rows in particular are clustered later in the file), so a
 *   cap much smaller than this would silently under-count or miss a real
 *   sample type entirely -- confirmed by hand against that exact file
 *   before landing on this value.
 * - No null-value or dtype checking (unlike the real validators) -- this
 *   only checks column presence + at least one row of the right sample type.
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
    // blank line before the real header -- skip any leading blank lines too,
    // or the header row would be misread as a single empty-string column.
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

  /**
   * Resolves the column indexes rows.js needs for row classification, or -1
   * if a given column isn't present in this file's header at all.
   * @param {Array<string>} header
   */
  function resolveColumnIndexes(header) {
    const names = window.LCSPoc.rows.COLUMN_NAMES;
    return {
      analyticalType: header.indexOf(names.analyticalType),
      standardStatus: header.indexOf(names.standardStatus),
      precisionStatus: header.indexOf(names.precisionStatus),
      analyteCode: header.indexOf(names.analyteCode),
    };
  }

  /**
   * Single pass over the file's data lines: splits each line once (rather
   * than once per detector, as a naive per-detector scan would), and in the
   * same loop both (a) counts how many rows match each detector's
   * analyticalTypeFilter, and (b) classifies + samples each row for the
   * row-data tables (see rows.js's createRowSampler() for the bounded
   * sampling strategy that keeps this small regardless of file size).
   * @param {Array<string>} header
   * @param {Array<string>} dataLines
   * @param {Array<Object>} detectors from data.js's getDetectors()
   * @returns {{matchingCounts: Object<string, number|null>, hasAnalyticalTypeColumn: boolean, hasStandardStatusColumn: boolean, hasPrecisionStatusColumn: boolean, scanLimit: number, rowSample: Object}}
   */
  function scanDataLines(header, dataLines, detectors) {
    const colIdx = resolveColumnIndexes(header);
    const hasAnalyticalTypeColumn = colIdx.analyticalType !== -1;
    const hasStandardStatusColumn = colIdx.standardStatus !== -1;
    const hasPrecisionStatusColumn = colIdx.precisionStatus !== -1;
    const hasAnalyteCodeColumn = colIdx.analyteCode !== -1;

    const matchingCounts = {};
    detectors.forEach(function (d) {
      matchingCounts[d.id] = hasAnalyticalTypeColumn ? 0 : null;
    });

    const sampler = window.LCSPoc.rows.createRowSampler(window.LCSPoc.rows.ROWS_PER_BUCKET_CAP);
    const scanLimit = Math.min(dataLines.length, MAX_ROWS_SCANNED);

    for (let i = 0; i < scanLimit; i++) {
      const fields = splitCsvLine(dataLines[i]);

      if (hasAnalyticalTypeColumn) {
        const rawType = (fields[colIdx.analyticalType] || "").toUpperCase();
        detectors.forEach(function (d) {
          if (rawType === d.analyticalTypeFilter.toUpperCase()) {
            matchingCounts[d.id] += 1;
          }
        });
      }

      const canonicalType = window.LCSPoc.rows.classifyRowType(fields[colIdx.analyticalType]);
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
      hasStandardStatusColumn: hasStandardStatusColumn,
      hasPrecisionStatusColumn: hasPrecisionStatusColumn,
      hasAnalyteCodeColumn: hasAnalyteCodeColumn,
      scanLimit: scanLimit,
      rowSample: sampler.build(),
    };
  }

  /**
   * Pure check: given a detector's required columns and its precomputed
   * matching-row count (from scanDataLines(), not re-scanned here), decides
   * usability. No file scanning happens in this function.
   * @param {Object} detector
   * @param {Array<string>} header
   * @param {number|null} matchingRowCount
   */
  function checkDetector(detector, header, matchingRowCount) {
    const missingColumns = detector.requiredColumns.filter(function (col) {
      return header.indexOf(col) === -1;
    });

    const usable = missingColumns.length === 0 && (matchingRowCount === null || matchingRowCount > 0);

    return {
      id: detector.id,
      label: detector.label,
      validatorImplemented: detector.validatorImplemented,
      analyticalTypeFilter: detector.analyticalTypeFilter,
      missingColumns: missingColumns,
      matchingRowCount: matchingRowCount,
      usable: usable,
    };
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
      rowSample: Object.assign(scan.rowSample, {
        hasAnalyticalTypeColumn: scan.hasAnalyticalTypeColumn,
        hasStandardStatusColumn: scan.hasStandardStatusColumn,
        hasPrecisionStatusColumn: scan.hasPrecisionStatusColumn,
        hasAnalyteCodeColumn: scan.hasAnalyteCodeColumn,
      }),
    };
  }

  function renderReport(report) {
    const summary = document.getElementById("load-summary");
    summary.hidden = false;
    summary.querySelector("[data-field='file-name']").textContent = report.fileName;
    summary.querySelector("[data-field='row-count']").textContent = report.nRows.toLocaleString();
    summary.querySelector("[data-field='column-count']").textContent = report.columns.length;

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
      if (!check.validatorImplemented) {
        notes.push("Backend validator not implemented yet — best-effort column check only.");
      }
      if (check.missingColumns.length > 0) {
        notes.push("Missing columns: " + check.missingColumns.join(", "));
      }
      if (check.matchingRowCount === null) {
        notes.push("Could not check row type (no ANALYTICAL_TYPE column found).");
      } else if (check.matchingRowCount === 0) {
        notes.push("No \"" + check.analyticalTypeFilter + "\" rows found.");
      } else {
        notes.push(check.matchingRowCount.toLocaleString() + " \"" + check.analyticalTypeFilter + "\" row(s) found.");
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

  let lastReport = null;

  function handleAnalyseClick() {
    clearError();
    document.getElementById("load-summary").hidden = true;
    document.getElementById("continue-button").disabled = true;
    lastReport = null;

    const input = document.getElementById("sample-file");
    if (!input.files || input.files.length === 0) {
      showError("Please choose a CSV file first.");
      return;
    }

    const file = input.files[0];
    const reader = new FileReader();

    reader.onerror = function () {
      showError("Could not read \"" + file.name + "\". Please choose a valid, readable CSV file.");
    };

    reader.onload = function (event) {
      try {
        const text = String(event.target.result || "");
        lastReport = buildReport(file.name, text);
        renderReport(lastReport);
        document.getElementById("continue-button").disabled = false;
      } catch (err) {
        showError("Could not analyse \"" + file.name + "\": " + err.message);
      }
    };

    reader.readAsText(file);
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
    document.getElementById("analyse-button").addEventListener("click", handleAnalyseClick);
    document.getElementById("continue-button").addEventListener("click", handleContinueClick);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
