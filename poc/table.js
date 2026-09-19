/**
 * table.js — pure DOM-writer functions for the real-row data table shown on
 * the Overview and Detail pages. Mirrors render.js's division of labor:
 * these functions never decide *what* to show (that's rows.js's
 * getTopRows()), only *how* to paint whatever they're given.
 */
window.LCSPoc = window.LCSPoc || {};

window.LCSPoc.table = (function () {
  "use strict";

  /**
   * Maps a row tag to the same lowercase CSS class suffix render.js's
   * severityClass() uses, so tag badges reuse the existing
   * .severity-badge.severity-* palette. "pass"/"unknown" are new suffixes
   * this feature adds to style.css.
   * @param {string} tag one of rows.js's TAGS
   * @returns {string}
   */
  function tagClass(tag) {
    return String(tag || "unknown").toLowerCase();
  }

  /**
   * Populates a <select> with "All" plus one <option> per known sample
   * type. Never offers "OTHER" as a choice.
   * @param {HTMLSelectElement} selectEl
   * @param {Array<string>} types rows.js's KNOWN_TYPES
   * @param {string} currentValue "ALL" or one of `types`
   */
  function renderTypeOptions(selectEl, types, currentValue) {
    selectEl.textContent = "";
    const allOption = document.createElement("option");
    allOption.value = "ALL";
    allOption.textContent = "All sample types";
    selectEl.appendChild(allOption);

    types.forEach(function (type) {
      const option = document.createElement("option");
      option.value = type;
      option.textContent = window.LCSPoc.rows.TYPE_LABELS[type] || type;
      selectEl.appendChild(option);
    });

    selectEl.value = currentValue;
  }

  /**
   * Populates a <select> with "All" plus one <option> per Detected-issue tag.
   * @param {HTMLSelectElement} selectEl
   * @param {Array<string>} tags rows.js's TAGS
   * @param {string} currentValue "ALL" or one of `tags`
   */
  function renderTagOptions(selectEl, tags, currentValue) {
    selectEl.textContent = "";
    const allOption = document.createElement("option");
    allOption.value = "ALL";
    allOption.textContent = "All";
    selectEl.appendChild(allOption);

    tags.forEach(function (tag) {
      const option = document.createElement("option");
      option.value = tag;
      option.textContent = window.LCSPoc.rows.TAG_LABELS[tag] || tag;
      selectEl.appendChild(option);
    });

    selectEl.value = currentValue;
  }

  /**
   * Populates a flat "All" + one-option-per-analyte-code <select>. Used by
   * the Overview page, where the analyte list spans whichever sample types
   * are currently in scope (rows.js's rowSample.analyteCodesAll).
   * @param {HTMLSelectElement} selectEl
   * @param {Array<string>} analyteCodes sorted, real codes only (no
   *   UNSPECIFIED_ANALYTE sentinel)
   * @param {string} currentValue "ALL" or one of `analyteCodes`
   */
  function renderAnalyteOptions(selectEl, analyteCodes, currentValue) {
    selectEl.textContent = "";
    const allOption = document.createElement("option");
    allOption.value = "ALL";
    allOption.textContent = "All analytes";
    selectEl.appendChild(allOption);

    analyteCodes.forEach(function (code) {
      const option = document.createElement("option");
      option.value = code;
      option.textContent = code;
      selectEl.appendChild(option);
    });

    selectEl.value = currentValue;
  }

  /**
   * Populates the Detail page's analyte <select> with "All", then a
   * "Currently shown" optgroup holding just the current item's analyte code,
   * then an "Others" optgroup with every other real analyte code seen for
   * the locked sample type.
   *
   * The dummy viz panel's item codes (e.g. "Cu") and real CCLAS analyte
   * codes (e.g. "CU") don't necessarily share casing, so `currentCode` is
   * matched against `allCodesForType` case-insensitively: if a real code
   * matches, THAT real code (correct casing) is used as the option's value
   * so selecting it actually filters real data, and it's excluded from
   * "Others" so it never appears twice under different casings. If no real
   * code matches at all, `currentCode` is still offered as-is (selecting it
   * then correctly shows "No rows match this filter" rather than silently
   * vanishing).
   * @param {HTMLSelectElement} selectEl
   * @param {string} currentCode the analyte code currently shown in the viz panel
   * @param {Array<string>} allCodesForType sorted real codes for the locked
   *   type (rows.js's rowSample.analyteCodesByType[lockedType])
   * @param {string} currentValue "ALL" or a code
   */
  function renderAnalyteOptionsGrouped(selectEl, currentCode, allCodesForType, currentValue) {
    selectEl.textContent = "";

    const allOption = document.createElement("option");
    allOption.value = "ALL";
    allOption.textContent = "All analytes";
    selectEl.appendChild(allOption);

    const currentCodeUpper = String(currentCode || "").toUpperCase();
    const matchedRealCode = allCodesForType.find(function (code) { return code.toUpperCase() === currentCodeUpper; });
    const currentOptionValue = matchedRealCode || currentCode;

    const currentGroup = document.createElement("optgroup");
    currentGroup.label = "Currently shown";
    const currentOption = document.createElement("option");
    currentOption.value = currentOptionValue;
    currentOption.textContent = currentOptionValue;
    currentGroup.appendChild(currentOption);
    selectEl.appendChild(currentGroup);

    const otherCodes = allCodesForType.filter(function (code) { return code.toUpperCase() !== currentCodeUpper; });
    if (otherCodes.length > 0) {
      const othersGroup = document.createElement("optgroup");
      othersGroup.label = "Others";
      otherCodes.forEach(function (code) {
        const option = document.createElement("option");
        option.value = code;
        option.textContent = code;
        othersGroup.appendChild(option);
      });
      selectEl.appendChild(othersGroup);
    }

    // If the previously-selected value no longer exists as an <option>
    // (e.g. it was "Others" and the item switch moved it into "Currently
    // shown" under a different code), fall back to "ALL" rather than
    // leaving the <select> on a stale/invalid value.
    const hasCurrentValue = Array.prototype.some.call(selectEl.options, function (opt) { return opt.value === currentValue; });
    selectEl.value = hasCurrentValue ? currentValue : "ALL";
  }

  /**
   * Renders the Detail page's fixed sample-type indicator as plain text --
   * deliberately not a <select> (even a disabled one), so it can never be
   * mistaken for a control that's just temporarily switched off.
   * @param {HTMLElement} el
   * @param {string} type one of rows.js's KNOWN_TYPES
   */
  function renderLockedTypeLabel(el, type) {
    el.textContent = window.LCSPoc.rows.TYPE_LABELS[type] || type;
  }

  /**
   * Rebuilds a <table>'s full contents (thead + tbody) from the raw column
   * list and a set of sampled row records. Built with createElement/
   * textContent throughout -- never innerHTML -- since cell values are real,
   * user-supplied file contents.
   * @param {HTMLTableElement} tableEl
   * @param {Array<string>} columns raw uppercase column names, in file order
   * @param {Array<{type: string, tag: string, fields: Array<string>}>} rowRecords
   */
  function renderRowTable(tableEl, columns, rowRecords) {
    tableEl.textContent = "";

    const thead = document.createElement("thead");
    const headRow = document.createElement("tr");
    const issueHeader = document.createElement("th");
    issueHeader.textContent = "Detected issue";
    headRow.appendChild(issueHeader);
    columns.forEach(function (col) {
      const th = document.createElement("th");
      th.textContent = col;
      headRow.appendChild(th);
    });
    thead.appendChild(headRow);
    tableEl.appendChild(thead);

    const tbody = document.createElement("tbody");

    if (rowRecords.length === 0) {
      const emptyRow = document.createElement("tr");
      emptyRow.className = "row-empty-state";
      const emptyCell = document.createElement("td");
      emptyCell.colSpan = columns.length + 1;
      emptyCell.textContent = "No rows match this filter.";
      emptyRow.appendChild(emptyCell);
      tbody.appendChild(emptyRow);
    } else {
      rowRecords.forEach(function (record) {
        const tr = document.createElement("tr");

        const issueCell = document.createElement("td");
        const badge = document.createElement("span");
        badge.className = "severity-badge severity-" + tagClass(record.tag);
        badge.textContent = window.LCSPoc.rows.TAG_LABELS[record.tag] || record.tag;
        issueCell.appendChild(badge);
        tr.appendChild(issueCell);

        columns.forEach(function (col, i) {
          const td = document.createElement("td");
          const value = record.fields[i] || "";
          td.textContent = value;
          td.title = value;
          tr.appendChild(td);
        });

        tbody.appendChild(tr);
      });
    }

    tableEl.appendChild(tbody);
  }

  /**
   * @param {HTMLElement} el
   * @param {number} shownCount
   * @param {number} totalMatching
   */
  function renderRowCountMessage(el, shownCount, totalMatching) {
    if (totalMatching === 0) {
      el.textContent = "No rows match this filter.";
    } else if (shownCount >= totalMatching) {
      el.textContent = "Showing all " + totalMatching.toLocaleString() + " matching row(s).";
    } else {
      el.textContent = "Showing " + shownCount + " of " + totalMatching.toLocaleString() + " matching row(s).";
    }
  }

  return {
    renderTypeOptions: renderTypeOptions,
    renderTagOptions: renderTagOptions,
    renderAnalyteOptions: renderAnalyteOptions,
    renderAnalyteOptionsGrouped: renderAnalyteOptionsGrouped,
    renderLockedTypeLabel: renderLockedTypeLabel,
    renderRowTable: renderRowTable,
    renderRowCountMessage: renderRowCountMessage,
  };
})();
