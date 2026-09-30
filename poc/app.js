/**
 * app.js — thin glue/controller for the detail page (detail.html).
 *
 * Reads which detector to show from the URL (?detector=control, etc, set
 * by summary.js when a card is clicked), loads that detector's dummy
 * items, ranks them to find the initial highest-priority item, wires up a
 * single delegated click listener on the controls container, and calls
 * into render.js to paint the page. Holds the one piece of mutable state
 * this POC needs: which item is currently shown.
 *
 * Adding a new item to any detector's array in data.js only ever requires
 * editing data.js — this file, ranking.js and render.js need no changes,
 * because controls are built from the data array and clicks are handled
 * via delegation rather than one hardcoded listener per item.
 */
(function () {
  "use strict";

  const DEFAULT_DETECTOR_ID = "control";
  let currentCode = null;

  function getRequestedDetectorId() {
    const params = new URLSearchParams(window.location.search);
    return params.get("detector") || DEFAULT_DETECTOR_ID;
  }

  function selectItem(items, code) {
    const item = items.find(function (i) { return i.code === code; });
    if (!item) {
      return; // unknown code, e.g. a stray click target — ignore safely
    }
    currentCode = item.code;
    window.LCSPoc.render.renderActivePanel(item);
    window.LCSPoc.render.setActiveControl(currentCode);
  }

  function handleControlsClick(items) {
    return function (event) {
      const button = event.target.closest("button[data-item-code]");
      if (!button) {
        return;
      }
      selectItem(items, button.dataset.itemCode);
    };
  }

  function showUnknownDetectorMessage(detectorId) {
    document.getElementById("detector-title").textContent = "Unknown detector";
    document.getElementById("detector-subtitle").textContent =
      "\"" + detectorId + "\" is not a recognised detector type.";
    document.getElementById("item-controls").textContent = "";
    document.getElementById("viz-panel").hidden = true;
  }

  function showNoItemsMessage(detector) {
    document.getElementById("item-controls").textContent =
      "No anomalies detected for " + detector.label + " in this dummy dataset.";
    document.getElementById("viz-panel").hidden = true;
  }

  function init() {
    const detectorId = getRequestedDetectorId();
    const detector = window.LCSPoc.data.getDetector(detectorId);

    if (!detector) {
      showUnknownDetectorMessage(detectorId);
      return;
    }

    document.title = detector.label + " — Anomaly Viewer (POC)";
    document.getElementById("detector-title").textContent = detector.label + " Anomaly Viewer";
    document.getElementById("detector-subtitle").textContent =
      "Phase 1 proof of concept — dummy data. Showing anomalous " + detector.label.toLowerCase() +
      " items with a detected issue in this dummy dataset.";

    const items = window.LCSPoc.data.getItems(detectorId);
    if (!items || items.length === 0) {
      showNoItemsMessage(detector);
      return;
    }

    const highestPriority = window.LCSPoc.ranking.getHighestPriority(items);

    window.LCSPoc.render.renderControls(items, highestPriority.code);
    selectItem(items, highestPriority.code);

    document
      .getElementById("item-controls")
      .addEventListener("click", handleControlsClick(items));
  }

  document.addEventListener("DOMContentLoaded", init);
})();
