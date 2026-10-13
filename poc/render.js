/**
 * render.js — pure(ish) DOM-writer functions for the detail page
 * (detail.html).
 *
 * These functions only ever receive data as arguments and never decide
 * *what* to show (that's app.js + ranking.js) — they only decide *how* to
 * paint it (including loading an item's chart image via api.js). Items follow
 * src/qc_report/contract.py's QCItem shape whichever QC method produced
 * them, so nothing here branches on detector type. Everything is built with
 * createElement/textContent (never innerHTML) since values come from real
 * data files.
 */
window.LCSPoc = window.LCSPoc || {};

window.LCSPoc.render = (function () {
  "use strict";

  const STATE_COLORS = { fail: "#d03b3b", warning: "#fab219", pass: "#0ca30c", unknown: "#7a7a7a" };
  const STATE_POSITION = { FAIL: 0.85, WARNING: 0.5, PASS: 0.12 };

  let elements = null;

  function getElements() {
    if (!elements) {
      elements = {
        contextJob: document.getElementById("context-job"),
        contextInstrument: document.getElementById("context-instrument"),
        jobSelectWrap: document.getElementById("job-select-wrap"),
        jobSelect: document.getElementById("job-select"),
        controlsContainer: document.getElementById("item-controls"),
        variantTabs: document.getElementById("variant-tabs"),
        image: document.getElementById("main-image"),
        imageFallback: document.getElementById("image-fallback"),
        placeholderChart: document.getElementById("placeholder-chart"),
        panelCode: document.getElementById("panel-code"),
        panelLabel: document.getElementById("panel-label"),
        panelState: document.getElementById("panel-state"),
        panelStatusDetail: document.getElementById("panel-status-detail"),
        panelReason: document.getElementById("panel-reason"),
        infoMetrics: document.getElementById("info-metrics"),
      };
    }
    return elements;
  }

  /**
   * Maps a state to a lowercase CSS class suffix; "unknown" for anything
   * unexpected so styling never breaks.
   * @param {string} state FAIL | WARNING | PASS
   */
  function stateClass(state) {
    return ["FAIL", "WARNING", "PASS"].indexOf(state) === -1 ? "unknown" : state.toLowerCase();
  }

  function stateLabel(state) {
    return window.LCSPoc.ranking.STATE_LABELS[state] || "Unknown";
  }

  /** Job + instrument labels (already "Job unknown"/"Instrument unknown" when missing). */
  function renderContext(jobLabel, instrumentLabel) {
    const els = getElements();
    els.contextJob.textContent = jobLabel || "Job unknown";
    els.contextInstrument.textContent = instrumentLabel || "Instrument unknown";
  }

  /**
   * Job <select> -- only shown when the analysis has more than one job.
   * @param {Array<Object>} jobs job summaries {key, job, first_date, counts}
   * @param {string} activeKey
   */
  function renderJobSelect(jobs, activeKey) {
    const els = getElements();
    els.jobSelect.textContent = "";
    els.jobSelectWrap.hidden = jobs.length <= 1;
    jobs.forEach(function (job) {
      const option = document.createElement("option");
      option.value = job.key;
      const c = job.counts || {};
      option.textContent = job.job + (job.first_date ? " (" + job.first_date.slice(0, 10) + ")" : "") +
        " · " + (c.FAIL || 0) + " fail, " + (c.WARNING || 0) + " warning, " + (c.PASS || 0) + " pass";
      els.jobSelect.appendChild(option);
    });
    els.jobSelect.value = activeKey;
  }

  /**
   * One toggle button per code (e.g. ANALYTE_CODE), coloured by the worst
   * state among that code's results.
   * @param {Array<Object>} groups from ranking.groupByCode()
   * @param {string} activeCode
   */
  function renderControls(groups, activeCode) {
    const els = getElements();
    els.controlsContainer.textContent = "";
    groups.forEach(function (group) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.itemCode = group.code;
      button.setAttribute("aria-pressed", String(group.code === activeCode));
      button.className = "analyte-button state-" + stateClass(group.state);
      button.title = group.code + ": " + stateLabel(group.state) +
        (group.items.length > 1 ? " (worst of " + group.items.length + " results)" : "");
      button.textContent = group.code;
      els.controlsContainer.appendChild(button);
    });
  }

  /** aria-pressed is the single source of truth for "currently selected". */
  function setActiveControl(activeCode) {
    const els = getElements();
    els.controlsContainer.querySelectorAll("button[data-item-code]").forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.dataset.itemCode === activeCode));
    });
  }

  /**
   * Tabs for the variants of one code (e.g. the same analyte measured under
   * two schemes in one job). Hidden when there is only one.
   * @param {Object} group from ranking.groupByCode()
   * @param {string} activeItemId
   */
  function renderVariantTabs(group, activeItemId) {
    const els = getElements();
    els.variantTabs.textContent = "";
    els.variantTabs.hidden = group.items.length <= 1;
    if (group.items.length <= 1) {
      return;
    }
    group.items.forEach(function (item) {
      const tab = document.createElement("button");
      tab.type = "button";
      tab.className = "variant-tab";
      tab.dataset.itemId = item.id;
      tab.setAttribute("aria-pressed", String(item.id === activeItemId));
      const dot = document.createElement("span");
      dot.className = "state-dot state-" + stateClass(item.state);
      dot.setAttribute("aria-hidden", "true");
      tab.appendChild(dot);
      tab.appendChild(document.createTextNode((item.variant || item.code) + " · " + stateLabel(item.state)));
      els.variantTabs.appendChild(tab);
    });
  }

  /**
   * Paints the visualisation and info panel for one item. Resets
   * image/fallback visibility every time so a broken-image state from a
   * previous selection can never persist onto a fine one (or vice versa).
   * @param {Object} item QCItem-shaped; real items carry imageUrl
   */
  function renderActivePanel(item) {
    const els = getElements();

    els.panelCode.textContent = item.code;
    els.panelLabel.textContent = item.label || item.variant || "";
    els.panelState.textContent = stateLabel(item.state);
    els.panelState.className = "state-badge state-" + stateClass(item.state);
    els.panelStatusDetail.textContent = item.status_detail && item.status_detail !== "PLACEHOLDER" ? item.status_detail : "";
    els.panelReason.textContent = item.reason || "";

    renderMetrics(item.metrics || []);

    if (item.imageUrl) {
      els.placeholderChart.hidden = true;
      els.placeholderChart.textContent = "";
      loadChart(item);
    } else {
      cancelChart();
      els.image.hidden = true;
      els.imageFallback.hidden = true;
      els.image.removeAttribute("src");
      els.placeholderChart.hidden = false;
      renderPlaceholderChart(item, els.placeholderChart);
    }
  }

  // ── Chart loading ──
  // At most one chart request is in flight: selecting another item aborts
  // the previous request, and the previous item's chart is never shown next
  // to the new item's text (hidden until the new chart has arrived).
  let chartRequest = null;   // AbortController of the in-flight request
  let chartObjectUrl = null; // object URL currently shown in the <img>

  function cancelChart() {
    if (chartRequest) {
      chartRequest.abort();
      chartRequest = null;
    }
  }

  function loadChart(item) {
    const els = getElements();
    cancelChart();
    const controller = new AbortController();
    chartRequest = controller;

    els.image.hidden = true;
    els.imageFallback.hidden = false;
    els.imageFallback.textContent = "Loading chart…";
    els.image.dataset.itemCode = item.code;
    els.image.alt = item.imageAlt || (item.code + " chart");

    window.LCSPoc.api.fetchChart(item.imageUrl, controller.signal).then(function (blob) {
      if (chartRequest !== controller) {
        return; // superseded while the response was arriving
      }
      chartRequest = null;
      if (chartObjectUrl) {
        URL.revokeObjectURL(chartObjectUrl);
      }
      chartObjectUrl = URL.createObjectURL(blob);
      els.image.src = chartObjectUrl;
      els.image.hidden = false;
      els.imageFallback.hidden = true;
      els.imageFallback.textContent = "";
    }).catch(function (err) {
      if (err.name === "AbortError" || chartRequest !== controller) {
        return;
      }
      chartRequest = null;
      els.image.hidden = true;
      els.imageFallback.hidden = false;
      els.imageFallback.textContent = "Chart unavailable for " + item.code + " — showing the text summary only.";
    });
  }

  // Leaving the page cancels a pending chart request too.
  window.addEventListener("pagehide", cancelChart);

  function renderMetrics(metrics) {
    const els = getElements();
    els.infoMetrics.textContent = "";
    metrics.forEach(function (metric) {
      const dt = document.createElement("dt");
      dt.textContent = metric.label;
      const dd = document.createElement("dd");
      dd.textContent = metric.value;
      els.infoMetrics.appendChild(dt);
      els.infoMetrics.appendChild(dd);
    });
  }

  /**
   * Generated placeholder (an inline SVG "state meter") for items without a
   * real chart -- detectors not yet connected to the server.
   */
  function renderPlaceholderChart(item, container) {
    const svgNS = "http://www.w3.org/2000/svg";
    const width = 480, height = 160;
    const trackX1 = 40, trackX2 = 440, trackY = 90;

    container.textContent = "";

    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", item.code + " placeholder chart");

    function add(tag, attrs, text) {
      const el = document.createElementNS(svgNS, tag);
      Object.keys(attrs).forEach(function (k) { el.setAttribute(k, String(attrs[k])); });
      if (text) {
        el.textContent = text;
      }
      svg.appendChild(el);
      return el;
    }

    add("rect", { width: width, height: height, fill: "#fafaf8", stroke: "#d9d9d6", "stroke-width": 2 });
    add("text", { x: 20, y: 30, "font-size": 18, "font-weight": "bold", fill: "#1f2421" },
      item.code + (item.label ? " — " + item.label : ""));
    add("line", { x1: trackX1, y1: trackY, x2: trackX2, y2: trackY, stroke: "#d9d9d6", "stroke-width": 10, "stroke-linecap": "round" });
    add("text", { x: trackX1, y: trackY + 30, "font-size": 11, fill: "#5b6360" }, "Pass");
    add("text", { x: trackX2, y: trackY + 30, "font-size": 11, fill: "#5b6360", "text-anchor": "end" }, "Fail");

    const position = STATE_POSITION.hasOwnProperty(item.state) ? STATE_POSITION[item.state] : 0.5;
    add("circle", {
      cx: trackX1 + position * (trackX2 - trackX1), cy: trackY, r: 12,
      fill: STATE_COLORS[stateClass(item.state)], stroke: "#fafaf8", "stroke-width": 2,
    });
    add("text", { x: width / 2, y: height - 15, "font-size": 12, "font-style": "italic", fill: "#5b6360", "text-anchor": "middle" },
      "Placeholder: " + stateLabel(item.state) + " (this detector is not connected yet)");

    container.appendChild(svg);
  }

  /** Wired via the <img>'s onerror attribute (e.g. a corrupt PNG). */
  function handleImageError(event) {
    const els = getElements();
    const img = event.target;
    if (!img.getAttribute("src")) {
      return;
    }
    const code = img.dataset.itemCode || "this item";
    img.hidden = true;
    els.imageFallback.hidden = false;
    els.imageFallback.textContent = "Chart unavailable for " + code + " — showing the text summary only.";
  }

  return {
    stateClass: stateClass,
    stateLabel: stateLabel,
    renderContext: renderContext,
    renderJobSelect: renderJobSelect,
    renderControls: renderControls,
    setActiveControl: setActiveControl,
    renderVariantTabs: renderVariantTabs,
    renderActivePanel: renderActivePanel,
    handleImageError: handleImageError,
  };
})();
