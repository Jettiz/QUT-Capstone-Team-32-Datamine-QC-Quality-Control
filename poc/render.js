/**
 * render.js — pure(ish) DOM-writer functions for the detail page
 * (detail.html).
 *
 * These functions only ever receive data as arguments; they never read
 * LCSPoc.data directly and never decide *what* to show (that's
 * LCSPoc.ranking's job) — they only decide *how* to paint whatever
 * they're given. This is what keeps rendering reusable regardless of
 * detector type or whether the data behind it is dummy (Phase 1) or real
 * (Phase 2).
 */
window.LCSPoc = window.LCSPoc || {};

window.LCSPoc.render = (function () {
  "use strict";

  const SEVERITY_COLORS = { critical: "#d03b3b", high: "#ec835a", medium: "#d9a300", none: "#1a8a3d", unknown: "#7a7a7a" };
  const SEVERITY_POSITION = { CRITICAL: 0.9, HIGH: 0.65, MEDIUM: 0.4, NONE: 0.1 };

  // DOM references are looked up once and cached, then only their
  // attributes/text are updated on each selection change (no innerHTML
  // rebuilding of the panel on every click).
  let elements = null;

  function getElements() {
    if (!elements) {
      elements = {
        controlsContainer: document.getElementById("item-controls"),
        image: document.getElementById("main-image"),
        imageFallback: document.getElementById("image-fallback"),
        placeholderChart: document.getElementById("placeholder-chart"),
        panelCode: document.getElementById("panel-code"),
        panelLabel: document.getElementById("panel-label"),
        panelSeverity: document.getElementById("panel-severity"),
        infoMetrics: document.getElementById("info-metrics"),
      };
    }
    return elements;
  }

  /**
   * Builds one toggle button per item inside the controls container.
   * Called once on load — the button set itself doesn't change between
   * selections, only which button is marked aria-pressed="true" does
   * (see setActiveControl).
   * @param {Array<Object>} items
   * @param {string} activeCode item.code that should start active
   */
  function renderControls(items, activeCode) {
    const els = getElements();
    els.controlsContainer.textContent = ""; // clear any placeholder content

    items.forEach(function (item) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.itemCode = item.code;
      button.setAttribute("aria-pressed", String(item.code === activeCode));
      button.className = "analyte-button severity-" + severityClass(item.severity);
      button.textContent = item.code;
      els.controlsContainer.appendChild(button);
    });
  }

  /**
   * Marks the button matching activeCode as pressed and all others as not.
   * aria-pressed is the single source of truth for "currently selected" —
   * CSS styles the active look purely off that attribute.
   * @param {string} activeCode
   */
  function setActiveControl(activeCode) {
    const els = getElements();
    const buttons = els.controlsContainer.querySelectorAll("button[data-item-code]");
    buttons.forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.dataset.itemCode === activeCode));
    });
  }

  /**
   * Renders the main visualisation (real image or generated placeholder)
   * and the anomaly info panel for one item. Resets image/fallback
   * visibility every time so a broken-image state from a previous
   * selection can never persist onto a perfectly fine one (or vice versa).
   * @param {Object} item a record shaped like those in data.js
   */
  function renderActivePanel(item) {
    const els = getElements();

    els.panelCode.textContent = item.code;
    els.panelLabel.textContent = item.label;
    els.panelSeverity.textContent = item.severity;
    els.panelSeverity.className = "severity-badge severity-" + severityClass(item.severity);

    renderMetrics(item.metrics || []);

    if (item.imagePath) {
      els.placeholderChart.hidden = true;
      els.placeholderChart.textContent = "";

      els.image.hidden = false;
      els.imageFallback.hidden = true;
      els.imageFallback.textContent = "";
      els.image.dataset.itemCode = item.code;
      els.image.alt = item.imageAlt || (item.label + " visualisation");
      els.image.src = item.imagePath;
    } else {
      els.image.hidden = true;
      els.imageFallback.hidden = true;
      els.image.removeAttribute("src");

      els.placeholderChart.hidden = false;
      renderPlaceholderChart(item, els.placeholderChart);
    }
  }

  /**
   * Rebuilds the <dl> info panel from an ordered [{label, value}] list.
   * Built with createElement/textContent (never innerHTML) since these
   * strings may eventually come from real, untrusted data.
   * @param {Array<{label: string, value: string}>} metrics
   */
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
   * Draws a simple generated placeholder chart (an inline SVG "severity
   * meter") for any item without a real dummy image (imagePath === null)
   * — every detector besides "control" so far. Built with
   * createElementNS/textContent, never innerHTML/string concatenation.
   * @param {Object} item
   * @param {HTMLElement} container element to render into (cleared first)
   */
  function renderPlaceholderChart(item, container) {
    const svgNS = "http://www.w3.org/2000/svg";
    const width = 480, height = 160;
    const trackX1 = 40, trackX2 = 440, trackY = 90;

    container.textContent = "";

    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", (item.imageAlt || (item.label + " placeholder chart")));

    const bg = document.createElementNS(svgNS, "rect");
    bg.setAttribute("width", String(width));
    bg.setAttribute("height", String(height));
    bg.setAttribute("fill", "#fafaf8");
    bg.setAttribute("stroke", "#d9d9d6");
    bg.setAttribute("stroke-width", "2");
    svg.appendChild(bg);

    const title = document.createElementNS(svgNS, "text");
    title.setAttribute("x", "20");
    title.setAttribute("y", "30");
    title.setAttribute("font-size", "18");
    title.setAttribute("font-weight", "bold");
    title.setAttribute("fill", "#1f2421");
    title.textContent = item.code + " — " + item.label;
    svg.appendChild(title);

    const track = document.createElementNS(svgNS, "line");
    track.setAttribute("x1", String(trackX1));
    track.setAttribute("y1", String(trackY));
    track.setAttribute("x2", String(trackX2));
    track.setAttribute("y2", String(trackY));
    track.setAttribute("stroke", "#d9d9d6");
    track.setAttribute("stroke-width", "10");
    track.setAttribute("stroke-linecap", "round");
    svg.appendChild(track);

    const labelLeft = document.createElementNS(svgNS, "text");
    labelLeft.setAttribute("x", String(trackX1));
    labelLeft.setAttribute("y", String(trackY + 30));
    labelLeft.setAttribute("font-size", "11");
    labelLeft.setAttribute("fill", "#8a8a8a");
    labelLeft.textContent = "Normal";
    svg.appendChild(labelLeft);

    const labelRight = document.createElementNS(svgNS, "text");
    labelRight.setAttribute("x", String(trackX2));
    labelRight.setAttribute("y", String(trackY + 30));
    labelRight.setAttribute("font-size", "11");
    labelRight.setAttribute("fill", "#8a8a8a");
    labelRight.setAttribute("text-anchor", "end");
    labelRight.textContent = "Critical";
    svg.appendChild(labelRight);

    const position = SEVERITY_POSITION.hasOwnProperty(item.severity) ? SEVERITY_POSITION[item.severity] : 0.5;
    const markerX = trackX1 + position * (trackX2 - trackX1);
    const color = SEVERITY_COLORS[severityClass(item.severity)] || SEVERITY_COLORS.unknown;

    const marker = document.createElementNS(svgNS, "circle");
    marker.setAttribute("cx", String(markerX));
    marker.setAttribute("cy", String(trackY));
    marker.setAttribute("r", "12");
    marker.setAttribute("fill", color);
    marker.setAttribute("stroke", "black");
    marker.setAttribute("stroke-width", "1");
    svg.appendChild(marker);

    const caption = document.createElementNS(svgNS, "text");
    caption.setAttribute("x", String(width / 2));
    caption.setAttribute("y", String(height - 15));
    caption.setAttribute("font-size", "12");
    caption.setAttribute("font-style", "italic");
    caption.setAttribute("fill", color);
    caption.setAttribute("text-anchor", "middle");
    caption.textContent = "Auto-generated placeholder — " + item.severity + " (mock)";
    svg.appendChild(caption);

    container.appendChild(svg);
  }

  /**
   * Wired via the <img>'s onerror attribute. Hides the broken image and
   * shows a visible, useful fallback message instead of a broken-image
   * icon or a crashed page.
   * @param {Event} event the error event from the <img> element
   */
  function handleImageError(event) {
    const els = getElements();
    const img = event.target;
    const code = img.dataset.itemCode || "this item";

    img.hidden = true;
    els.imageFallback.hidden = false;
    els.imageFallback.textContent =
      "Image unavailable for " + code + " — showing text summary only.";
  }

  /**
   * Maps a severity string to a lowercase CSS class suffix. Falls back to
   * "unknown" for any unexpected value so styling never breaks on
   * unfamiliar data.
   * @param {string} severity
   * @returns {string}
   */
  function severityClass(severity) {
    const known = ["CRITICAL", "HIGH", "MEDIUM", "NONE"];
    if (known.indexOf(severity) === -1) {
      return "unknown";
    }
    return severity.toLowerCase();
  }

  return {
    renderControls: renderControls,
    setActiveControl: setActiveControl,
    renderActivePanel: renderActivePanel,
    handleImageError: handleImageError,
  };
})();
