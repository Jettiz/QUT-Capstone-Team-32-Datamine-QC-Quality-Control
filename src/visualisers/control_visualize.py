"""
Control (LCS) Sample Drift Visualiser

Renders one matplotlib PNG per LCS evaluation (one job x one
(analyte_code, std_code, scheme_code) group), reproducing the chart in
notebooks/LCS_drift_detection_historic.ipynb (plot_analyte_drift):

- x-axis = equal-step observation order (the same basis the trend is fitted
  on), with the real analysed dates as tick labels;
- historical offsets, with the trend window highlighted;
- the Theil-Sen trend line over the trend window;
- target (0), warning bounds and failure bounds (+/-1);
- the evaluated (current) observation, coloured by its FAIL / WARNING / PASS
  state, hollow when it was excluded from the stored history.

This module only consumes already-computed detector output: each
evaluation dict from LCSAnomalyResult.details["results"] carries its own
point-in-time `history_window`, so nothing is re-read from disk and a chart
for an older job shows exactly the history that job was judged against.
Detection logic (control_detector.py) stays fully decoupled from rendering.

Entry points:
- plot_evaluation(evaluation, output_path): one PNG for one evaluation (used
  by the POC server, which renders on demand).
- generate_control_plots(result, output_dir, only_flagged=True): one PNG per
  evaluation of a completed detect() run (CLI / notebook use).
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..detectors.base_detector import LCSAnomalyResult
from ..qc_status import FAIL, PASS, WARNING

# Colours: status hues are reserved for state (current point and the
# warning/failure boundaries); the history series is one blue hue in two
# steps (de-emphasised history vs the trend window); text and the trend line
# use ink, never a data colour. Same status hues as poc/style.css.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_HISTORY = "#86b6ef"
_TREND_WINDOW = "#2a78d6"
_STATE_COLORS = {FAIL: "#d03b3b", WARNING: "#fab219", PASS: "#0ca30c"}
_STATE_LABELS = {FAIL: "Fail", WARNING: "Warning", PASS: "Pass"}

# The LCS transform always maps target -> 0 and the failure limits -> +/-1.
_TARGET = 0.0
_FAILURE = 1.0
# Notebook's fixed y-range; widened only when a plotted point lies outside it,
# so an extreme (e.g. failing) observation is never cut off.
_BASE_Y_LIMIT = 1.3
_MAX_X_TICKS = 15


def _safe_name(*parts: Any) -> str:
    return "_".join(
        "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in str(p)) for p in parts
    )


def chart_file_name(evaluation: Dict[str, Any]) -> str:
    """control_<JOB>_<ANALYTE>_<STD_CODE>_<SCHEME>.png, unsafe characters replaced."""
    return "control_" + _safe_name(
        evaluation.get("job_code") or "job-unknown", evaluation.get("analyte_code"),
        evaluation.get("std_code"), evaluation.get("scheme_code"),
    ) + ".png"


def plot_evaluation(evaluation: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """
    Render one LCS evaluation as a PNG at `output_path` (parent folders are
    created). Returns the path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    window = evaluation.get("history_window") or []
    seq = np.array([p["sequence"] for p in window], dtype=float)
    offsets = np.array([p["offset"] for p in window], dtype=float)
    in_trend = np.array([bool(p["in_trend_window"]) for p in window])
    is_current = np.array([bool(p["is_current"]) for p in window])

    state = evaluation.get("warning_state", PASS)
    state_color = _STATE_COLORS.get(state, _MUTED)

    fig = Figure(figsize=(10, 4.8), dpi=110, facecolor=_SURFACE)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(_SURFACE)

    # Recessive frame and grid.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelcolor=_INK_SECONDARY, labelsize=8)

    # Reference lines.
    upper_warn = evaluation.get("transformed_max_warning")
    lower_warn = evaluation.get("transformed_min_warning")
    ax.axhline(_TARGET, color=_MUTED, linestyle=":", linewidth=1.2, label="Target (0)", zorder=1)
    if upper_warn is not None and pd.notna(upper_warn):
        ax.axhline(upper_warn, color=_STATE_COLORS[WARNING], linestyle="--", linewidth=1.4,
                   label="Warning limits", zorder=1)
    if lower_warn is not None and pd.notna(lower_warn):
        ax.axhline(lower_warn, color=_STATE_COLORS[WARNING], linestyle="--", linewidth=1.4, zorder=1)
    ax.axhline(_FAILURE, color=_STATE_COLORS[FAIL], linewidth=1.6, label="Failure limits (±1)", zorder=1)
    ax.axhline(-_FAILURE, color=_STATE_COLORS[FAIL], linewidth=1.6, zorder=1)

    # History: older points de-emphasised, trend window in the accent step.
    older = ~in_trend & ~is_current
    window_pts = in_trend & ~is_current
    if older.any():
        ax.scatter(seq[older], offsets[older], s=36, color=_HISTORY, edgecolors=_SURFACE,
                   linewidths=1.5, zorder=3, label="Earlier observations")
    if window_pts.any():
        n_used = int(evaluation.get("trend_n_used") or 0)
        ax.scatter(seq[window_pts], offsets[window_pts], s=44, color=_TREND_WINDOW, edgecolors=_SURFACE,
                   linewidths=1.5, zorder=4, label=f"Trend window (last {n_used})")

    # Theil-Sen trend line over the trend window (x in sequence coordinates).
    slope = evaluation.get("trend_slope")
    intercept = evaluation.get("trend_intercept")
    if (slope is not None and intercept is not None and pd.notna(slope) and pd.notna(intercept)
            and in_trend.sum() >= 2):
        xs = seq[in_trend]
        ax.plot([xs.min(), xs.max()], [intercept + slope * xs.min(), intercept + slope * xs.max()],
                color=_INK_SECONDARY, linewidth=2, solid_capstyle="round", zorder=5,
                label=f"Trend ({evaluation.get('trend_direction', 'FLAT').lower()}, "
                      f"{slope:+.3f}/result)")

    # Current (evaluated) observation.
    if is_current.any():
        eligible = bool(evaluation.get("eligible_for_history", True))
        label = f"Current result ({_STATE_LABELS.get(state, state)})"
        if eligible:
            ax.scatter(seq[is_current], offsets[is_current], s=150, color=state_color,
                       edgecolors=_SURFACE, linewidths=2, zorder=6, label=label)
        else:
            ax.scatter(seq[is_current], offsets[is_current], s=150, facecolors=_SURFACE,
                       edgecolors=state_color, linewidths=2.5, zorder=6,
                       label=label + " — excluded from history")

    # y-range: notebook's +/-1.3, widened to keep every plotted point visible.
    extent = max(_BASE_Y_LIMIT, float(np.nanmax(np.abs(offsets))) * 1.1 if len(offsets) else 0.0)
    ax.set_ylim(-extent, extent)
    if len(seq):
        ax.set_xlim(seq.min() - 0.6, seq.max() + 0.6)

    # Date tick labels on the sequence axis, thinned so long histories stay legible.
    if window:
        # Evenly spaced, always including the first and the current observation.
        idx = np.unique(np.linspace(0, len(window) - 1, min(len(window), _MAX_X_TICKS)).round().astype(int))
        ticks = [window[i] for i in idx]
        ax.set_xticks([p["sequence"] for p in ticks])
        ax.set_xticklabels([pd.Timestamp(p["analysed_date"]).strftime("%Y-%m-%d") for p in ticks],
                           rotation=45, ha="right")

    ax.set_xlabel("Observation sequence (equal steps, not calendar time; dates for reference)",
                  color=_INK_SECONDARY, fontsize=8.5)
    ax.set_ylabel("Offset (target = 0, failure limits = ±1)", color=_INK_SECONDARY, fontsize=8.5)
    ax.set_title(
        f"{evaluation.get('analyte_code')} | {evaluation.get('std_code')} | {evaluation.get('scheme_code')}"
        f" — {evaluation.get('drift_status')} ({_STATE_LABELS.get(state, state)})",
        color=_INK, fontsize=11, loc="left",
    )
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False,
                       labelcolor=_INK_SECONDARY)
    legend.set_zorder(10)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=_SURFACE)
    return output_path


def generate_control_plots(result: LCSAnomalyResult, output_dir: Union[str, Path],
                            only_flagged: bool = True) -> Dict[Tuple[Any, str, str, str], Path]:
    """
    Render one chart per evaluation of a completed LCSDetector.detect() run.

    Args:
        result: the LCSAnomalyResult returned by LCSDetector.detect().
        output_dir: directory the PNGs are written into.
        only_flagged: when True (default), skip PASS evaluations.

    Returns:
        Dict keyed by (job_code, analyte_code, std_code, scheme_code) -> PNG path.
    """
    output_dir = Path(output_dir)
    plots: Dict[Tuple[Any, str, str, str], Path] = {}
    for evaluation in result.details.get("results", []):
        if only_flagged and evaluation.get("warning_state") == PASS:
            continue
        key = (evaluation.get("job_code"), evaluation["analyte_code"],
               evaluation["std_code"], evaluation["scheme_code"])
        plots[key] = plot_evaluation(evaluation, output_dir / chart_file_name(evaluation))
    return plots
