"""
Control (LCS) Sample Drift Visualiser

Renders a matplotlib PNG per Control sample analyte group, showing its
retained rolling history (up to `max_history_per_analyte` observations,
see src/detectors/control_detector.py) plotted against the target/limit/
warning reference lines, with the current/latest observation highlighted
and colour-coded by its DRIFT_SEVERITY.

This module only ever consumes already-computed detector output -- an
LCSAnomalyResult (from LCSDetector.detect()) and the persisted history CSV
it wrote -- it never runs detection itself. That mirrors the split already
used in poc/ (ranking/rendering logic kept separate from data): here,
detection logic (control_detector.py) stays fully decoupled from rendering
logic (this file).

Two entry points:
- plot_analyte_history(history, current, output_dir): renders one PNG for
  one already-selected group. The unit to test and reuse independently.
- generate_control_plots(result, history_path, output_dir): the
  orchestration entry a caller actually uses after detect() -- regroups
  result.details["results"], loads the matching history slice per group,
  and calls plot_analyte_history() once per group.

Phase 2 integration note (see poc/data.js): a future data.js-equivalent
that maps real detector output into the report's data shape would call
generate_control_plots(...) once per detection run and use the returned
Path objects as that shape's `imagePath` values.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from ..detectors.base_detector import LCSAnomalyResult
from ..detectors.control_detector import GROUP_KEYS, HISTORY_COLUMNS
from ..utils import load_csv_or_empty

# Same palette as poc/style.css and poc/images/*.svg, so a Phase 1 dummy
# chart and its real Phase 2 replacement read as the same visual language.
# Keys match DRIFT_SEVERITY exactly.
_SEVERITY_COLORS = {
    "CRITICAL": "#d03b3b",
    "HIGH": "#ec835a",
    "MEDIUM": "#d9a300",
    "NONE": "#1a8a3d",
}
_SEVERITY_LABELS = {"CRITICAL": "Critical", "HIGH": "High", "MEDIUM": "Medium", "NONE": "None"}

# The LCS transform (control_detector._transform_lcs_values) always maps
# target -> 0 and the failure limits -> +/-1 by construction, for every
# row -- these are true constants, not something to read back out of data.
_TARGET = 0.0
_UPPER_LIMIT = 1.0
_LOWER_LIMIT = -1.0


def plot_analyte_history(history: pd.DataFrame, current: Dict[str, Any],
                          output_dir: Union[str, Path]) -> Path:
    """
    Render one Control (LCS) sample drift-history chart for a single
    (ANALYTE_CODE, STD_CODE, SCHEME_CODE) group.

    Args:
        history: that group's slice of the persisted rolling history (same
            columns as control_detector.HISTORY_COLUMNS), e.g. loaded from
            LCSDetector(...).history_path and filtered to one group. May be
            empty (e.g. INSUFFICIENT_HISTORY on a group's very first
            observation). May or may not already include the current
            observation -- both are handled correctly (see `current`).
        current: one per-observation result dict from
            LCSAnomalyResult.details["results"] for the SAME group -- i.e.
            one entry produced by LCSDetector.detect(). Carries OFFSET,
            DRIFT_SEVERITY, ANALYSED_DATE, ANALYTE_CODE, STD_CODE,
            SCHEME_CODE, ELIGIBLE_FOR_HISTORY, etc. Always drawn as a
            separate highlighted marker on top of the historical series,
            so it stays visible whether or not it happens to already be
            one of the points in `history`.
        output_dir: directory the PNG is written into (created if missing).

    Returns:
        Path to the saved PNG, named
        control_<ANALYTE_CODE>_<STD_CODE>_<SCHEME_CODE>.png (path
        separators and spaces in those identifiers are sanitised to "-"/"_").
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backend forced to Agg (headless-safe, no display needed) before pyplot
    # is ever imported -- this is the first plotting path in the repo with
    # real test coverage, so it shouldn't depend on whatever GUI backend
    # matplotlib would otherwise try to auto-detect in a CI/test environment.
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = (
        history.sort_values("ANALYSED_DATE", kind="stable").reset_index(drop=True)
        if not history.empty else history
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))

    if not hist.empty:
        ax.plot(hist["ANALYSED_DATE"], hist["TRANSFORMED_VALUE"], color="#4C78A8",
                 marker="o", markersize=4, linewidth=1.4, label="Historical offset", zorder=2)
        upper_warn = hist.iloc[-1].get("TRANSFORMED_MAX_WARNING")
        lower_warn = hist.iloc[-1].get("TRANSFORMED_MIN_WARNING")
    else:
        upper_warn = lower_warn = None

    ax.axhline(_TARGET, color="#8a8a8a", linestyle="--", linewidth=1, label="Target (0)")
    ax.axhline(_UPPER_LIMIT, color="#b3b3b3", linewidth=1.4, label="Failure limit (±1)")
    ax.axhline(_LOWER_LIMIT, color="#b3b3b3", linewidth=1.4)
    if pd.notna(upper_warn):
        ax.axhline(float(upper_warn), color="#cfcfcf", linestyle=":", linewidth=1, label="Warning band")
    if pd.notna(lower_warn):
        ax.axhline(float(lower_warn), color="#cfcfcf", linestyle=":", linewidth=1)

    severity = str(current.get("DRIFT_SEVERITY", "NONE"))
    color = _SEVERITY_COLORS.get(severity, "#7a7a7a")
    eligible = current.get("ELIGIBLE_FOR_HISTORY", True)
    current_label = f"Current ({_SEVERITY_LABELS.get(severity, severity)})"
    if not eligible:
        current_label += " — excluded from history"

    ax.scatter([current.get("ANALYSED_DATE")], [current.get("OFFSET")], s=110, color=color,
               edgecolors="black", linewidths=1, zorder=5, label=current_label)

    analyte = current.get("ANALYTE_CODE", "?")
    std_code = current.get("STD_CODE", "?")
    scheme_code = current.get("SCHEME_CODE", "?")

    ax.set_title(f"{analyte} control sample drift — {std_code} / {scheme_code}")
    ax.set_xlabel("Analysed date")
    ax.set_ylabel("Offset (target = 0, limit = ±1)")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    safe_name = "_".join(
        str(value).replace("/", "-").replace(" ", "_") for value in (analyte, std_code, scheme_code)
    )
    path = output_dir / f"control_{safe_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_control_plots(result: LCSAnomalyResult, history_path: Union[str, Path],
                            output_dir: Union[str, Path],
                            only_flagged: bool = True) -> Dict[Tuple[str, str, str], Path]:
    """
    Generate one drift-history chart per Control (LCS) sample group touched
    by a completed LCSDetector.detect() run.

    This is the orchestration entry point a caller uses after calling
    detect(): it regroups result.details["results"] by
    (ANALYTE_CODE, STD_CODE, SCHEME_CODE), takes each group's most recent
    observation THIS run as `current`, loads that group's slice of the
    persisted history CSV LCSDetector already wrote to `history_path`, and
    renders one PNG per group via plot_analyte_history().

    Args:
        result: the LCSAnomalyResult returned by LCSDetector.detect().
        history_path: the same history CSV path the detector used (e.g.
            `LCSDetector(...).history_path` or config's `history_path`).
        output_dir: directory the PNGs are written into.
        only_flagged: when True (default), only render groups whose latest
            result this run has DRIFT_SEVERITY != "NONE" -- a report only
            needs a visual for anomalous analytes. Pass False to render
            every touched group regardless of severity.

    Returns:
        Dict keyed by (ANALYTE_CODE, STD_CODE, SCHEME_CODE) -> the saved
        PNG path, one entry per rendered group. Empty if `result` has no
        per-observation results at all (e.g. detected=False with no data).
    """
    rows: List[Dict[str, Any]] = result.details.get("results", [])
    if not rows:
        return {}

    by_group: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in GROUP_KEYS)
        by_group.setdefault(key, []).append(row)

    history_df = load_csv_or_empty(history_path, HISTORY_COLUMNS, parse_dates=["ANALYSED_DATE"])

    plots: Dict[Tuple[str, str, str], Path] = {}
    for group_key, group_rows in by_group.items():
        latest = max(group_rows, key=lambda r: r["ANALYSED_DATE"])
        if only_flagged and latest["DRIFT_SEVERITY"] == "NONE":
            continue

        mask = pd.Series(True, index=history_df.index)
        for column, value in zip(GROUP_KEYS, group_key):
            mask &= history_df[column].eq(value)
        group_history = history_df.loc[mask]

        plots[group_key] = plot_analyte_history(group_history, latest, output_dir)

    return plots
