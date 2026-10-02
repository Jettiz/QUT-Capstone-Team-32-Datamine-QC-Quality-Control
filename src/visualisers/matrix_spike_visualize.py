"""Matrix Spike (MS) drift visualiser.

Place at: src/visualisers/ms_visualise.py

Consumes already-computed outputs from matrix_spike_detector.py. It does not
run or alter anomaly detection. By default it renders one PNG for each
Scheme-Analyte-Unit series whose series-level drift is Warning or Failure.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np
import pandas as pd

GROUP_KEYS = ("SCHEME_CODE", "ANALYTE_CODE", "UNIT_CODE")
DRIFT_COLORS = {"Failure": "#d03b3b", "Warning": "#d9a300", "None": "#1a8a3d"}


def _safe_name(*values: Any) -> str:
    text = "_".join(str(v) for v in values)
    for char in '/\\:*?"<>|':
        text = text.replace(char, "-")
    return text.replace(" ", "_")


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _directional_normalise(value, target, lower, upper) -> pd.Series:
    deviation = value - target
    upper_dist = upper - target
    lower_dist = target - lower
    out = pd.Series(np.nan, index=value.index, dtype=float)
    up = (deviation >= 0) & (upper_dist > 0)
    down = (deviation < 0) & (lower_dist > 0)
    out.loc[up] = deviation.loc[up] / upper_dist.loc[up]
    out.loc[down] = deviation.loc[down] / lower_dist.loc[down]
    return out.replace([np.inf, -np.inf], np.nan)


def _prepare_history(history: pd.DataFrame, rolling_window: int = 10) -> pd.DataFrame:
    """Recreate notebook plotting scales only; no detection decisions are made."""
    hist = history.copy()
    if hist.empty:
        return hist
    hist["ANALYSED_DATE"] = pd.to_datetime(hist["ANALYSED_DATE"], errors="coerce")
    hist = hist.sort_values("ANALYSED_DATE", kind="stable").reset_index(drop=True)

    numeric = [
        "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE", "INTERNAL_MIN_VALUE",
        "INTERNAL_MAX_VALUE", "INTERNAL_MIN_WARNING_VALUE",
        "INTERNAL_MAX_WARNING_VALUE", "ROLLING_WARNING_POSITION",
    ]
    for col in numeric:
        if col in hist:
            hist[col] = pd.to_numeric(hist[col], errors="coerce")

    warning_cols = {
        "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE",
        "INTERNAL_MIN_WARNING_VALUE", "INTERNAL_MAX_WARNING_VALUE",
    }
    if warning_cols.issubset(hist.columns):
        hist["WARNING_SCALE_POSITION"] = _directional_normalise(
            hist["NUMERIC_FINAL_VALUE"], hist["INTERNAL_TARGET_VALUE"],
            hist["INTERNAL_MIN_WARNING_VALUE"], hist["INTERNAL_MAX_WARNING_VALUE"],
        )
        if "ROLLING_WARNING_POSITION" not in hist.columns:
            win = min(max(int(rolling_window), 1), len(hist))
            hist["ROLLING_WARNING_POSITION"] = hist["WARNING_SCALE_POSITION"].rolling(
                win, min_periods=max(1, win // 2)
            ).mean()

    failure_cols = {
        "NUMERIC_FINAL_VALUE", "INTERNAL_TARGET_VALUE",
        "INTERNAL_MIN_VALUE", "INTERNAL_MAX_VALUE",
    }
    if failure_cols.issubset(hist.columns):
        hist["FAILURE_SCALE_POSITION"] = _directional_normalise(
            hist["NUMERIC_FINAL_VALUE"], hist["INTERNAL_TARGET_VALUE"],
            hist["INTERNAL_MIN_VALUE"], hist["INTERNAL_MAX_VALUE"],
        )
    return hist


def plot_ms_history(
    history: pd.DataFrame,
    drift_row: Mapping[str, Any],
    output_dir: Union[str, Path],
    rolling_window: int = 10,
) -> Path:
    """Render one Scheme-Analyte-Unit history chart from detector output."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = _prepare_history(history, rolling_window)
    scheme = drift_row.get("SCHEME_CODE", "?")
    analyte = drift_row.get("ANALYTE_CODE", "?")
    unit = drift_row.get("UNIT_CODE", "?")
    level = str(drift_row.get("drift_level", "None"))

    fig, ax = plt.subplots(figsize=(10, 5))
    if not hist.empty and "WARNING_SCALE_POSITION" in hist:
        ax.plot(hist["ANALYSED_DATE"], hist["WARNING_SCALE_POSITION"], marker="o",
                markersize=3.5, linewidth=1.1, label="Warning-scale position", zorder=2)
    if not hist.empty and "ROLLING_WARNING_POSITION" in hist:
        ax.plot(hist["ANALYSED_DATE"], hist["ROLLING_WARNING_POSITION"], linewidth=2,
                label="Rolling warning position", zorder=3)

    ax.axhline(0.0, linestyle="--", linewidth=1, label="Target (0)")
    ax.axhline(1.0, linestyle=":", linewidth=1.3, label="Drift boundary (±1)")
    ax.axhline(-1.0, linestyle=":", linewidth=1.3)

    warning_onset = pd.to_datetime(drift_row.get("warning_drift_start"), errors="coerce")
    failure_onset = pd.to_datetime(drift_row.get("failure_drift_start"), errors="coerce")
    if _as_bool(drift_row.get("warning_drift_detected", False)) and pd.notna(warning_onset):
        ax.axvline(warning_onset, linestyle="--", linewidth=1.4, label="Warning drift onset")
    if _as_bool(drift_row.get("failure_drift_detected", False)) and pd.notna(failure_onset):
        ax.axvline(failure_onset, linestyle="-.", linewidth=1.6, label="Failure drift onset")

    # Highlight only the row-level confirmation observations already selected by detector.
    if not hist.empty and {"DRIFT_FLAG", "WARNING_SCALE_POSITION"}.issubset(hist.columns):
        for flag in ("Warning", "Failure"):
            pts = hist[hist["DRIFT_FLAG"].eq(flag)]
            if not pts.empty:
                ax.scatter(pts["ANALYSED_DATE"], pts["WARNING_SCALE_POSITION"], s=70,
                           color=DRIFT_COLORS[flag], edgecolors="black", linewidths=.7,
                           zorder=5, label=f"{flag} confirmation rows")

    if not hist.empty and {"IF_ANOMALY", "WARNING_SCALE_POSITION"}.issubset(hist.columns):
        pts = hist[hist["IF_ANOMALY"].map(_as_bool)]
        if not pts.empty:
            ax.scatter(pts["ANALYSED_DATE"], pts["WARNING_SCALE_POSITION"], marker="x",
                       s=55, linewidths=1.5, zorder=6, label="Isolation Forest anomaly")

    direction = (drift_row.get("failure_direction") if level == "Failure"
                 else drift_row.get("warning_direction") if level == "Warning" else "")
    status = level + (f" / {direction}" if direction and str(direction) != "None" else "")
    ax.set_title(f"{analyte} Matrix Spike drift — {scheme} / {unit}\nSeries result: {status}",
                 color=DRIFT_COLORS.get(level, "#7a7a7a"))
    ax.set_xlabel("Analysed date")
    ax.set_ylabel("Warning-scale position (target = 0, boundary = ±1)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=.2)
    fig.autofmt_xdate()
    fig.tight_layout()

    path = output_dir / f"ms_{_safe_name(scheme, analyte, unit)}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_ms_plots(
    results: pd.DataFrame,
    drift_summary: pd.DataFrame,
    output_dir: Union[str, Path],
    only_flagged: bool = True,
    rolling_window: int = 10,
) -> Dict[Tuple[str, str, str], Path]:
    """Generate one plot per detector Scheme-Analyte-Unit series."""
    if results.empty or drift_summary.empty:
        return {}
    for label, frame in (("results", results), ("drift summary", drift_summary)):
        missing = [c for c in GROUP_KEYS if c not in frame.columns]
        if missing:
            raise ValueError(f"MS visualiser: {label} missing grouping columns: {missing}")

    data = results.copy()
    data["ANALYSED_DATE"] = pd.to_datetime(data["ANALYSED_DATE"], errors="coerce")
    plots: Dict[Tuple[str, str, str], Path] = {}

    for _, row in drift_summary.iterrows():
        if only_flagged and str(row.get("drift_level", "None")) == "None":
            continue
        key = tuple(str(row[k]) for k in GROUP_KEYS)
        mask = pd.Series(True, index=data.index)
        for col, value in zip(GROUP_KEYS, key):
            mask &= data[col].astype(str).eq(value)
        history = data.loc[mask].copy()
        if history.empty:
            continue
        plots[key] = plot_ms_history(history, row.to_dict(), output_dir, rolling_window)
    return plots


def load_detector_outputs(results_path: Union[str, Path], drift_path: Union[str, Path]):
    """Load CSV files written by matrix_spike_detector.export_results()."""
    results_path, drift_path = Path(results_path), Path(drift_path)
    if not results_path.exists():
        raise FileNotFoundError(f"MS detection results not found: {results_path}")
    if not drift_path.exists():
        raise FileNotFoundError(f"MS drift results not found: {drift_path}")
    results = pd.read_csv(results_path, low_memory=False, keep_default_na=False)
    drift = pd.read_csv(drift_path, low_memory=False, keep_default_na=False)
    results["ANALYSED_DATE"] = pd.to_datetime(results["ANALYSED_DATE"], errors="coerce")
    for col in ("warning_drift_start", "failure_drift_start"):
        if col in drift:
            drift[col] = pd.to_datetime(drift[col], errors="coerce")
    return results, drift



def _setup_matplotlib(output_dir: Union[str, Path]):
    """Configure a headless-safe matplotlib backend and return pyplot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_bar_counts(
    counts: pd.Series,
    title: str,
    xlabel: str,
    output_path: Union[str, Path],
    order=None,
) -> Path:
    """Render a simple count bar chart."""
    output_path = Path(output_path)
    plt = _setup_matplotlib(output_path.parent)

    if order is not None:
        counts = counts.reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(counts.index.astype(str), counts.values)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of records")
    ax.grid(axis="y", alpha=0.2)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_final_risk_distribution(
    results: pd.DataFrame, output_dir: Union[str, Path]
) -> Path:
    """Plot the detector's final combined risk distribution."""
    output_dir = Path(output_dir)
    counts = results["FINAL_RISK"].value_counts()
    return _save_bar_counts(
        counts,
        "Matrix Spike final risk distribution",
        "Final risk",
        output_dir / "final_risk_distribution.png",
        order=["Critical", "High", "Medium", "Low", "Ignored"],
    )


def plot_rule_flag_distribution(
    results: pd.DataFrame, output_dir: Union[str, Path]
) -> Path:
    """Plot the rule-based QC result distribution."""
    output_dir = Path(output_dir)
    counts = results["RULE_FLAG"].value_counts()
    return _save_bar_counts(
        counts,
        "Matrix Spike rule-based QC results",
        "Rule flag",
        output_dir / "rule_flag_distribution.png",
        order=["Failure", "Warning", "Pass", "Ignored"],
    )


def plot_isolation_forest_summary(
    results: pd.DataFrame, output_dir: Union[str, Path]
) -> Path:
    """Plot Isolation Forest normal/anomaly counts."""
    output_dir = Path(output_dir)

    anomaly = results["IF_ANOMALY"].map(_as_bool)
    counts = pd.Series(
        {
            "Normal": int((~anomaly).sum()),
            "Anomaly": int(anomaly.sum()),
        }
    )
    return _save_bar_counts(
        counts,
        "Isolation Forest anomaly summary",
        "Isolation Forest result",
        output_dir / "isolation_forest_summary.png",
    )


def plot_detection_method_distribution(
    results: pd.DataFrame, output_dir: Union[str, Path]
) -> Path:
    """Plot which analytical mechanism(s) contributed to record flags."""
    output_dir = Path(output_dir)
    counts = results["DETECTION_METHOD"].fillna("None").value_counts()

    plt = _setup_matplotlib(output_dir)
    fig_height = max(4.8, 0.45 * len(counts) + 2)
    fig, ax = plt.subplots(figsize=(9, fig_height))

    y = np.arange(len(counts))
    bars = ax.barh(y, counts.values)
    ax.set_yticks(y)
    ax.set_yticklabels(counts.index.astype(str))
    ax.invert_yaxis()
    ax.set_title("Matrix Spike detection methods")
    ax.set_xlabel("Number of records")
    ax.set_ylabel("Detection method")
    ax.grid(axis="x", alpha=0.2)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {int(value):,}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    path = output_dir / "detection_method_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cclas_vs_final_risk(
    results: pd.DataFrame, output_dir: Union[str, Path]
) -> Path:
    """
    Compare original CCLAS STANDARD_STATUS with the detector's FINAL_RISK.

    This is especially useful for seeing Pass records elevated by drift/IF.
    """
    output_dir = Path(output_dir)
    plt = _setup_matplotlib(output_dir)

    table = pd.crosstab(results["STANDARD_STATUS"], results["FINAL_RISK"])
    risk_order = ["Critical", "High", "Medium", "Low", "Ignored"]
    table = table.reindex(columns=risk_order, fill_value=0)

    fig_height = max(5.0, 0.6 * len(table) + 2.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    left = np.zeros(len(table), dtype=float)
    y = np.arange(len(table))

    for risk in risk_order:
        values = table[risk].to_numpy()
        ax.barh(y, values, left=left, label=risk)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(table.index.astype(str))
    ax.invert_yaxis()
    ax.set_title("CCLAS status vs final detector risk")
    ax.set_xlabel("Number of records")
    ax.set_ylabel("Original CCLAS status")
    ax.legend(title="Final risk", fontsize=8)
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    path = output_dir / "cclas_vs_final_risk.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_top_anomalous_analytes(
    results: pd.DataFrame,
    output_dir: Union[str, Path],
    top_n: int = 20,
) -> Path:
    """Plot analytes with the most Critical/High/Medium detector results."""
    output_dir = Path(output_dir)
    plt = _setup_matplotlib(output_dir)

    flagged = results[
        results["FINAL_RISK"].isin(["Critical", "High", "Medium"])
    ].copy()

    counts = (
        flagged["ANALYTE_CODE"]
        .astype(str)
        .value_counts()
        .head(top_n)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, max(5.5, 0.35 * len(counts) + 2)))
    bars = ax.barh(counts.index, counts.values)

    ax.set_title(f"Top {min(top_n, len(counts))} anomalous Matrix Spike analytes")
    ax.set_xlabel("Critical / High / Medium records")
    ax.set_ylabel("Analyte")
    ax.grid(axis="x", alpha=0.2)

    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {int(value):,}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    path = output_dir / "top_anomalous_analytes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_if_score_vs_norm_dev(
    results: pd.DataFrame, output_dir: Union[str, Path]
) -> Path:
    """
    Plot Isolation Forest score against normalised target deviation.

    IF anomalies are highlighted so the ML layer can be interpreted alongside
    the QC-distance feature used by the overall MS detection workflow.
    """
    output_dir = Path(output_dir)
    plt = _setup_matplotlib(output_dir)

    data = results.copy()
    data["IF_SCORE"] = pd.to_numeric(data["IF_SCORE"], errors="coerce")
    data["NORM_DEV"] = pd.to_numeric(data["NORM_DEV"], errors="coerce")
    data = data.dropna(subset=["IF_SCORE", "NORM_DEV"])

    anomaly = data["IF_ANOMALY"].map(_as_bool)
    normal = data.loc[~anomaly]
    abnormal = data.loc[anomaly]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    if not normal.empty:
        ax.scatter(
            normal["NORM_DEV"],
            normal["IF_SCORE"],
            s=14,
            alpha=0.35,
            label="Normal",
        )

    if not abnormal.empty:
        ax.scatter(
            abnormal["NORM_DEV"],
            abnormal["IF_SCORE"],
            s=38,
            marker="x",
            linewidths=1.3,
            label="IF anomaly",
        )

    ax.axvline(-1.0, linestyle=":", linewidth=1)
    ax.axvline(1.0, linestyle=":", linewidth=1)
    ax.set_title("Isolation Forest score vs normalised deviation")
    ax.set_xlabel("Normalised deviation (warning boundary = ±1)")
    ax.set_ylabel("Isolation Forest anomaly score")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    path = output_dir / "if_score_vs_norm_dev.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cclas_pass_new_detections(
    results: pd.DataFrame, output_dir: Union[str, Path]
) -> Path:
    """
    Show what happened specifically to records CCLAS originally classified Pass.

    This is useful for evaluating the detector's additional anomaly candidates.
    """
    output_dir = Path(output_dir)

    passed = results[
        results["STANDARD_STATUS"].astype(str).str.strip().eq("Pass")
    ].copy()

    counts = passed["FINAL_RISK"].value_counts()
    return _save_bar_counts(
        counts,
        "Final risk for original CCLAS Pass records",
        "Final detector risk",
        output_dir / "cclas_pass_final_risk.png",
        order=["Critical", "High", "Medium", "Low", "Ignored"],
    )


def plot_drift_series_summary(
    drift_summary: pd.DataFrame, output_dir: Union[str, Path]
) -> Path:
    """Plot series-level historical drift classifications."""
    output_dir = Path(output_dir)
    counts = drift_summary["drift_level"].value_counts()

    return _save_bar_counts(
        counts,
        "Matrix Spike historical drift series",
        "Series drift level",
        output_dir / "drift_series_summary.png",
        order=["Failure", "Warning", "None"],
    )


def plot_detection_overview(
    results: pd.DataFrame,
    drift_summary: pd.DataFrame,
    output_dir: Union[str, Path],
) -> Path:
    """
    Render a compact report-level overview of the three MS analytical layers:
    rule-based QC, historical drift, and Isolation Forest, plus final risk.
    """
    output_dir = Path(output_dir)
    plt = _setup_matplotlib(output_dir)

    risk_order = ["Critical", "High", "Medium", "Low", "Ignored"]
    rule_order = ["Failure", "Warning", "Pass", "Ignored"]
    drift_order = ["Failure", "Warning", "None"]

    risk_counts = results["FINAL_RISK"].value_counts().reindex(
        risk_order, fill_value=0
    )
    rule_counts = results["RULE_FLAG"].value_counts().reindex(
        rule_order, fill_value=0
    )
    drift_counts = drift_summary["drift_level"].value_counts().reindex(
        drift_order, fill_value=0
    )
    if_anomaly = results["IF_ANOMALY"].map(_as_bool)
    if_counts = pd.Series(
        {"Normal": int((~if_anomaly).sum()), "Anomaly": int(if_anomaly.sum())}
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    panels = [
        (axes[0, 0], risk_counts, "Final risk", "Risk"),
        (axes[0, 1], rule_counts, "Rule-based QC", "Rule flag"),
        (axes[1, 0], if_counts, "Isolation Forest", "IF result"),
        (axes[1, 1], drift_counts, "Historical drift", "Series drift"),
    ]

    for ax, counts, title, xlabel in panels:
        bars = ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=20)

        for bar, value in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(value):,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Matrix Spike anomaly detection overview", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = output_dir / "ms_detection_overview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_overall_detection_plots(
    results: pd.DataFrame,
    drift_summary: pd.DataFrame,
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """
    Generate report-level plots for the complete MS detection workflow.

    These plots complement the per-series historical drift charts and use only
    values already computed by matrix_spike_detector.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots: Dict[str, Path] = {}

    required = {
        "FINAL_RISK",
        "RULE_FLAG",
        "IF_ANOMALY",
        "DETECTION_METHOD",
        "STANDARD_STATUS",
        "ANALYTE_CODE",
        "IF_SCORE",
        "NORM_DEV",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(
            f"MS visualiser: detector results missing overall-plot columns: {missing}"
        )

    plots["overview"] = plot_detection_overview(
        results, drift_summary, output_dir
    )
    plots["final_risk"] = plot_final_risk_distribution(results, output_dir)
    plots["rule_flags"] = plot_rule_flag_distribution(results, output_dir)
    plots["isolation_forest"] = plot_isolation_forest_summary(
        results, output_dir
    )
    plots["detection_methods"] = plot_detection_method_distribution(
        results, output_dir
    )
    plots["cclas_vs_risk"] = plot_cclas_vs_final_risk(results, output_dir)
    plots["cclas_pass"] = plot_cclas_pass_new_detections(results, output_dir)
    plots["top_analytes"] = plot_top_anomalous_analytes(results, output_dir)
    plots["if_scatter"] = plot_if_score_vs_norm_dev(results, output_dir)

    if "drift_level" in drift_summary.columns:
        plots["drift_summary"] = plot_drift_series_summary(
            drift_summary, output_dir
        )

    return plots


def main() -> None:
    """
    Run directly from VS Code/PyCharm.

    Run matrix_spike_detector first. This visualiser then consumes its existing
    CSV outputs and creates:
      - overall/: report-level detection plots
      - drift/: per-series historical drift plots
    """
    results_path = Path("ms_outputs/ms_detection_results.csv")
    drift_path = Path("ms_outputs/ms_drift_results.csv")
    visualisation_dir = Path("ms_outputs/visualisations")
    overall_dir = visualisation_dir / "overall"
    drift_dir = visualisation_dir / "drift"

    print(f"Loading MS results from: {results_path}")
    print(f"Loading MS drift summary from: {drift_path}")

    results, drift_summary = load_detector_outputs(results_path, drift_path)

    print("\nGenerating overall detection plots...")
    overall_plots = generate_overall_detection_plots(
        results,
        drift_summary,
        overall_dir,
    )

    print("Generating flagged drift-series plots...")
    drift_plots = generate_ms_plots(
        results,
        drift_summary,
        drift_dir,
        only_flagged=True,
        rolling_window=10,
    )

    print("\nMS Visualisation Summary")
    print("=" * 45)
    print(f"Detection rows:          {len(results):,}")
    print(f"Drift series:            {len(drift_summary):,}")
    print(f"Overall plots:           {len(overall_plots):,}")
    print(f"Flagged drift plots:     {len(drift_plots):,}")
    print(f"Visualisation directory: {visualisation_dir}")

    print("\nOverall detection plots:")
    for name, path in overall_plots.items():
        print(f"  {name:<20} {path}")

    print("\nDrift plots:")
    print(f"  {len(drift_plots):,} files written to {drift_dir}")


if __name__ == "__main__":
    main()
