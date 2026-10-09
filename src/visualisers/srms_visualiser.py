from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .base_detector import AnomalyResult, DetectionResultSet
except ImportError:  # pragma: no cover - supports direct local execution.
    from base_detector import AnomalyResult, DetectionResultSet


RISK_ORDER = ["Low", "Medium", "High", "Critical", "Not_Evaluated"]
RISK_COLORS = {
    "Low": "#4C78A8",
    "Medium": "#F2CF5B",
    "High": "#F58518",
    "Critical": "#D62728",
    "Not_Evaluated": "#8C8C8C",
}
LIMIT_COLORS = {
    "PASS": "#4C78A8",
    "WARNING": "#F2CF5B",
    "FAIL": "#D62728",
    "INSUFFICIENT_HISTORY": "#8C8C8C",
    "NOT_EVALUATED": "#B0B0B0",
}
GROUP_COLUMNS = ["SRMSStandardCode", "SchemeCode", "Analyte"]


@dataclass(frozen=True)
class SRMSVisualisationResult:
    """Container returned by the SRMS visualiser for MVP integration."""

    detector_name: str = "srms"
    figures: dict[str, Any] = field(default_factory=dict)
    saved_paths: dict[str, Path] = field(default_factory=dict)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    flagged_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    explanations: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


def visualise_srms_results(
    srms_output: DetectionResultSet | pd.DataFrame | list[AnomalyResult],
    *,
    output_dir: str | Path | None = None,
    max_records: int = 10,
    max_groups: int = 3,
    close_figures: bool = False,
) -> SRMSVisualisationResult:
    """
    Build SRMS visualisations from detector outputs without re-running detection.

    Accepts the shared DetectionResultSet, a detector results DataFrame, or a list
    of AnomalyResult objects. When output_dir is supplied, PNG charts are saved
    there; otherwise matplotlib Figure objects are returned only in memory.
    """

    results, anomalies = _coerce_srms_output(srms_output)
    warnings: list[str] = []

    if results.empty:
        return SRMSVisualisationResult(
            summary=_empty_summary(),
            warnings=["No SRMS results were supplied for visualisation."],
        )

    prepared = _prepare_results(results)
    explanations = build_srms_explanations(prepared, anomalies=anomalies, max_records=max_records)
    flagged = _flagged_rows(prepared).head(max_records).copy()
    summary = build_srms_visual_summary(prepared)

    out_path = Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(out_path / ".mplconfig"))

    figures: dict[str, Any] = {}
    saved_paths: dict[str, Path] = {}
    chart_builders = [
        ("risk_distribution", plot_risk_distribution),
        ("status_by_risk", plot_status_by_risk),
        ("value_vs_target", plot_value_vs_target),
        ("deviation_over_time", plot_deviation_over_time),
        ("top_high_risk", lambda data: plot_top_high_risk(data, max_records=max_records)),
    ]

    for name, builder in chart_builders:
        try:
            fig = builder(prepared)
        except ValueError as exc:
            warnings.append(f"{name}: {exc}")
            continue
        figures[name] = fig
        if out_path is not None:
            saved_paths[name] = _save_figure(fig, out_path / f"srms_{name}.png")
        if close_figures:
            _close_figure(fig)

    for name, fig in plot_historical_group_trends(prepared, max_groups=max_groups).items():
        figures[name] = fig
        if out_path is not None:
            saved_paths[name] = _save_figure(fig, out_path / f"srms_{name}.png")
        if close_figures:
            _close_figure(fig)

    return SRMSVisualisationResult(
        figures=figures,
        saved_paths=saved_paths,
        summary=summary,
        flagged_results=flagged,
        explanations=explanations,
        warnings=warnings,
    )


def build_srms_visual_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Return compact counts by risk level, anomaly flag, and limit status."""

    if results.empty:
        return _empty_summary()

    prepared = _prepare_results(results)
    rows = []
    for risk_level in RISK_ORDER:
        risk_rows = prepared[prepared["_RiskLevel"].eq(risk_level)]
        if risk_rows.empty:
            continue
        rows.append(
            {
                "RiskLevel": risk_level,
                "Count": int(len(risk_rows)),
                "Percent": round(len(risk_rows) / len(prepared) * 100, 2),
                "FlaggedCount": int(risk_rows["_AnomalyFlag"].sum()),
                "LimitFailures": int(risk_rows["_LimitStatus"].eq("FAIL").sum()),
                "HistoricalFlags": int(risk_rows["_HistoricalFlag"].sum()),
                "ModelFlags": int(risk_rows["_ModelFlag"].sum()),
                "DriftFlags": int(risk_rows["_DriftFlag"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_srms_explanations(
    results: pd.DataFrame,
    *,
    anomalies: list[AnomalyResult] | None = None,
    max_records: int = 10,
) -> pd.DataFrame:
    """Create human-readable explanations from detector output fields only."""

    prepared = _prepare_results(results)
    anomaly_reasons = {item.record_id: item.reason for item in anomalies or []}
    rows = []

    for _, row in _flagged_rows(prepared).head(max_records).iterrows():
        record_id = str(_first_present(row, ["SampleCode", "SampleID", "ResultRow"], "unknown"))
        reason = anomaly_reasons.get(record_id) or _first_present(row, ["RiskReason", "Reason"], "")
        parts = []
        if reason:
            parts.append(str(reason))
        if row["_LimitStatus"] == "FAIL":
            parts.append("result is outside the supplied SRMS acceptance limits")
        elif row["_LimitStatus"] == "WARNING":
            parts.append("result is inside acceptance limits but inside the warning zone")
        if bool(row.get("StatisticalAnomalyFlag", False)):
            parts.append(f"historical robust Z-score is elevated ({_format_number(row.get('RobustZScore'))})")
        if bool(row.get("IForestAnomaly", row.get("IsolationForestAnomaly", False))):
            parts.append("Isolation Forest marked the result as unusual against available history")
        if bool(row.get("DriftFlag", False)):
            parts.append("recent historical trend indicates drift")
        if row.get("InsufficientHistory") is True or str(row.get("Historical_Status", "")).upper() == "INSUFFICIENT_HISTORY":
            parts.append("historical baseline is insufficient, so history-based evidence is limited")

        rows.append(
            {
                "RecordID": record_id,
                "RiskLevel": row["_RiskLevel"],
                "RiskScore": row.get("_RiskScore"),
                "RuleFlag": row.get("_RuleFlag"),
                "Explanation": "; ".join(dict.fromkeys(part for part in parts if part))
                or "No detector explanation was supplied.",
            }
        )

    return pd.DataFrame(rows, columns=["RecordID", "RiskLevel", "RiskScore", "RuleFlag", "Explanation"])


def plot_risk_distribution(results: pd.DataFrame) -> Any:
    """Bar chart showing Low/Medium/High/Critical result counts."""

    plt = _matplotlib_pyplot()
    prepared = _prepare_results(results)
    counts = prepared["_RiskLevel"].value_counts().reindex(RISK_ORDER, fill_value=0)
    counts = counts[counts.gt(0)]
    if counts.empty:
        raise ValueError("risk level fields are not available")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values, color=[RISK_COLORS.get(str(level), "#8C8C8C") for level in counts.index])
    ax.set_title("SRMS Risk Distribution")
    ax.set_xlabel("Risk level")
    ax.set_ylabel("Result count")
    ax.bar_label(ax.containers[0], padding=3)
    fig.tight_layout()
    return fig


def plot_status_by_risk(results: pd.DataFrame) -> Any:
    """Stacked bar chart comparing limit/rule status across risk levels."""

    plt = _matplotlib_pyplot()
    prepared = _prepare_results(results)
    table = pd.crosstab(prepared["_RiskLevel"], prepared["_LimitStatus"]).reindex(RISK_ORDER).dropna(how="all")
    if table.empty:
        raise ValueError("limit status fields are not available")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bottom = np.zeros(len(table))
    for status in table.columns:
        values = table[status].to_numpy()
        ax.bar(table.index, values, bottom=bottom, label=status, color=LIMIT_COLORS.get(str(status), "#8C8C8C"))
        bottom += values
    ax.set_title("SRMS Limit Status by Risk Level")
    ax.set_xlabel("Risk level")
    ax.set_ylabel("Result count")
    ax.legend(title="Limit status", fontsize=8)
    fig.tight_layout()
    return fig


def plot_value_vs_target(results: pd.DataFrame) -> Any:
    """Scatter plot of measured SRMS value against target value."""

    plt = _matplotlib_pyplot()
    prepared = _prepare_results(results)
    required = ["Value", "TargetValue"]
    if not _has_columns(prepared, required):
        raise ValueError("Value and TargetValue columns are required")
    data = prepared.dropna(subset=required)
    if data.empty:
        raise ValueError("no rows have both value and target")

    fig, ax = plt.subplots(figsize=(7, 6))
    for risk_level in RISK_ORDER:
        group = data[data["_RiskLevel"].eq(risk_level)]
        if group.empty:
            continue
        ax.scatter(
            group["TargetValue"],
            group["Value"],
            s=np.where(group["_AnomalyFlag"], 42, 24),
            color=RISK_COLORS.get(risk_level, "#8C8C8C"),
            alpha=0.8,
            label=risk_level,
        )
    low = float(np.nanmin([data["TargetValue"].min(), data["Value"].min()]))
    high = float(np.nanmax([data["TargetValue"].max(), data["Value"].max()]))
    ax.plot([low, high], [low, high], color="#222222", linestyle="--", linewidth=1, label="Value = target")
    ax.set_title("SRMS Value vs Target")
    ax.set_xlabel("Target value")
    ax.set_ylabel("Measured value")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_deviation_over_time(results: pd.DataFrame) -> Any:
    """Time series of deviation percent with anomalous points highlighted."""

    plt = _matplotlib_pyplot()
    prepared = _prepare_results(results)
    if "AnalysisDate" not in prepared.columns or "DeviationPct" not in prepared.columns:
        raise ValueError("AnalysisDate and DeviationPct columns are required")
    data = prepared.dropna(subset=["AnalysisDate", "DeviationPct"]).sort_values("AnalysisDate")
    if data.empty:
        raise ValueError("no rows have both analysis date and deviation percent")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    normal = data[~data["_AnomalyFlag"]]
    flagged = data[data["_AnomalyFlag"]]
    ax.scatter(normal["AnalysisDate"], normal["DeviationPct"], s=18, color="#4C78A8", alpha=0.65, label="Normal")
    for risk_level in RISK_ORDER:
        risk_rows = flagged[flagged["_RiskLevel"].eq(risk_level)]
        if risk_rows.empty:
            continue
        ax.scatter(
            risk_rows["AnalysisDate"],
            risk_rows["DeviationPct"],
            s=42,
            color=RISK_COLORS.get(risk_level, "#8C8C8C"),
            label=risk_level,
            zorder=3,
        )
    ax.axhline(0, color="#222222", linewidth=1, linestyle="--")
    ax.set_title("SRMS Deviation Over Time")
    ax.set_xlabel("Analysis date")
    ax.set_ylabel("Deviation from target (%)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_top_high_risk(results: pd.DataFrame, *, max_records: int = 10) -> Any:
    """Horizontal bar chart of the highest-priority SRMS results."""

    plt = _matplotlib_pyplot()
    prepared = _prepare_results(results)
    ranked = _flagged_rows(prepared).head(max_records)
    if ranked.empty:
        ranked = prepared.sort_values(["_RiskWeight", "_RiskScore"], ascending=False).head(max_records)
    if ranked.empty:
        raise ValueError("no rows are available")

    labels = ranked.apply(_row_label, axis=1)
    values = ranked["_RiskScore"].fillna(ranked["_RiskWeight"] * 25)
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.4 * len(ranked))))
    colors = [RISK_COLORS.get(str(level), "#8C8C8C") for level in ranked["_RiskLevel"]]
    ax.barh(labels.iloc[::-1], values.iloc[::-1], color=colors[::-1])
    ax.set_title("Top SRMS Results for Review")
    ax.set_xlabel("Risk score")
    fig.tight_layout()
    return fig


def plot_historical_group_trends(results: pd.DataFrame, *, max_groups: int = 3) -> dict[str, Any]:
    """Create trend charts for the highest-risk SRMS standard/scheme/analyte groups."""

    plt = _matplotlib_pyplot()
    prepared = _prepare_results(results)
    required = [column for column in GROUP_COLUMNS if column in prepared.columns]
    if "AnalysisDate" not in prepared.columns or "Value" not in prepared.columns or not required:
        return {}

    data = prepared.dropna(subset=["AnalysisDate", "Value"])
    if data.empty:
        return {}

    group_keys = (
        data.groupby(required, dropna=False)["_RiskWeight"]
        .max()
        .sort_values(ascending=False)
        .head(max_groups)
        .index
    )
    figures: dict[str, Any] = {}
    for idx, group_key in enumerate(group_keys, start=1):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        mask = np.logical_and.reduce([data[column].eq(value) for column, value in zip(required, group_key)])
        group = data.loc[mask].sort_values("AnalysisDate")
        if group.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(group["AnalysisDate"], group["Value"], color="#4C78A8", linewidth=1.3, label="Result")
        _maybe_hline(ax, group, "TargetValue", "Target", "#222222", "--")
        _maybe_hline(ax, group, "LowerLimit", "Lower limit", "#D62728", ":")
        _maybe_hline(ax, group, "UpperLimit", "Upper limit", "#D62728", ":")
        flagged = group[group["_AnomalyFlag"]]
        if not flagged.empty:
            ax.scatter(
                flagged["AnalysisDate"],
                flagged["Value"],
                s=52,
                color=[RISK_COLORS.get(str(level), "#8C8C8C") for level in flagged["_RiskLevel"]],
                zorder=3,
                label="Flagged",
            )
        title = " / ".join(str(value) for value in group_key)
        ax.set_title(f"SRMS Historical Trend: {title}")
        ax.set_xlabel("Analysis date")
        ax.set_ylabel("Measured value")
        ax.legend(fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        figures[f"historical_trend_{idx}"] = fig
    return figures


def _coerce_srms_output(
    srms_output: DetectionResultSet | pd.DataFrame | list[AnomalyResult],
) -> tuple[pd.DataFrame, list[AnomalyResult]]:
    if isinstance(srms_output, DetectionResultSet):
        return srms_output.results.copy(), list(srms_output.anomalies)
    if isinstance(srms_output, pd.DataFrame):
        return srms_output.copy(), []
    if isinstance(srms_output, list) and all(isinstance(item, AnomalyResult) for item in srms_output):
        return pd.DataFrame([_anomaly_to_row(item) for item in srms_output]), list(srms_output)
    raise TypeError("srms_output must be a DetectionResultSet, DataFrame, or list[AnomalyResult].")


def _anomaly_to_row(item: AnomalyResult) -> dict[str, Any]:
    row = dict(item.details)
    row["SampleID"] = item.record_id
    row["FinalRiskLevel"] = item.severity
    row["RiskReason"] = item.reason
    row["AnomalyDetected"] = item.anomaly_detected
    return row


def _prepare_results(results: pd.DataFrame) -> pd.DataFrame:
    prepared = results.copy()
    if "AnalysisDate" in prepared.columns:
        prepared["AnalysisDate"] = pd.to_datetime(prepared["AnalysisDate"], errors="coerce")

    prepared["_RiskLevel"] = prepared.apply(
        lambda row: str(_first_present(row, ["FinalRiskLevel", "Final_Risk", "RiskLevel", "severity"], "Not_Evaluated")),
        axis=1,
    )
    prepared["_RiskLevel"] = prepared["_RiskLevel"].where(prepared["_RiskLevel"].isin(RISK_ORDER), "Not_Evaluated")
    prepared["_RiskWeight"] = prepared["_RiskLevel"].map(
        {"Low": 1, "Medium": 2, "High": 3, "Critical": 4, "Not_Evaluated": 0}
    ).fillna(0)

    prepared["_RiskScore"] = pd.to_numeric(
        _first_existing_series(prepared, ["RiskScore", "IForestScoreNorm"]), errors="coerce"
    )
    prepared["_RuleFlag"] = _first_existing_series(prepared, ["RuleFlag", "LimitStatus", "Limit_Status"]).fillna(
        "NOT_EVALUATED"
    )
    prepared["_LimitStatus"] = _first_existing_series(prepared, ["LimitStatus", "Limit_Status", "StandardStatus"]).fillna(
        "NOT_EVALUATED"
    )
    prepared["_LimitStatus"] = prepared["_LimitStatus"].astype(str).str.upper().str.replace(" ", "_")
    prepared["_HistoricalFlag"] = (
        prepared.get("StatisticalAnomalyFlag", False).fillna(False).astype(bool)
        if isinstance(prepared.get("StatisticalAnomalyFlag"), pd.Series)
        else False
    )
    model_flag = _first_existing_series(prepared, ["IForestAnomaly", "IsolationForestAnomaly"]).fillna(False)
    prepared["_ModelFlag"] = model_flag.astype(bool)
    prepared["_DriftFlag"] = (
        prepared.get("DriftFlag", False).fillna(False).astype(bool)
        if isinstance(prepared.get("DriftFlag"), pd.Series)
        else False
    )
    anomaly_detected = (
        prepared.get("AnomalyDetected", False).fillna(False).astype(bool)
        if isinstance(prepared.get("AnomalyDetected"), pd.Series)
        else False
    )
    prepared["_AnomalyFlag"] = (
        prepared["_RiskLevel"].isin(["Medium", "High", "Critical"])
        | prepared["_LimitStatus"].isin(["FAIL", "WARNING"])
        | prepared["_HistoricalFlag"]
        | prepared["_ModelFlag"]
        | prepared["_DriftFlag"]
        | anomaly_detected
    )
    return prepared


def _flagged_rows(results: pd.DataFrame) -> pd.DataFrame:
    return results.sort_values(["_RiskWeight", "_RiskScore"], ascending=[False, False]).loc[results["_AnomalyFlag"]]


def _first_present(row: pd.Series, columns: list[str], default: Any = "") -> Any:
    for column in columns:
        if column in row.index and pd.notna(row.get(column)) and str(row.get(column)).strip() != "":
            return row.get(column)
    return default


def _first_existing_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return df[column]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    return all(column in df.columns for column in columns)


def _row_label(row: pd.Series) -> str:
    record_id = _first_present(row, ["SampleCode", "SampleID", "ResultRow"], "row")
    analyte = _first_present(row, ["Analyte", "AnalyteCode"], "")
    standard = _first_present(row, ["SRMSStandardCode", "StandardCode"], "")
    label_parts = [str(part) for part in [record_id, standard, analyte] if str(part).strip()]
    return " | ".join(label_parts)


def _maybe_hline(ax: Any, group: pd.DataFrame, column: str, label: str, color: str, linestyle: str) -> None:
    if column not in group.columns:
        return
    value = pd.to_numeric(group[column], errors="coerce").median()
    if pd.notna(value):
        ax.axhline(value, color=color, linestyle=linestyle, linewidth=1, label=label)


def _format_number(value: Any) -> str:
    try:
        if pd.isna(value):
            return "not available"
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "RiskLevel",
            "Count",
            "Percent",
            "FlaggedCount",
            "LimitFailures",
            "HistoricalFlags",
            "ModelFlags",
            "DriftFlags",
        ]
    )


def _matplotlib_pyplot() -> Any:
    import matplotlib

    if "matplotlib.pyplot" not in sys.modules and not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig: Any, path: Path) -> Path:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _close_figure(fig: Any) -> None:
    plt = _matplotlib_pyplot()
    plt.close(fig)
