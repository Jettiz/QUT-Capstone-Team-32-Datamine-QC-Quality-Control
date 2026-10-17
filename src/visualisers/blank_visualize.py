
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..detectors.base_detector import LCSAnomalyResult
from ..qc_status import FAIL, PASS, WARNING

GROUP_KEYS = ("SCHEME_CODE", "ANALYTE_CODE", "UNIT_CODE")
ANOM_COLORS = {"Point and Drift Anomaly": "#d03b3b", "Point Anomaly": "#d9a300", "Drift Anomaly": "#ff7045", "No Anomaly": "#1a8a3d", "Ignored" : "#1127bb"}

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

def generate_b_plots(
        results: pd.DataFrame,
        output_dir: Union[str, Path],
) -> Dict[Tuple[str, str, str], Path]:
    if results.empty:
        return {}
    for label, frame in (("results", results)):
        missing = [c for c in GROUP_KEYS if c not in frame.columns]
        if missing:
            raise ValueError(f"Blank Visualiser: {label} missing column: {missing}")

    data = results.copy()
    data["ANALYSED_DATE"] = pd.to_datetime(data["ANALYSED_DATE"], errors="coerce")
    plots: Dict[Tuple[str, str, str], Path] = {}

    return plots

def load_b_outputs(results_path: Union[str, Path]):
    results_path = Path(results_path)

    if not results_path.exists():
        raise FileNotFoundError(f"Blamk detection results not found: {results_path}")
    results = pd.read_csv(results, low_memory=False, keep_default_na=False)
    results["ANALYSED_DATE"] = pd.to_datetime(results["ANALYSED_DATE"], errors="coerce")
    return results

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

def plot_b_final_risk(
        results: pd.DataFrame, output_dir: Union[str, Path]
) -> Path:
    output_dir = Path(output_dir)
    counts = results["FINAL_RISK"].value_counts()
    return _save_bar_counts(
        counts,
        "Blank Anomaly Disatribution",
        "Risk Flag",
        output_dir / "blank_flag_distribution.png",
        order=["No Anomaly", "Point Anomaly", "Drift Anomaly", "Point and Drift Anomaly"],
    )

def plot_b_detection_overview(
        results: pd.DataFrame,
        output_dir: Union[str, Path],
) -> Path:
    output_dir = Path(output_dir)
    plt = _setup_matplotlib(output_dir)

    r_order = ["No Anomaly", "Point Anomaly", "Drift Anomaly", "Point and Drift Anomaly"]

    r_counts = results["FINAL_RISK"].value_counts().reindex(
        r_order, fill_value=0
    )

    if_anomaly = results["ANOMALY"].map(_as_bool)

    fig, axes = plt.scatter(2, 2, figsize=(40, 10))

    panels = [
        (axes[0,0], r_counts, "Final Detection", "Detetion Type")
    ]

    for ax, counts, title,  xlabel in panels:
        bars = ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=20)

        for bar, value in zip(bar, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(value):,}",
                ha="center",
                va="botom",
                fontsize=8,
            )
    fig.suptitle("Blank Anomaly Detection Overview", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    path = output_dir / "b_anomaly_detection_overview.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def generate_b_overall_plots(
        results: pd.DataFrame,
        output_dir: Union[str, Path],
) -> Dict[str, Path]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots: Dict[str, Path] = {}

    required = {
        "STD_CODE",
        "ANALYTE_CODE",
        "ANALYSED_CODE",
        "IS_IGNORED",
        "STANDARD_STATUS",
        "NUMERIC_FINAL_VALUE",
        "ROLLING_CHANGE_WEEK",
        "ANOMALY",
        "FINAL_RISK",
    }
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(
            f"Blank Visualiser: results missing plot columns: {missing}"
        )

    plots["overview"] = plot_b_detection_overview(
        results, output_dir
    )
    plots["FINAL_RISK"] = plot_b_final_risk(
        results, output_dir
    )
    return plots

def main() -> None:

    results_path = Path("b_outputs/b_detection_results.csv")
    visualiser_dir = Path("b_outputs/b_visualisations")
    overall_dir = visualiser_dir / "overall"

    print(f"Loading Blank results from: {results_path}")

    overall_plot = generate_b_overall_plots(results, output_dir)

    print("\nMS Visualisation Summary")
    print("=" * 10)
    print(f"Detection rows:          {len(results):,}")
    print(f"Overall plots:           {len(overall_plot):,}")
    print(f"Visualisation directory: {visualisation_dir}")

if __name__ == "__main__":
    main()