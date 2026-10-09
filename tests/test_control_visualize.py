"""
Tests for src/visualisers/control_visualize.py.

- plot_evaluation(): renders a real, non-empty PNG from one evaluation dict
  alone (its point-in-time history_window), including a single-point
  history, an excluded (hollow) current observation, and extreme offsets.
- generate_control_plots(): renders one PNG per evaluation of a real
  LCSDetector.detect() run, skipping PASS evaluations unless asked not to.
"""

from pathlib import Path

import pandas as pd
import yaml

from src.detectors.control_detector import LCSDetector
from src.visualisers.control_visualize import chart_file_name, generate_control_plots, plot_evaluation

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _make_row(*, analyte, job_code, analysed_date, value):
    return {
        "analytical_type": "Standard", "std_lot_code": "Sample", "std_code": "OREAS_502C",
        "scheme_code": "GE_IMS40Q12", "job_code": job_code, "analyte_code": analyte,
        "analysed_date": pd.Timestamp(analysed_date), "measured_value": value,
        "target_value": 100.0, "limit_max": 110.0, "limit_min": 90.0,
        "limit_max_warning": 106.0, "limit_min_warning": 94.0,
        "limit_max_inclusive": "Y", "limit_min_inclusive": "Y",
        "limit_max_warning_inclusive": "Y", "limit_min_warning_inclusive": "Y",
        "unit_code": "MG_KG", "standard_status": "Pass",
    }


def _detector(tmp_path: Path) -> LCSDetector:
    config = {"history_path": str(tmp_path / "history.csv"), "max_history_per_analyte": 30}
    config_path = tmp_path / "lcs_config.yaml"
    config_path.write_text(yaml.dump(config))
    return LCSDetector(str(config_path))


def _evaluations(tmp_path, values, analyte="PB"):
    rows = [
        _make_row(analyte=analyte, job_code=f"J{i}", analysed_date=f"2024-01-{i + 1:02d}", value=v)
        for i, v in enumerate(values)
    ]
    return _detector(tmp_path).detect(pd.DataFrame(rows), history=pd.DataFrame(), persist=False)


def _is_png(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1000 and path.read_bytes()[:8] == PNG_MAGIC


def test_plot_evaluation_creates_a_real_png(tmp_path):
    result = _evaluations(tmp_path, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 107.0])
    path = plot_evaluation(result.details["results"][-1], tmp_path / "out" / "chart.png")
    assert _is_png(path)


def test_plot_evaluation_handles_a_single_observation(tmp_path):
    result = _evaluations(tmp_path, [101.0])
    evaluation = result.details["results"][0]
    assert len(evaluation["history_window"]) == 1
    assert _is_png(plot_evaluation(evaluation, tmp_path / "single.png"))


def test_plot_evaluation_handles_excluded_extreme_current_point(tmp_path):
    result = _evaluations(tmp_path, [100.0, 100.5, 99.5, 130.0])  # last offset = 3.0 -> excluded
    evaluation = result.details["results"][-1]
    assert evaluation["eligible_for_history"] is False
    assert _is_png(plot_evaluation(evaluation, tmp_path / "excluded.png"))


def test_chart_file_name_is_sanitised():
    name = chart_file_name({"job_code": "J 1/2", "analyte_code": "PB", "std_code": "OREAS/905",
                            "scheme_code": "GE_IMS40Q12"})
    assert name == "control_J-1-2_PB_OREAS-905_GE_IMS40Q12.png"
    assert chart_file_name({"job_code": None, "analyte_code": "PB", "std_code": "S",
                            "scheme_code": "X"}).startswith("control_job-unknown_")


def test_generate_control_plots_only_renders_flagged_by_default(tmp_path):
    rows = [_make_row(analyte="PB", job_code="J1", analysed_date="2024-01-01", value=115.0),   # FAIL
            _make_row(analyte="CU", job_code="J1", analysed_date="2024-01-01", value=100.0)]   # PASS
    result = _detector(tmp_path).detect(pd.DataFrame(rows), history=pd.DataFrame(), persist=False)

    flagged = generate_control_plots(result, tmp_path / "flagged")
    assert [key[1] for key in flagged] == ["PB"]
    assert all(_is_png(p) for p in flagged.values())

    everything = generate_control_plots(result, tmp_path / "all", only_flagged=False)
    assert sorted(key[1] for key in everything) == ["CU", "PB"]


def test_generate_control_plots_empty_result_returns_empty_dict(tmp_path):
    result = _detector(tmp_path).detect(pd.DataFrame([
        {**_make_row(analyte="PB", job_code="J1", analysed_date="2024-01-01", value=100.0), "std_lot_code": "OTHER"}
    ]), history=pd.DataFrame(), persist=False)
    assert generate_control_plots(result, tmp_path / "none") == {}
