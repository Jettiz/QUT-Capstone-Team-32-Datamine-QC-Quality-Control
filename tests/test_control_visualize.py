"""
Tests for src/visualisers/control_visualize.py.

Covers the two entry points independently:
- plot_analyte_history(): given an already-selected group's history slice
  plus one "current" result dict, does it render a real, non-empty PNG,
  including the edge cases of empty history (no prior observations yet)
  and group identifiers that need filename sanitising.
- generate_control_plots(): given a real LCSAnomalyResult from
  LCSDetector.detect() plus its history CSV, does it correctly regroup
  results, skip/include NONE-severity groups per `only_flagged`, and
  produce one real PNG per rendered group.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.detectors.base_detector import LCSAnomalyResult
from src.detectors.control_detector import (
    LCSDetector,
    HISTORY_COLUMNS,
    _ensure_history_columns,
    _transform_lcs_values,
)
from src.visualisers.control_visualize import generate_control_plots, plot_analyte_history


GROUP = dict(analyte="PB", std_code="OREAS_905", scheme_code="GE_IMS40Q12")
OTHER_GROUP = dict(analyte="CU", std_code="OREAS_905", scheme_code="GE_IMS40Q12")


def _make_row(*, analyte, std_code, scheme_code, job_code, analysed_date, value,
              target=100.0, max_v=110.0, min_v=90.0, max_warn=106.0, min_warn=94.0):
    return {
        "ANALYTICAL_TYPE": "Standard",
        "STD_LOT_CODE": std_code,
        "STD_CODE": std_code,
        "SCHEME_CODE": scheme_code,
        "JOB_CODE": job_code,
        "ANALYTE_CODE": analyte,
        "ANALYSED_DATE": pd.Timestamp(analysed_date),
        "NUMERIC_FINAL_VALUE": value,
        "INTERNAL_TARGET_VALUE": target,
        "INTERNAL_MAX_VALUE": max_v,
        "INTERNAL_MIN_VALUE": min_v,
        "INTERNAL_MAX_WARNING_VALUE": max_warn,
        "INTERNAL_MIN_WARNING_VALUE": min_warn,
        "INTERNAL_MAX_INCLUSIVE": "Y",
        "INTERNAL_MIN_INCLUSIVE": "Y",
        "INTERNAL_MAX_WARNING_INCLUSIVE": "Y",
        "INTERNAL_MIN_WARNING_INCLUSIVE": "Y",
        "UNIT_CODE": "MG_KG",
        "STANDARD_STATUS": "Pass",
    }


def _history_df(rows: list) -> pd.DataFrame:
    """Build a HISTORY_COLUMNS-shaped, already-transformed history frame."""
    return _ensure_history_columns(_transform_lcs_values(pd.DataFrame(rows)))


def _make_detector(tmp_path: Path, history_path: Path, **overrides) -> LCSDetector:
    config = {
        "history_path": str(history_path),
        "excluded_std_codes": ["", "Sample", "TSV_BLANK"],
        "max_history_per_analyte": 30,
        "min_history_point": 3,
        "min_history_trend": 6,
        "trend_window": 10,
        "slope_eps": 1.0e-6,
        "trend_strength_min_for_escalation": 0.6,
        "trend_progress_min_for_failure_drift": 0.75,
        "offset_exclusion_bound": 2.0,
        "enabled": True,
    }
    config.update(overrides)
    config_path = tmp_path / "lcs_config.yaml"
    config_path.write_text(yaml.dump(config))
    return LCSDetector(str(config_path))


# ── plot_analyte_history: unit tests ────────────────────────────────────────

def _current_result(*, severity="CRITICAL", eligible=True, offset=1.4):
    return {
        "ANALYTE_CODE": GROUP["analyte"],
        "STD_CODE": GROUP["std_code"],
        "SCHEME_CODE": GROUP["scheme_code"],
        "ANALYSED_DATE": pd.Timestamp("2024-02-01"),
        "NUMERIC_FINAL_VALUE": 114.0,
        "OFFSET": offset,
        "N_HISTORY": 4,
        "DRIFT_STATUS": "UPPER_FAILURE",
        "DRIFT_SEVERITY": severity,
        "DRIFT_REASON": "Latest result has already breached the upper failure limit.",
        "ELIGIBLE_FOR_HISTORY": eligible,
    }


def test_plot_analyte_history_creates_a_real_png(tmp_path):
    history = _history_df([
        _make_row(**GROUP, job_code=f"J{i}", analysed_date=f"2024-01-0{i}", value=100.0 + i)
        for i in range(1, 4)
    ])

    path = plot_analyte_history(history, _current_result(), tmp_path)

    assert path.parent == tmp_path
    assert path.name == "control_PB_OREAS_905_GE_IMS40Q12.png"
    assert path.is_file()
    assert path.stat().st_size > 0  # a real rendered image, not an empty stub


def test_plot_analyte_history_handles_empty_history(tmp_path):
    # A group's very first observation: no prior history yet
    # (INSUFFICIENT_HISTORY case) -- must still render without error, just
    # without a historical line/warning band.
    empty_history = _ensure_history_columns(pd.DataFrame(columns=HISTORY_COLUMNS))

    path = plot_analyte_history(empty_history, _current_result(severity="NONE", offset=0.05), tmp_path)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_analyte_history_marks_excluded_current_observation(tmp_path):
    # An extreme offset is analysed but excluded from persisted history --
    # the chart must still render it, just flagged as excluded.
    history = _history_df([
        _make_row(**GROUP, job_code="J1", analysed_date="2024-01-01", value=100.0),
    ])

    path = plot_analyte_history(history, _current_result(eligible=False, offset=2.5), tmp_path)

    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_analyte_history_sanitises_filename(tmp_path):
    current = _current_result()
    current["STD_CODE"] = "OREAS/905"
    current["SCHEME_CODE"] = "GE IMS40Q12"

    path = plot_analyte_history(pd.DataFrame(columns=HISTORY_COLUMNS), current, tmp_path)

    assert "/" not in path.name
    assert " " not in path.name
    assert path.name == "control_PB_OREAS-905_GE_IMS40Q12.png"


# ── generate_control_plots: end-to-end via a real LCSDetector run ──────────

def _seed_history(history_path: Path) -> None:
    """
    Pre-existing history for both groups (2 normal rows each) so a new
    observation lands at n_history=3, meeting the default
    min_history_point=3 -- otherwise LCSDetector reports INSUFFICIENT_HISTORY
    (severity NONE) regardless of how extreme that new value is, since the
    history-depth check runs before the point-breach check
    (see control_detector._classify_drift).
    """
    existing = [
        _make_row(**GROUP, job_code=f"J{i}", analysed_date=f"2024-01-0{i}", value=100.0)
        for i in range(1, 3)
    ] + [
        _make_row(**OTHER_GROUP, job_code=f"J{i}", analysed_date=f"2024-01-0{i}", value=100.0)
        for i in range(1, 3)
    ]
    _history_df(existing).to_csv(history_path, index=False)


def test_generate_control_plots_only_renders_flagged_groups_by_default(tmp_path):
    history_path = tmp_path / "history.csv"
    _seed_history(history_path)
    detector = _make_detector(tmp_path, history_path)

    # PB breaches the upper failure limit (CRITICAL); CU stays normal (NONE).
    new_rows = pd.DataFrame([
        _make_row(**GROUP, job_code="J3", analysed_date="2024-01-10", value=121.0),
        _make_row(**OTHER_GROUP, job_code="J3", analysed_date="2024-01-10", value=100.0),
    ])
    result = detector.detect(new_rows)
    assert isinstance(result, LCSAnomalyResult)

    output_dir = tmp_path / "plots"
    flagged_only = generate_control_plots(result, history_path, output_dir)

    assert set(flagged_only.keys()) == {(GROUP["analyte"], GROUP["std_code"], GROUP["scheme_code"])}
    for path in flagged_only.values():
        assert path.is_file() and path.stat().st_size > 0


def test_generate_control_plots_only_flagged_false_renders_every_group(tmp_path):
    history_path = tmp_path / "history.csv"
    _seed_history(history_path)
    detector = _make_detector(tmp_path, history_path)

    new_rows = pd.DataFrame([
        _make_row(**GROUP, job_code="J3", analysed_date="2024-01-10", value=121.0),
        _make_row(**OTHER_GROUP, job_code="J3", analysed_date="2024-01-10", value=100.0),
    ])
    result = detector.detect(new_rows)

    output_dir = tmp_path / "plots"
    all_plots = generate_control_plots(result, history_path, output_dir, only_flagged=False)

    expected_keys = {
        (GROUP["analyte"], GROUP["std_code"], GROUP["scheme_code"]),
        (OTHER_GROUP["analyte"], OTHER_GROUP["std_code"], OTHER_GROUP["scheme_code"]),
    }
    assert set(all_plots.keys()) == expected_keys
    for path in all_plots.values():
        assert path.is_file() and path.stat().st_size > 0


def test_generate_control_plots_empty_results_returns_empty_dict(tmp_path):
    empty_result = LCSAnomalyResult(detected=False, confidence=0.0, severity="low",
                                     details={"reason": "no data"}, visualizable=False)

    plots = generate_control_plots(empty_result, tmp_path / "history.csv", tmp_path / "plots")

    assert plots == {}
