"""
Tests for src/detectors/control_detector.py (LCSDetector), covering the
historic rolling-history behaviour ported from
notebooks/LCS_drift_detection_historic.ipynb plus the project-specific
offset-exclusion rule (analyse everything, only persist |offset| <= bound).

All fixtures use a symmetric target/limit setup (target=100, max=110,
min=90, warning=+/-6 from target) so offset simplifies to
(value - 100) / 10 for every row, keeping expected values easy to check
by hand.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.detectors.control_detector import (
    LCSDetector,
    HISTORY_COLUMNS,
    _ensure_history_columns,
    _is_eligible_for_history,
    _transform_lcs_values,
)
from src.data_validator import validate_lcs_data


GROUP = dict(analyte="PB", std_code="OREAS_905", scheme_code="GE_IMS40Q12")
OTHER_GROUP = dict(analyte="CU", std_code="OREAS_905", scheme_code="GE_IMS40Q12")


def _make_row(*, analyte, std_code, scheme_code, job_code, analysed_date, value,
              target=100.0, max_v=110.0, min_v=90.0, max_warn=106.0, min_warn=94.0,
              std_lot_code=None):
    return {
        "ANALYTICAL_TYPE": "Standard",
        "STD_LOT_CODE": std_lot_code or std_code,
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


def _history_csv(tmp_path: Path, rows: list) -> Path:
    """Write a synthetic pre-existing history CSV (already transformed)."""
    raw = pd.DataFrame(rows)
    transformed = _transform_lcs_values(raw)
    history = _ensure_history_columns(transformed)
    path = tmp_path / "history.csv"
    history.to_csv(path, index=False)
    return path


def _make_detector(tmp_path: Path, history_path: Path = None, **overrides) -> LCSDetector:
    config = {
        "history_path": str(history_path or (tmp_path / "history.csv")),
        "excluded_std_codes": ["", "Sample", "TSV_BLANK"],
        "max_history_per_analyte": 5,
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
    return LCSDetector(str(config_path), debug=overrides.get("debug", False))


def _read_history(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["ANALYSED_DATE"])


def _result_for(result, analyte=GROUP["analyte"]):
    matches = [r for r in result.details["results"] if r["ANALYTE_CODE"] == analyte]
    assert len(matches) == 1
    return matches[0]


# ── Scenario 1: at cap + 1 valid observation -> oldest evicted, stays at cap ──

def test_at_cap_plus_one_valid_evicts_oldest(tmp_path):
    existing = [
        _make_row(**GROUP, job_code=f"J{i}", analysed_date=f"2024-01-0{i}", value=100.0)
        for i in range(1, 6)  # 5 rows, at cap (max_history_per_analyte=5)
    ]
    history_path = _history_csv(tmp_path, existing)
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=5)

    new_row = pd.DataFrame([_make_row(**GROUP, job_code="J6", analysed_date="2024-01-06", value=101.0)])
    result = detector.detect(new_row)

    r = _result_for(result)
    assert r["N_HISTORY"] == 6  # 5 existing + the new one, evaluated before trim
    assert r["ELIGIBLE_FOR_HISTORY"] is True

    final = _read_history(history_path)
    final_group = final[final["ANALYTE_CODE"] == GROUP["analyte"]]
    assert len(final_group) == 5
    assert final_group["ANALYSED_DATE"].min() == pd.Timestamp("2024-01-02")  # 01-01 evicted
    assert final_group["ANALYSED_DATE"].max() == pd.Timestamp("2024-01-06")


# ── Scenario 2: below cap + 1 valid observation -> added, nothing evicted ──

def test_below_cap_plus_one_valid_adds_without_eviction(tmp_path):
    existing = [
        _make_row(**GROUP, job_code=f"J{i}", analysed_date=f"2024-01-0{i}", value=100.0)
        for i in range(1, 4)  # 3 rows, below cap of 5
    ]
    history_path = _history_csv(tmp_path, existing)
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=5)

    new_row = pd.DataFrame([_make_row(**GROUP, job_code="J4", analysed_date="2024-01-04", value=100.5)])
    detector.detect(new_row)

    final = _read_history(history_path)
    assert len(final) == 4


# ── Scenario 3 & 4: offset beyond +/-2 -> analysed and flagged, excluded from history ──

@pytest.mark.parametrize("value,expected_status", [(121.0, "UPPER_FAILURE"), (79.0, "LOWER_FAILURE")])
def test_extreme_offset_analysed_but_excluded(tmp_path, value, expected_status):
    existing = [
        _make_row(**GROUP, job_code=f"J{i}", analysed_date=f"2024-01-0{i}", value=100.0)
        for i in range(1, 4)
    ]
    history_path = _history_csv(tmp_path, existing)
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=5)

    new_row = pd.DataFrame([_make_row(**GROUP, job_code="J4", analysed_date="2024-01-04", value=value)])
    result = detector.detect(new_row)

    r = _result_for(result)
    assert abs(r["OFFSET"]) > 2.0
    assert r["ELIGIBLE_FOR_HISTORY"] is False
    assert r["DRIFT_STATUS"] == expected_status
    assert r["DRIFT_SEVERITY"] == "CRITICAL"
    assert result.detected is True  # still flagged in the overall result

    final = _read_history(history_path)
    assert len(final) == 3  # unchanged -- extreme observation never persisted


def test_offset_exactly_at_bound_is_eligible():
    # value=120 -> offset = (120-100)/10 = 2.0 exactly -- eligible per spec
    # value=80  -> offset = -2.0 exactly -- eligible per spec
    assert _is_eligible_for_history(2.0, bound=2.0) is True
    assert _is_eligible_for_history(-2.0, bound=2.0) is True
    assert _is_eligible_for_history(2.0001, bound=2.0) is False
    assert _is_eligible_for_history(-2.0001, bound=2.0) is False


# ── Scenario 5 & 6: independent groups -- untouched group unchanged, touched group updated ──

def test_untouched_analyte_history_unchanged_while_other_updates(tmp_path):
    pb_existing = [_make_row(**GROUP, job_code="J1", analysed_date="2024-01-01", value=100.0)]
    cu_existing = [
        _make_row(**OTHER_GROUP, job_code="J1", analysed_date="2024-01-01", value=100.0),
        _make_row(**OTHER_GROUP, job_code="J2", analysed_date="2024-01-02", value=101.0),
    ]
    history_path = _history_csv(tmp_path, pb_existing + cu_existing)
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=5)

    # Only PB gets a new observation this run; CU gets none.
    new_row = pd.DataFrame([_make_row(**GROUP, job_code="J2", analysed_date="2024-01-02", value=100.5)])
    detector.detect(new_row)

    final = _read_history(history_path)
    cu_final = final[final["ANALYTE_CODE"] == OTHER_GROUP["analyte"]].sort_values("ANALYSED_DATE")
    pb_final = final[final["ANALYTE_CODE"] == GROUP["analyte"]]

    assert len(cu_final) == 2
    assert list(cu_final["NUMERIC_FINAL_VALUE"]) == [100.0, 101.0]  # byte-for-byte unchanged
    assert len(pb_final) == 2  # PB grew by the new eligible observation


# ── Scenario 7: multiple valid new observations for one analyte in one run ──

def test_multiple_new_observations_same_run_use_pre_run_baseline(tmp_path):
    existing = [
        _make_row(**GROUP, job_code="J1", analysed_date="2024-01-01", value=100.0),
        _make_row(**GROUP, job_code="J2", analysed_date="2024-01-02", value=100.0),
    ]
    history_path = _history_csv(tmp_path, existing)
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=5)

    new_rows = pd.DataFrame([
        _make_row(**GROUP, job_code="J3", analysed_date="2024-01-03", value=100.1),
        _make_row(**GROUP, job_code="J3", analysed_date="2024-01-04", value=100.2),
        _make_row(**GROUP, job_code="J3", analysed_date="2024-01-05", value=100.3),
    ])
    result = detector.detect(new_rows)

    rows = [r for r in result.details["results"] if r["ANALYTE_CODE"] == GROUP["analyte"]]
    assert len(rows) == 3
    # Each new observation was analysed against the SAME pre-run baseline
    # (2 existing rows + itself), never against the other new rows from
    # this same run -- proven by every one seeing identical N_HISTORY.
    assert all(r["N_HISTORY"] == 3 for r in rows)

    final = _read_history(history_path)
    assert len(final) == 5  # 2 existing + 3 new, all eligible, under the cap of 5


# ── Scenario 8: debug mode does not change detection results ──

def test_debug_mode_does_not_change_results(tmp_path):
    existing = [
        _make_row(**GROUP, job_code=f"J{i}", analysed_date=f"2024-01-0{i}", value=100.0)
        for i in range(1, 4)
    ]
    new_row_data = [_make_row(**GROUP, job_code="J4", analysed_date="2024-01-04", value=103.0)]

    (tmp_path / "quiet").mkdir()
    (tmp_path / "loud").mkdir()
    history_quiet = _history_csv(tmp_path / "quiet", existing)
    history_loud = _history_csv(tmp_path / "loud", existing)

    detector_quiet = _make_detector(tmp_path / "quiet", history_path=history_quiet, max_history_per_analyte=5, debug=False)
    detector_loud = _make_detector(tmp_path / "loud", history_path=history_loud, max_history_per_analyte=5, debug=True)

    result_quiet = detector_quiet.detect(pd.DataFrame(new_row_data))
    result_loud = detector_loud.detect(pd.DataFrame(new_row_data))

    r_quiet = _result_for(result_quiet)
    r_loud = _result_for(result_loud)

    for key in ("DRIFT_STATUS", "DRIFT_SEVERITY", "OFFSET", "N_HISTORY", "ELIGIBLE_FOR_HISTORY", "TREND_DIRECTION"):
        assert r_quiet[key] == r_loud[key], f"{key} differs between debug=False and debug=True"
    assert result_quiet.detected == result_loud.detected
    assert result_quiet.severity == result_loud.severity
    assert result_quiet.confidence == result_loud.confidence


# ── Scenario 9: spot-check against the historic notebook's own worked example ──
# (design doc section K -- ZN/OREAS_905/GE_IMS40Q12, target 100, max 110, max-warning 106)

def test_matches_historic_notebook_worked_example(tmp_path):
    zn = dict(analyte="ZN", std_code="OREAS_905", scheme_code="GE_IMS40Q12")
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
             "2024-01-19", "2024-01-20"]  # 14-day gap before the 6th obs, per the worked example
    values = [100.5, 99.8, 101.2, 102.0, 101.5, 103.4, 104.1]
    existing = [
        _make_row(**zn, job_code=f"J{i}", analysed_date=d, value=v, max_warn=106.0, min_warn=94.0)
        for i, (d, v) in enumerate(zip(dates, values), start=1)
    ]
    history_path = _history_csv(tmp_path, existing)
    detector = _make_detector(
        tmp_path, history_path=history_path,
        max_history_per_analyte=30, min_history_point=3, min_history_trend=6,
        trend_window=10, trend_strength_min_for_escalation=0.6,
        trend_progress_min_for_failure_drift=0.75,
    )

    new_row = pd.DataFrame([_make_row(**zn, job_code="J8", analysed_date="2024-01-21", value=105.6,
                                       max_warn=106.0, min_warn=94.0)])
    result = detector.detect(new_row)

    r = _result_for(result, analyte="ZN")
    assert r["N_HISTORY"] == 8
    assert r["POINT_STATUS"] == "NORMAL"  # 0.56 hasn't crossed the 0.6 warning boundary yet
    assert r["TREND_DIRECTION"] == "UP"
    # The design doc's prose claims strength ~0.86 (6/7 consistent pairs, "every
    # pair non-decreasing except point 5"), but hand-checking its own value
    # table finds TWO dips, not one (0.05->-0.02 at point 2, and 0.20->0.15 at
    # point 5) -- 5 of 7 consecutive diffs share the overall UP sign, i.e.
    # 5/7 ~= 0.714. Verified by hand against calculate_trend()'s formula
    # (ported verbatim from the notebook's executed code); the prose narrative
    # undercounted its own dips, not a porting bug here.
    assert r["TREND_STRENGTH"] == pytest.approx(5 / 7, abs=0.01)
    # Similarly, the design doc's prose narrates the final status as
    # UPPER_WARNING_DRIFT/MEDIUM, but its own escalation formula -- progress_to_warning = 1 - distance_to_warning
    # / abs(warning_bound) = 1 - (0.6-0.56)/0.6 ~= 0.933 -- clears
    # TREND_PROGRESS_MIN_FOR_FAILURE_DRIFT (0.75) with full-confidence history
    # (N_HISTORY=8 >= min_history_trend=6), which escalate_to_failure in
    # classify_drift() (ported verbatim from the notebook's executed code, not
    # the prose) correctly resolves to *_FAILURE_DRIFT/HIGH. Verified this is
    # the formula's real behaviour, not a porting bug.
    assert r["DRIFT_STATUS"] == "UPPER_FAILURE_DRIFT"
    assert r["DRIFT_SEVERITY"] == "HIGH"


# ── config/lcs_config.yaml is actually wired up (no silent-default drift) ──

def test_real_config_file_keys_are_read():
    repo_root = Path(__file__).resolve().parent.parent
    config_path = repo_root / "config" / "lcs_config.yaml"
    detector = LCSDetector(str(config_path))

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    assert detector.max_history_per_analyte == raw_config["max_history_per_analyte"] == 30
    assert detector.min_history_point == raw_config["min_history_point"] == 3
    assert detector.min_history_trend == raw_config["min_history_trend"] == 6
    assert detector.offset_exclusion_bound == raw_config["offset_exclusion_bound"] == 2.0
    assert detector.history_path == Path(raw_config["history_path"])


# ── data_validator.validate_lcs_data now groups by STD_CODE, not the ──
# ── STD_LOT_CODE == "Sample" quirk ──

def test_validate_lcs_data_uses_std_code_not_sample_quirk():
    df = pd.DataFrame([
        _make_row(**GROUP, job_code="J1", analysed_date="2024-01-01", value=100.0,
                  std_lot_code="Sample"),  # STD_LOT_CODE=="Sample" but STD_CODE=="OREAS_905" -- a real standard
    ])
    report = validate_lcs_data(df)
    assert report["n_rows"] == 1
    assert report["status"] is True
