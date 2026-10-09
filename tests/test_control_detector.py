"""
Tests for src/detectors/control_detector.py (LCSDetector), covering the
historic behaviour of notebooks/LCS_drift_detection_historic.ipynb
(job-by-job processing, de-duplication, trim-before-evaluate, trend and
classification) plus the project-specific additions: the offset-exclusion
rule, the point-in-time guard, limit checks with too little history, and the
FAIL / WARNING / PASS mapping.

Input frames use the INTERNAL column names produced by
src/data_loader.load_qc_data(). All fixtures use a symmetric target/limit
setup (target=100, max=110, min=90, warning=+/-6 from target) so offset
simplifies to (value - 100) / 10 for every row.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.detectors.control_detector import (
    HISTORY_COLUMNS,
    LCSDetector,
    WARNING_STATE_BY_DRIFT_STATUS,
    _ensure_history_columns,
    _is_eligible_for_history,
    _transform_lcs_values,
    lcs_warning_state,
)
from src.qc_status import FAIL, PASS, WARNING

GROUP = dict(analyte="PB", std_code="OREAS_502C", scheme_code="GE_IMS40Q12")
OTHER_GROUP = dict(analyte="CU", std_code="OREAS_502C", scheme_code="GE_IMS40Q12")


def _make_row(*, analyte, std_code, scheme_code, job_code, analysed_date, value,
              target=100.0, max_v=110.0, min_v=90.0, max_warn=106.0, min_warn=94.0,
              std_lot_code="Sample", analytical_type="Standard"):
    return {
        "analytical_type": analytical_type,
        "std_lot_code": std_lot_code,
        "std_code": std_code,
        "scheme_code": scheme_code,
        "job_code": job_code,
        "analyte_code": analyte,
        "analysed_date": pd.Timestamp(analysed_date),
        "measured_value": value,
        "target_value": target,
        "limit_max": max_v,
        "limit_min": min_v,
        "limit_max_warning": max_warn,
        "limit_min_warning": min_warn,
        "limit_max_inclusive": "Y",
        "limit_min_inclusive": "Y",
        "limit_max_warning_inclusive": "Y",
        "limit_min_warning_inclusive": "Y",
        "unit_code": "MG_KG",
        "standard_status": "Pass",
    }


def _history_csv(tmp_path: Path, rows: list) -> Path:
    """Write a synthetic pre-existing history CSV (already transformed)."""
    history = _ensure_history_columns(_transform_lcs_values(pd.DataFrame(rows)))
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
    return pd.read_csv(path, parse_dates=["analysed_date"])


def _result_for(result, analyte=GROUP["analyte"]):
    matches = [r for r in result.details["results"] if r["analyte_code"] == analyte]
    assert len(matches) == 1
    return matches[0]


def _rows(group, values, start_job=1, day0=1):
    return [
        _make_row(**group, job_code=f"J{start_job + i}", analysed_date=f"2024-01-{day0 + i:02d}", value=v)
        for i, v in enumerate(values)
    ]


# ── Cap: history trimmed BEFORE the evaluation (notebook) ──

def test_at_cap_plus_one_valid_evicts_oldest(tmp_path):
    history_path = _history_csv(tmp_path, _rows(GROUP, [100.0] * 5))  # at cap of 5
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=5)

    result = detector.detect(pd.DataFrame(_rows(GROUP, [101.0], start_job=6, day0=6)))

    r = _result_for(result)
    assert r["n_history"] == 5  # trimmed to the cap before evaluating, as in the notebook
    assert r["eligible_for_history"] is True

    final = _read_history(history_path)
    assert len(final) == 5
    assert final["analysed_date"].min() == pd.Timestamp("2024-01-02")  # 01-01 evicted
    assert final["analysed_date"].max() == pd.Timestamp("2024-01-06")


def test_below_cap_plus_one_valid_adds_without_eviction(tmp_path):
    history_path = _history_csv(tmp_path, _rows(GROUP, [100.0] * 3))
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=5)

    detector.detect(pd.DataFrame(_rows(GROUP, [100.5], start_job=4, day0=4)))

    assert len(_read_history(history_path)) == 4


# ── Offset-exclusion rule (src addition) ──

@pytest.mark.parametrize("value,expected_status", [(121.0, "UPPER_FAILURE"), (79.0, "LOWER_FAILURE")])
def test_extreme_offset_analysed_but_excluded(tmp_path, value, expected_status):
    history_path = _history_csv(tmp_path, _rows(GROUP, [100.0] * 3))
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=5)

    result = detector.detect(pd.DataFrame(_rows(GROUP, [value], start_job=4, day0=4)))

    r = _result_for(result)
    assert abs(r["offset"]) > 2.0
    assert r["eligible_for_history"] is False
    assert r["drift_status"] == expected_status
    assert r["warning_state"] == FAIL
    assert result.detected is True
    assert result.severity == "critical"

    assert len(_read_history(history_path)) == 3  # unchanged -- extreme observation never stored


def test_offset_exactly_at_bound_is_eligible():
    assert _is_eligible_for_history(2.0, bound=2.0) is True
    assert _is_eligible_for_history(-2.0, bound=2.0) is True
    assert _is_eligible_for_history(2.0001, bound=2.0) is False
    assert _is_eligible_for_history(-2.0001, bound=2.0) is False


# ── Groups are independent ──

def test_untouched_analyte_history_unchanged_while_other_updates(tmp_path):
    pb_existing = _rows(GROUP, [100.0])
    cu_existing = _rows(OTHER_GROUP, [100.0, 101.0])
    history_path = _history_csv(tmp_path, pb_existing + cu_existing)
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=5)

    detector.detect(pd.DataFrame(_rows(GROUP, [100.5], start_job=2, day0=2)))

    final = _read_history(history_path)
    cu_final = final[final["analyte_code"] == OTHER_GROUP["analyte"]].sort_values("analysed_date")
    assert list(cu_final["measured_value"]) == [100.0, 101.0]
    assert len(final[final["analyte_code"] == GROUP["analyte"]]) == 2


# ── Notebook semantics: de-duplication and job-by-job evaluation ──

def test_rerunning_the_same_input_is_a_no_op_for_history(tmp_path):
    history_path = tmp_path / "history.csv"
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=30)
    batch = pd.DataFrame(_rows(GROUP, [100.0, 101.0, 102.0, 103.0]))

    first = detector.detect(batch)
    after_first = _read_history(history_path)
    second = detector.detect(batch)
    after_second = _read_history(history_path)

    assert len(after_first) == len(after_second) == 4
    for a, b in zip(first.details["results"], second.details["results"]):
        assert (a["drift_status"], a["n_history"], a["offset"]) == (b["drift_status"], b["n_history"], b["offset"])


def test_one_evaluation_per_job_and_group_on_its_latest_row(tmp_path):
    history_path = _history_csv(tmp_path, _rows(GROUP, [100.0, 100.0]))
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=5)

    new_rows = pd.DataFrame([
        _make_row(**GROUP, job_code="J3", analysed_date="2024-01-03", value=100.1),
        _make_row(**GROUP, job_code="J3", analysed_date="2024-01-05", value=100.3),
        _make_row(**GROUP, job_code="J3", analysed_date="2024-01-04", value=100.2),
    ])
    result = detector.detect(new_rows)

    r = _result_for(result)  # exactly one evaluation for the job
    assert r["n_history"] == 5  # 2 stored + all 3 rows of the job
    assert r["analysed_date"] == pd.Timestamp("2024-01-05")
    assert r["measured_value"] == pytest.approx(100.3)
    assert len(_read_history(history_path)) == 5


def test_later_jobs_in_one_file_see_earlier_jobs(tmp_path):
    detector = _make_detector(tmp_path, max_history_per_analyte=30)
    result = detector.detect(pd.DataFrame(_rows(GROUP, [100.0, 100.5, 101.0, 101.5])),
                             history=pd.DataFrame(), persist=False)

    assert [r["job_code"] for r in result.details["results"]] == ["J1", "J2", "J3", "J4"]
    assert [r["n_history"] for r in result.details["results"]] == [1, 2, 3, 4]
    assert result.details["job_order"] == ["J1", "J2", "J3", "J4"]


def test_point_in_time_guard_ignores_later_stored_rows(tmp_path):
    # Stored history already contains a row dated AFTER the incoming job.
    stored = _rows(GROUP, [100.0, 100.0, 100.0]) + [
        _make_row(**GROUP, job_code="J9", analysed_date="2024-02-01", value=105.0)
    ]
    history_path = _history_csv(tmp_path, stored)
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=30)

    result = detector.detect(pd.DataFrame(_rows(GROUP, [101.0], start_job=4, day0=10)))

    r = _result_for(result)
    assert r["analysed_date"] == pd.Timestamp("2024-01-10")  # the job's own observation
    assert r["n_history"] == 4  # 3 earlier stored rows + itself; the Feb row is ignored
    assert all(p["analysed_date"] <= pd.Timestamp("2024-01-10") for p in r["history_window"])


# ── Too little history: limits are still checked ──

@pytest.mark.parametrize("value,status,state", [
    (115.0, "UPPER_FAILURE", FAIL),
    (107.0, "UPPER_WARNING", WARNING),
    (101.0, "INSUFFICIENT_HISTORY", PASS),
])
def test_insufficient_history_still_checks_limits(tmp_path, value, status, state):
    detector = _make_detector(tmp_path)
    result = detector.detect(pd.DataFrame(_rows(GROUP, [value])), history=pd.DataFrame(), persist=False)

    r = _result_for(result)
    assert r["n_history"] == 1
    assert r["drift_status"] == status
    assert r["warning_state"] == state
    if state != PASS:
        assert "Trend not assessed" in r["drift_reason"]
    assert r["data_quality_flag"].startswith("insufficient history")


# ── Warning states ──

def test_every_drift_status_maps_to_one_of_three_states():
    assert set(WARNING_STATE_BY_DRIFT_STATUS.values()) == {FAIL, WARNING, PASS}
    assert lcs_warning_state("UPPER_FAILURE") == FAIL
    assert lcs_warning_state("LOWER_FAILURE") == FAIL
    for status in ("UPPER_WARNING", "LOWER_WARNING", "UPPER_WARNING_DRIFT", "LOWER_WARNING_DRIFT",
                   "UPPER_FAILURE_DRIFT", "LOWER_FAILURE_DRIFT"):
        assert lcs_warning_state(status) == WARNING
    assert lcs_warning_state("NORMAL") == PASS
    assert lcs_warning_state("INSUFFICIENT_HISTORY") == PASS


# ── History handling options ──

def test_persist_false_writes_nothing(tmp_path):
    history_path = tmp_path / "history.csv"
    detector = _make_detector(tmp_path, history_path=history_path)
    detector.detect(pd.DataFrame(_rows(GROUP, [100.0, 101.0])), history=pd.DataFrame(), persist=False)
    assert not history_path.exists()


def test_build_history_keeps_eligible_unique_rows_up_to_the_cap(tmp_path):
    detector = _make_detector(tmp_path, max_history_per_analyte=3)
    rows = _rows(GROUP, [100.0, 101.0, 125.0, 102.0, 103.0])  # 125 -> offset 2.5, not eligible
    rows.append(dict(rows[0]))  # exact duplicate

    history = detector.build_history(pd.DataFrame(rows))

    assert list(history.columns) == HISTORY_COLUMNS
    assert list(history["measured_value"]) == [101.0, 102.0, 103.0]


def test_rows_not_routed_to_lcs_are_ignored(tmp_path):
    detector = _make_detector(tmp_path)
    rows = (
        _rows(GROUP, [100.0])
        + [_make_row(**OTHER_GROUP, job_code="J1", analysed_date="2024-01-01", value=100.0,
                     std_lot_code="OREAS_502C")]          # SRM lot
        + [_make_row(**OTHER_GROUP, job_code="J1", analysed_date="2024-01-01", value=100.0,
                     analytical_type="Blank")]             # not a Standard
    )
    result = detector.detect(pd.DataFrame(rows), history=pd.DataFrame(), persist=False)

    assert [r["analyte_code"] for r in result.details["results"]] == [GROUP["analyte"]]
    assert result.details["validation_report"]["n_routed_to_lcs"] == 1


def test_missing_required_column_raises(tmp_path):
    detector = _make_detector(tmp_path)
    df = pd.DataFrame(_rows(GROUP, [100.0])).drop(columns=["limit_max"])
    with pytest.raises(ValueError, match="limit_max"):
        detector.detect(df, history=pd.DataFrame(), persist=False)


def test_history_window_marks_trend_window_and_current(tmp_path):
    detector = _make_detector(tmp_path, max_history_per_analyte=30, trend_window=4)
    result = detector.detect(pd.DataFrame(_rows(GROUP, [100.0 + i * 0.1 for i in range(6)])),
                             history=pd.DataFrame(), persist=False)

    last = result.details["results"][-1]["history_window"]
    assert [p["sequence"] for p in last] == list(range(6))
    assert [p["in_trend_window"] for p in last] == [False, False, True, True, True, True]
    assert [p["is_current"] for p in last] == [False] * 5 + [True]


# ── debug mode does not change detection results ──

def test_debug_mode_does_not_change_results(tmp_path):
    existing = _rows(GROUP, [100.0] * 3)
    new_rows = pd.DataFrame(_rows(GROUP, [103.0], start_job=4, day0=4))

    (tmp_path / "quiet").mkdir()
    (tmp_path / "loud").mkdir()
    quiet = _make_detector(tmp_path / "quiet", history_path=_history_csv(tmp_path / "quiet", existing), debug=False)
    loud = _make_detector(tmp_path / "loud", history_path=_history_csv(tmp_path / "loud", existing), debug=True)

    r_quiet, r_loud = _result_for(quiet.detect(new_rows)), _result_for(loud.detect(new_rows))
    for key in ("drift_status", "warning_state", "offset", "n_history", "eligible_for_history", "trend_direction"):
        assert r_quiet[key] == r_loud[key], f"{key} differs between debug=False and debug=True"


# ── Spot-check against the historic notebook's own worked example ──
# (ZN, target 100, max 110, max-warning 106)

def test_matches_historic_notebook_worked_example(tmp_path):
    zn = dict(analyte="ZN", std_code="OREAS_502C", scheme_code="GE_IMS40Q12")
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
             "2024-01-19", "2024-01-20"]  # 14-day gap before the 6th obs, per the worked example
    values = [100.5, 99.8, 101.2, 102.0, 101.5, 103.4, 104.1]
    existing = [
        _make_row(**zn, job_code=f"J{i}", analysed_date=d, value=v)
        for i, (d, v) in enumerate(zip(dates, values), start=1)
    ]
    history_path = _history_csv(tmp_path, existing)
    detector = _make_detector(tmp_path, history_path=history_path, max_history_per_analyte=30)

    result = detector.detect(pd.DataFrame([
        _make_row(**zn, job_code="J8", analysed_date="2024-01-21", value=105.6)
    ]))

    r = _result_for(result, analyte="ZN")
    assert r["n_history"] == 8
    assert r["point_status"] == "NORMAL"  # 0.56 hasn't crossed the 0.6 warning boundary yet
    assert r["trend_direction"] == "UP"
    # 5 of 7 consecutive moves share the UP sign (two dips: points 2 and 5),
    # exactly as the notebook's calculate_trend() formula computes.
    assert r["trend_strength"] == pytest.approx(5 / 7, abs=0.01)
    # progress_to_warning = 1 - (0.6-0.56)/0.6 ~= 0.933 >= 0.75 with >= 6
    # observations -> *_FAILURE_DRIFT, which maps to WARNING (still within limits).
    assert r["drift_status"] == "UPPER_FAILURE_DRIFT"
    assert r["warning_state"] == WARNING


# ── config/lcs_config.yaml is actually wired up ──

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
    assert raw_config["demo_history_path"] != raw_config["history_path"]


def test_history_path_override(tmp_path):
    override = tmp_path / "elsewhere.csv"
    repo_root = Path(__file__).resolve().parent.parent
    detector = LCSDetector(str(repo_root / "config" / "lcs_config.yaml"), history_path=str(override))
    assert detector.history_path == override
