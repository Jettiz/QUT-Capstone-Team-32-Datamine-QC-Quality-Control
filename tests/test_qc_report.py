"""
Tests for src/qc_report: the display helpers, the result contract's JSON
shape, the registry, and the LCS adapter (History on/off behaviour, job and
item grouping, Job/Instrument labels).
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.qc_report import (
    INSTRUMENT_UNKNOWN, JOB_UNKNOWN, QC_METHODS, display_instrument, display_instruments, display_job, register,
)
from src.qc_report.lcs_method import LCSMethod

REPO_ROOT = Path(__file__).resolve().parent.parent


def _row(job, day, analyte="CU", value=100.0, scheme="GE_ICP40Q12", std_code="OREAS_502C",
         instrument=None, lot="Sample"):
    return {
        "analytical_type": "Standard", "std_lot_code": lot, "std_code": std_code,
        "scheme_code": scheme, "job_code": job, "instrument_id": instrument, "analyte_code": analyte,
        "analysed_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day), "measured_value": value,
        "target_value": 100.0, "limit_max": 110.0, "limit_min": 90.0,
        "limit_max_warning": 106.0, "limit_min_warning": 94.0,
        "limit_max_inclusive": "Y", "limit_min_inclusive": "Y",
        "limit_max_warning_inclusive": "Y", "limit_min_warning_inclusive": "Y",
        "unit_code": "MG_KG", "standard_status": "Pass",
    }


def _method(tmp_path) -> LCSMethod:
    """LCS adapter whose History-on store lives in tmp_path, not in data/processed."""
    config = yaml.safe_load((REPO_ROOT / "config" / "lcs_config.yaml").read_text())
    config["demo_history_path"] = str(tmp_path / "demo_history.csv")
    config["history_path"] = str(tmp_path / "history.csv")
    path = tmp_path / "lcs_config.yaml"
    path.write_text(yaml.dump(config))
    return LCSMethod(config_path=path)


def _four_jobs():
    rows = []
    for j in range(4):
        rows.append(_row(f"J{j}", j, "CU", 100.0 + j))
        rows.append(_row(f"J{j}", j, "PB", 100.0 - j))
    return pd.DataFrame(rows)


# ── display helpers ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [None, np.nan, pd.NA, pd.NaT, "", "   ", "nan", "NaN", "None", "null"])
def test_missing_values_never_reach_the_user(value):
    assert display_job(value) == JOB_UNKNOWN == "Job unknown"
    assert display_instrument(value) == INSTRUMENT_UNKNOWN == "Instrument unknown"


def test_real_values_are_shown_trimmed():
    assert display_job(" JOB_1 ") == "JOB_1"
    assert display_instrument("ICP-2") == "ICP-2"


def test_display_instruments_lists_distinct_values_and_unknown_last():
    assert display_instruments([None, "B", "A", "B"]) == ["A", "B", "Instrument unknown"]
    assert display_instruments([np.nan, ""]) == ["Instrument unknown"]
    assert display_instruments(["A"]) == ["A"]


# ── registry ─────────────────────────────────────────────────────────────────

def test_lcs_is_registered_under_the_poc_detector_id():
    assert isinstance(QC_METHODS["control"], LCSMethod)


def test_registering_a_duplicate_id_fails():
    with pytest.raises(ValueError):
        register(QC_METHODS["control"])


# ── LCS adapter ──────────────────────────────────────────────────────────────

def test_history_off_analyses_every_job_and_stores_nothing(tmp_path):
    method = _method(tmp_path)
    result = method.run(_four_jobs(), use_history=False)

    assert result.usable and result.history_mode == "off"
    assert [job.job for job in result.jobs] == ["J0", "J1", "J2", "J3"]
    assert all(len(job.items) == 2 for job in result.jobs)
    assert not (tmp_path / "demo_history.csv").exists()
    assert "History off" in result.notices[0]


def test_history_on_uses_the_older_half_as_history(tmp_path):
    method = _method(tmp_path)
    result = method.run(_four_jobs(), use_history=True)

    assert [job.job for job in result.jobs] == ["J2", "J3"]          # newer half analysed
    first_items = {i.code: i for i in result.jobs[0].items}
    assert "3 (since" in next(m["value"] for m in first_items["CU"].metrics if m["label"] == "Observations used")
    assert "older 2 of 4" in result.notices[0]
    stored = pd.read_csv(tmp_path / "demo_history.csv")
    assert len(stored) == 8                                            # all 4 jobs x 2 analytes


def test_history_on_with_a_single_job_explains_itself(tmp_path):
    method = _method(tmp_path)
    result = method.run(pd.DataFrame([_row("J0", 0)]), use_history=True)
    assert len(result.jobs) == 1
    assert "only one LCS job" in result.notices[0]


def test_not_usable_without_lcs_rows(tmp_path):
    method = _method(tmp_path)
    result = method.run(pd.DataFrame([_row("J0", 0, lot="OREAS_905")]), use_history=False)
    assert result.usable is False
    assert result.jobs == []
    assert result.notices


def test_same_analyte_in_two_schemes_becomes_two_variants(tmp_path):
    df = pd.DataFrame([_row("J0", 0, "BI", scheme="GE_ICP40Q12"), _row("J0", 0, "BI", value=120.0, scheme="GE_IMS40Q12")])
    result = _method(tmp_path).run(df, use_history=False)

    job = result.jobs[0].to_dict()
    assert [(i["code"], i["variant"], i["state"]) for i in job["items"]] == [
        ("BI", "GE_IMS40Q12", "FAIL"), ("BI", "GE_ICP40Q12", "PASS"),  # worst first
    ]


def test_job_and_instrument_labels_never_show_missing_values(tmp_path):
    df = pd.DataFrame([_row(None, 0, "CU"), _row(None, 0, "PB", instrument="ICP-7"), _row("J1", 1, "CU", instrument="")])
    result = _method(tmp_path).run(df, use_history=False)

    jobs = {job.job: job for job in result.jobs}
    assert set(jobs) == {"Job unknown", "J1"}
    assert jobs["Job unknown"].instruments == ["ICP-7", "Instrument unknown"]
    assert jobs["J1"].instruments == ["Instrument unknown"]

    payload = json.dumps([job.to_dict() for job in result.jobs] + [result.to_summary()], allow_nan=False)
    for bad in ('"None"', '"nan"', '"NaN"', '"NaT"'):
        assert bad not in payload, f"{bad} leaked into the result payload"
    assert not re.search(r"[:\[,]\s*null\b", payload), "a JSON null value leaked into the result payload"


def test_items_explain_their_state(tmp_path):
    df = pd.DataFrame([_row("J0", 0, "CU", value=115.0)])
    item = _method(tmp_path).run(df, use_history=False).jobs[0].items[0]

    assert item.state == "FAIL" and item.status_detail == "UPPER_FAILURE"
    labels = [m["label"] for m in item.metrics]
    for label in ("Result", "Target", "Warning limits", "Failure limits", "Offset", "Point check",
                  "Company status (STANDARD_STATUS)", "Instrument"):
        assert label in labels
    assert "breached the upper failure limit" in item.reason


def test_render_chart_writes_a_png(tmp_path):
    method = _method(tmp_path)
    item = method.run(_four_jobs(), use_history=False).jobs[-1].items[0]
    path = method.render_chart(item, tmp_path / "chart.png")
    assert path.read_bytes()[:4] == b"\x89PNG"
