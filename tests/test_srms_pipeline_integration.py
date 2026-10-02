from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.detectors import DETECTOR_REGISTRY, AnomalyResult, DetectionResultSet
from src.main import run_common_pipeline


def _history_row(value: float, date: str, sample_id: int) -> dict[str, object]:
    return {
        "ANALYTICAL_TYPE": "Standard",
        "STD_LOT_CODE": "LOT_A",
        "STD_CODE": "STD_A",
        "JOB_CODE": f"H{sample_id}",
        "NUMERIC_FINAL_VALUE": value,
        "ANALYSED_DATE": date,
        "SCHEME_CODE": "SCH",
        "ANALYTE_CODE": "AU",
        "STANDARD_STATUS": "PASS",
        "PRECISION_STATUS": "PASS",
        "INTERNAL_MIN_VALUE": 8.0,
        "INTERNAL_MAX_VALUE": 12.0,
        "INTERNAL_MIN_INCLUSIVE": "Y",
        "INTERNAL_MAX_INCLUSIVE": "Y",
        "INTERNAL_MAX_WARNING_VALUE": 11.0,
        "INTERNAL_MIN_WARNING_VALUE": 9.0,
        "INTERNAL_MIN_WARNING_INCLUSIVE": "Y",
        "INTERNAL_MAX_WARNING_INCLUSIVE": "Y",
        "INTERNAL_TARGET_VALUE": 10.0,
        "PARENT_NUMERIC_FINAL_VALUE": pd.NA,
        "UNIT_CODE": "MG_KG",
        "SPECIFICATION_CODE": "STD_A",
    }


def _write_history(path: Path, count: int) -> None:
    rows = [
        _history_row(
            value=9.8 + (i % 5) * 0.1,
            date=str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)),
            sample_id=i,
        )
        for i in range(count)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _append_sample(rows: list[dict[str, object]], sample_code: str, qc_type: str, result: float) -> None:
    sample_props = {
        "S_JobCode": "JOB_CURRENT",
        "S_SampleCode": sample_code,
        "S_PrimaryAnalyticalType": "Standard" if qc_type == "STD" else "Unknown",
        "S_QcTypeCode": qc_type,
        "S_StandardCode": "STD_A",
        "S_StandardLotCode": "LOT_A",
    }
    result_props = {
        "RSC_SchemeCode": "SCH",
        "RSA_AnalyteCode": "AU",
        "SSA_UnitCode": "MG_KG",
        "SSA_NumericFinalValue": result,
        "PA_InternalTargetValue": 10.0,
        "PA_InternalMinValue": 8.0,
        "PA_InternalMinInclusive": "Y",
        "PA_InternalMinWarningValue": 9.0,
        "PA_InternalMinWarningInclusive": "Y",
        "PA_InternalMaxWarningValue": 11.0,
        "PA_InternalMaxWarningInclusive": "Y",
        "PA_InternalMaxValue": 12.0,
        "PA_InternalMaxInclusive": "Y",
        "SSA_StandardStatus": "FAIL" if result > 12 else "PASS",
    }
    rows.extend({"OBJECT": "VSAMPLE", "PROPERTY": key, "VALUE": value} for key, value in sample_props.items())
    rows.extend({"OBJECT": "VSSA", "PROPERTY": key, "VALUE": value} for key, value in result_props.items())


def _write_cclas_tsv(path: Path) -> None:
    rows: list[dict[str, object]] = []
    _append_sample(rows, "STD_PASS", "STD", 10.0)
    _append_sample(rows, "STD_FAIL", "STD", 13.0)
    _append_sample(rows, "UNKNOWN_CONTEXT", "UNK", 10.0)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_srms_is_registered_in_common_detector_registry() -> None:
    assert "srms" in DETECTOR_REGISTRY


def test_srms_runs_through_common_pipeline_and_routes_only_standard_records(tmp_path: Path) -> None:
    current_path = tmp_path / "current.tsv"
    history_path = tmp_path / "history.csv"
    _write_cclas_tsv(current_path)
    _write_history(history_path, 35)

    result_sets = run_common_pipeline(str(current_path), detector_name="srms", history_path=str(history_path))

    result_set = result_sets["srms"]
    assert isinstance(result_set, DetectionResultSet)
    assert len(result_set.results) == 2
    assert set(result_set.results["SampleCode"]) == {"STD_PASS", "STD_FAIL"}
    assert all(isinstance(result, AnomalyResult) for result in result_set.anomalies)


def test_srms_common_output_preserves_pass_fail_and_history_window(tmp_path: Path) -> None:
    current_path = tmp_path / "current.tsv"
    history_path = tmp_path / "history.csv"
    _write_cclas_tsv(current_path)
    _write_history(history_path, 35)

    result_set = run_common_pipeline(str(current_path), detector_name="srms", history_path=str(history_path))["srms"]
    results = result_set.results.set_index("SampleCode")

    assert results.at["STD_PASS", "Limit_Status"] == "PASS"
    assert results.at["STD_PASS", "Final_Risk"] == "Low"
    assert results.at["STD_FAIL", "Limit_Status"] == "FAIL"
    assert results.at["STD_FAIL", "Final_Risk"] == "Critical"
    assert int(results.at["STD_PASS", "TotalMatchingHistory"]) == 35
    assert int(results.at["STD_PASS", "HistoryWindowUsed"]) == 30

    fail_common = next(result for result in result_set.anomalies if result.record_id == "STD_FAIL")
    assert fail_common.anomaly_detected is True
    assert fail_common.severity == "Critical"
    assert fail_common.details["Limit_Status"] == "FAIL"


def test_srms_common_pipeline_handles_insufficient_history(tmp_path: Path) -> None:
    current_path = tmp_path / "current.tsv"
    history_path = tmp_path / "history.csv"
    _write_cclas_tsv(current_path)
    _write_history(history_path, 4)

    results = run_common_pipeline(str(current_path), detector_name="srms", history_path=str(history_path))["srms"].results

    pass_row = results.set_index("SampleCode").loc["STD_PASS"]
    assert pass_row["Limit_Status"] == "PASS"
    assert pass_row["Historical_Status"] == "INSUFFICIENT_HISTORY"
    assert pass_row["IsolationForestStatus"] == "INSUFFICIENT_HISTORY"
