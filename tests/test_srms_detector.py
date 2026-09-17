from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors import srms_detector as srms


def _history_row(
    value: float,
    date: str,
    *,
    standard: str = "STD_A",
    lot: str = "LOT_A",
    scheme: str = "SCH",
    scheme_version: str | None = None,
    analyte: str = "AU",
    unit: str = "MG_KG",
    sample_id: str | None = None,
) -> dict[str, object]:
    row = {
        "SRMSStandardCode": standard,
        "SRMSLotCode": lot,
        "SchemeCode": scheme,
        "Analyte": analyte,
        "UnitCode": unit,
        "Value": value,
        "AnalysisDate": pd.Timestamp(date),
        "SampleID": sample_id or f"{standard}-{date}-{value}",
        "DeviationPct": value,
        "RobustZScore": value,
        "RollingSlopePct": value / 100,
        "RollingStdPctTarget": 1.0,
        "WithinRangePct": 50.0,
        "DistanceToLowerPctSpan": 50.0,
        "DistanceToUpperPctSpan": 50.0,
    }
    if scheme_version is not None:
        row["SchemeVersion"] = scheme_version
    return row


def _current_row(**overrides: object) -> pd.Series:
    data = {
        "StandardCode": "STD_A",
        "StandardLotCode": "LOT_A",
        "Scheme": "SCH",
        "Analyte": "AU",
        "Unit": "MG_KG",
        "Result": 10.0,
        "Target": 10.0,
        "LowerLimit": 8.0,
        "LowerWarning": 9.0,
        "UpperWarning": 11.0,
        "UpperLimit": 12.0,
        "AnalysisDate": pd.Timestamp("2024-02-15"),
    }
    data.update(overrides)
    return pd.Series(data)


def test_exact_historical_identity_matches_and_mismatches() -> None:
    historical = pd.DataFrame(
        [
            _history_row(1, "2024-01-01", sample_id="match"),
            _history_row(2, "2024-01-02", scheme="OTHER", sample_id="scheme"),
            _history_row(3, "2024-01-03", analyte="AG", sample_id="analyte"),
            _history_row(4, "2024-01-04", unit="PPM", sample_id="unit"),
            _history_row(5, "2024-01-05", standard="STD_B", sample_id="standard"),
            _history_row(6, "2024-01-06", lot="LOT_B", sample_id="lot"),
        ]
    )

    group = srms.matching_historical_group(historical, _current_row())

    assert group["SampleID"].tolist() == ["match"]


def test_scheme_version_participates_when_available() -> None:
    historical = pd.DataFrame(
        [
            _history_row(1, "2024-01-01", scheme_version="V1", sample_id="v1"),
            _history_row(2, "2024-01-02", scheme_version="V2", sample_id="v2"),
        ]
    )
    current = _current_row(SchemeVersion="V1")

    group = srms.matching_historical_group(historical, current)

    assert group["SampleID"].tolist() == ["v1"]
    assert "SchemeVersion" in srms.available_historical_identity_columns(historical, current)


def test_recent_window_reports_total_and_uses_latest_30() -> None:
    historical = pd.DataFrame(
        [
            _history_row(value=i, date=str(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)), sample_id=f"h{i}")
            for i in range(35)
        ]
    )
    current = pd.DataFrame([_current_row(Result=50.0)])
    config = srms.SRMSConfig(min_history=5, history_window_size=30, min_iforest_history=100)

    results = srms.add_cclas_standard_current_features(current, historical, config, "synthetic.xlsx")

    assert int(results.at[0, "TotalMatchingHistory"]) == 35
    assert int(results.at[0, "HistoryWindowUsed"]) == 30
    assert int(results.at[0, "HistoryCount"]) == 30
    assert results.at[0, "HistoricalMedian"] == np.median(np.arange(5, 35))


def test_fewer_than_window_but_sufficient_uses_all_available_history() -> None:
    historical = pd.DataFrame(
        [_history_row(value=i, date=f"2024-01-{i + 1:02d}", sample_id=f"h{i}") for i in range(8)]
    )
    current = pd.DataFrame([_current_row(Result=20.0)])
    config = srms.SRMSConfig(min_history=5, history_window_size=30, min_iforest_history=100)

    results = srms.add_cclas_standard_current_features(current, historical, config, "synthetic.xlsx")

    assert int(results.at[0, "TotalMatchingHistory"]) == 8
    assert int(results.at[0, "HistoryWindowUsed"]) == 8
    assert results.at[0, "Historical_Status"] == "WARNING"


def test_insufficient_history_skips_historical_and_ml_but_keeps_limit_status() -> None:
    historical = pd.DataFrame(
        [_history_row(value=i, date=f"2024-01-{i + 1:02d}", sample_id=f"h{i}") for i in range(4)]
    )
    current = pd.DataFrame([_current_row(Result=10.0)])
    config = srms.SRMSConfig(min_history=5, history_window_size=30, min_iforest_history=20)
    limit_results = current.apply(srms.detect_standard_limit_status, axis=1, result_type="expand")
    current["Limit_Status"] = limit_results[0]

    featured = srms.add_cclas_standard_current_features(current, historical, config, "synthetic.xlsx")
    modelled = srms.add_cclas_standard_historical_isolation_forest(featured, historical, config)

    assert modelled.at[0, "Limit_Status"] == "PASS"
    assert modelled.at[0, "Historical_Status"] == "INSUFFICIENT_HISTORY"
    assert modelled.at[0, "IsolationForestStatus"] == "INSUFFICIENT_HISTORY"
    assert pd.isna(modelled.at[0, "RobustZScore"])


def test_future_and_current_observations_are_excluded_from_baseline() -> None:
    historical = pd.DataFrame(
        [
            _history_row(1, "2024-01-01", sample_id="past"),
            _history_row(2, "2024-02-15", sample_id="current_time"),
            _history_row(3, "2024-03-01", sample_id="future"),
        ]
    )

    group = srms.matching_historical_group(historical, _current_row(AnalysisDate=pd.Timestamp("2024-02-15")))

    assert group["SampleID"].tolist() == ["past"]


def test_known_limit_fail_remains_critical_with_supporting_signals() -> None:
    row = pd.Series(
        {
            "Limit_Status": "FAIL",
            "Historical_Status": "PASS",
            "IsolationForestAnomaly": False,
            "DriftFlag": False,
        }
    )

    assert srms.assign_cclas_standard_final_risk(row) == "Critical"


def test_inclusive_exclusive_limit_boundaries() -> None:
    inclusive_lower = _current_row(Result=8.0, LowerLimit=8.0, LowerLimitInclusive="Y")
    exclusive_lower = _current_row(Result=8.0, LowerLimit=8.0, LowerLimitInclusive="N")
    inclusive_warning = _current_row(Result=9.0, LowerWarning=9.0, LowerWarningInclusive="Y")
    exclusive_warning = _current_row(Result=9.0, LowerWarning=9.0, LowerWarningInclusive="N")

    assert srms.detect_standard_limit_status(inclusive_lower)[0] == "WARNING"
    assert srms.detect_standard_limit_status(exclusive_lower)[0] == "FAIL"
    assert srms.detect_standard_limit_status(inclusive_warning)[0] == "PASS"
    assert srms.detect_standard_limit_status(exclusive_warning)[0] == "WARNING"


def test_historical_backtest_uses_prior_window_without_lookahead() -> None:
    raw = pd.DataFrame(
        {
            "ANALYTICAL_TYPE": ["Standard"] * 7,
            "STD_LOT_CODE": ["LOT_A"] * 7,
            "STD_CODE": ["STD_A"] * 7,
            "JOB_CODE": [f"J{i}" for i in range(7)],
            "NUMERIC_FINAL_VALUE": [1, 2, 3, 4, 5, 6, 100],
            "ANALYSED_DATE": pd.date_range("2024-01-01", periods=7, freq="D"),
            "SCHEME_CODE": ["SCH"] * 7,
            "ANALYTE_CODE": ["AU"] * 7,
            "STANDARD_STATUS": ["PASS"] * 7,
            "PRECISION_STATUS": ["PASS"] * 7,
            "INTERNAL_MIN_VALUE": [0] * 7,
            "INTERNAL_MAX_VALUE": [200] * 7,
            "INTERNAL_MIN_INCLUSIVE": ["Y"] * 7,
            "INTERNAL_MAX_INCLUSIVE": ["Y"] * 7,
            "INTERNAL_MAX_WARNING_VALUE": [150] * 7,
            "INTERNAL_MIN_WARNING_VALUE": [0.5] * 7,
            "INTERNAL_MIN_WARNING_INCLUSIVE": ["Y"] * 7,
            "INTERNAL_MAX_WARNING_INCLUSIVE": ["Y"] * 7,
            "INTERNAL_TARGET_VALUE": [5] * 7,
            "PARENT_NUMERIC_FINAL_VALUE": [np.nan] * 7,
            "UNIT_CODE": ["MG_KG"] * 7,
            "SPECIFICATION_CODE": ["STD_A"] * 7,
        }
    )
    config = srms.SRMSConfig(min_history=3, history_window_size=3, min_iforest_history=100)

    results = srms.run_historical_backtest(raw, config)
    last = results.sort_values("AnalysisDate").iloc[-1]

    assert int(last["TotalMatchingHistory"]) == 6
    assert int(last["HistoryWindowUsed"]) == 3
    assert last["HistoricalMedian"] == 5
