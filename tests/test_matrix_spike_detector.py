from pathlib import Path


import numpy as np
import pandas as pd
import pytest


from src.detectors.matrix_spike_detector import (
   DEFAULT_CONFIG_PATH,
   IF_FEATURES,
   OUTPUT_COLUMNS,
   MSConfig,
   _assign_detection_method,
   _assign_final_risk,
   _merge_drift_flags,
   apply_rule_based_flags,
   compute_rolling_warning_position,
   filter_ms_records,
   load_ms_config,
   load_ms_data,
   run_isolation_forest,
   run_ms_detection,
   validate_against_cclas,
   validate_ms_frame,
)




# ---------------------------------------------------------------------------
# Shared synthetic Matrix Spike row
# ---------------------------------------------------------------------------


def _ms_row(**overrides):
   row = {
       "ANALYTICAL_TYPE": "Spike",
       "QC_TYPE": "MS",
       "STD_LOT_CODE": "LOT_A",
       "STD_CODE": "STD_A",
       "JOB_CODE": "JOB_1",
       "NUMERIC_FINAL_VALUE": 100.0,
       "ANALYSED_DATE": pd.Timestamp("2024-01-01"),
       "SCHEME_CODE": "SCH",
       "ANALYTE_CODE": "AG",
       "UNIT_CODE": "PPM",
       "STANDARD_STATUS": "Pass",
       "INTERNAL_TARGET_VALUE": 100.0,
       "INTERNAL_MIN_VALUE": 75.0,
       "INTERNAL_MAX_VALUE": 125.0,
       "INTERNAL_MIN_WARNING_VALUE": 85.0,
       "INTERNAL_MAX_WARNING_VALUE": 115.0,
       "INTERNAL_MIN_INCLUSIVE": "Y",
       "INTERNAL_MAX_INCLUSIVE": "Y",
       "INTERNAL_MIN_WARNING_INCLUSIVE": "Y",
       "INTERNAL_MAX_WARNING_INCLUSIVE": "Y",
   }
   row.update(overrides)
   return row




# ---------------------------------------------------------------------------
# Filtering / input handling
# ---------------------------------------------------------------------------


def test_filter_ms_records_preserves_na_analyte():
   df = pd.DataFrame(
       {
           "ANALYTICAL_TYPE": [
               "Spike",
               "Spike",
               "Spike",
               "Blank",
           ],
           "QC_TYPE": [
               "MS",
               "MS",
               "MSD",
               "MS",
           ],
           "ANALYTE_CODE": [
               "NA",
               "AG",
               "AG",
               "AG",
           ],
       }
   )


   result = filter_ms_records(df)


   assert len(result) == 2
   assert "NA" in result["ANALYTE_CODE"].values


   assert (
       result["QC_TYPE"]
       .str.upper()
       .eq("MS")
       .all()
   )




def test_filter_ms_records_rejects_missing_filter_columns():
   df = pd.DataFrame(
       {
           "ANALYTICAL_TYPE": ["Spike"],
           "ANALYTE_CODE": ["AG"],
       }
   )


   with pytest.raises(
       ValueError,
       match="missing filter columns",
   ):
       filter_ms_records(df)




def test_filter_ms_records_rejects_blank_missing_analytes_without_mutating_input():
   df = pd.DataFrame(
       {
           "ANALYTICAL_TYPE": [
               "Spike",
               "Spike",
               "Spike",
               "Spike",
           ],
           "QC_TYPE": [
               "MS",
               "MS",
               "MS",
               "MS",
           ],
           "ANALYTE_CODE": [
               "AG",
               np.nan,
               "",
               "NA",
           ],
       }
   )


   original = df.copy(deep=True)


   result = filter_ms_records(df)


   assert (
       result["ANALYTE_CODE"].tolist()
       == ["AG", "NA"]
   )


   pd.testing.assert_frame_equal(
       df,
       original,
   )




def test_load_ms_data_preserves_literal_na_analyte(
   tmp_path,
):
   path = (
       tmp_path
       / "matrix_spike.csv"
   )


   pd.DataFrame(
       [
           {
               "ANALYTICAL_TYPE": "Spike",
               "QC_TYPE": "MS",
               "ANALYTE_CODE": "NA",
           }
       ]
   ).to_csv(
       path,
       index=False,
   )


   loaded = load_ms_data(
       str(path)
   )


   assert (
       loaded.loc[
           0,
           "ANALYTE_CODE",
       ]
       == "NA"
   )


   assert not pd.isna(
       loaded.loc[
           0,
           "ANALYTE_CODE",
       ]
   )




# ---------------------------------------------------------------------------
# Required schema / rule-based CCLAS classification
# ---------------------------------------------------------------------------


def test_validate_ms_frame_missing_required_column():
   df = pd.DataFrame(
       {
           "ANALYTICAL_TYPE": ["Spike"],
           "QC_TYPE": ["MS"],
       }
   )


   with pytest.raises(
       ValueError
   ):
       validate_ms_frame(df)




def test_rule_based_classification():
   df = pd.DataFrame(
       {
           "TRANSFORMED_VALUE": [
               0.0,
               0.75,
               1.2,
           ],
           "TRANSFORMED_MAX_WARNING": [
               0.5,
               0.5,
               0.5,
           ],
           "TRANSFORMED_MIN_WARNING": [
               -0.5,
               -0.5,
               -0.5,
           ],
           "IS_IGNORED": [
               False,
               False,
               False,
           ],
       }
   )


   result = apply_rule_based_flags(
       df
   )


   assert (
       result.loc[
           0,
           "RULE_FLAG",
       ]
       == "Pass"
   )


   assert (
       result.loc[
           1,
           "RULE_FLAG",
       ]
       == "Warning"
   )


   assert (
       result.loc[
           2,
           "RULE_FLAG",
       ]
       == "Failure"
   )




@pytest.mark.parametrize(
   (
       "value,"
       "max_inclusive,"
       "min_inclusive,"
       "max_warning_inclusive,"
       "min_warning_inclusive,"
       "expected"
   ),
   [
       (
           1.0,
           "Y",
           "Y",
           "Y",
           "Y",
           "Failure",
       ),
       (
           -1.0,
           "Y",
           "Y",
           "Y",
           "Y",
           "Failure",
       ),
       (
           1.0,
           "N",
           "Y",
           "Y",
           "Y",
           "Warning",
       ),
       (
           -1.0,
           "Y",
           "N",
           "Y",
           "Y",
           "Warning",
       ),
       (
           0.5,
           "Y",
           "Y",
           "Y",
           "Y",
           "Warning",
       ),
       (
           -0.5,
           "Y",
           "Y",
           "Y",
           "Y",
           "Warning",
       ),
       (
           0.5,
           "Y",
           "Y",
           "N",
           "Y",
           "Pass",
       ),
       (
           -0.5,
           "Y",
           "Y",
           "Y",
           "N",
           "Pass",
       ),
   ],
)
def test_inclusive_exclusive_rule_boundaries(
   value,
   max_inclusive,
   min_inclusive,
   max_warning_inclusive,
   min_warning_inclusive,
   expected,
):
   df = pd.DataFrame(
       {
           "TRANSFORMED_VALUE": [
               value
           ],
           "TRANSFORMED_MAX_WARNING": [
               0.5
           ],
           "TRANSFORMED_MIN_WARNING": [
               -0.5
           ],
           "IS_IGNORED": [
               False
           ],
           "INTERNAL_MAX_INCLUSIVE": [
               max_inclusive
           ],
           "INTERNAL_MIN_INCLUSIVE": [
               min_inclusive
           ],
           "INTERNAL_MAX_WARNING_INCLUSIVE": [
               max_warning_inclusive
           ],
           "INTERNAL_MIN_WARNING_INCLUSIVE": [
               min_warning_inclusive
           ],
       }
   )


   result = apply_rule_based_flags(
       df
   )


   assert (
       result.loc[
           0,
           "RULE_FLAG",
       ]
       == expected
   )




# ---------------------------------------------------------------------------
# Historical ordering / drift behaviour
# ---------------------------------------------------------------------------


def test_rolling_warning_position_uses_chronological_drift_group_order():
   cfg = MSConfig(
       ts_rolling_window=2,
       ts_min_periods=1,
   )


   # Intentionally supplied out of chronological
   # order and across two different lots.
   #
   # Drift history itself is Scheme-Anayte-Unit,
   # so lot ordering must not disturb chronology.
   df = pd.DataFrame(
       {
           "STD_LOT_CODE": [
               "LOT_A",
               "LOT_A",
               "LOT_B",
               "LOT_B",
           ],
           "SCHEME_CODE": [
               "SCH",
               "SCH",
               "SCH",
               "SCH",
           ],
           "ANALYTE_CODE": [
               "AG",
               "AG",
               "AG",
               "AG",
           ],
           "UNIT_CODE": [
               "PPM",
               "PPM",
               "PPM",
               "PPM",
           ],
           "ANALYSED_DATE": pd.to_datetime(
               [
                   "2024-01-01",
                   "2024-01-03",
                   "2024-01-02",
                   "2024-01-04",
               ]
           ),
           "NUMERIC_FINAL_VALUE": [
               100.0,
               120.0,
               110.0,
               130.0,
           ],
           "INTERNAL_TARGET_VALUE": [
               100.0,
               100.0,
               100.0,
               100.0,
           ],
           "INTERNAL_MIN_WARNING_VALUE": [
               90.0,
               90.0,
               90.0,
               90.0,
           ],
           "INTERNAL_MAX_WARNING_VALUE": [
               110.0,
               110.0,
               110.0,
               110.0,
           ],
       }
   )


   result = (
       compute_rolling_warning_position(
           df,
           cfg,
       )
   )


   expected_dates = list(
       pd.to_datetime(
           [
               "2024-01-01",
               "2024-01-02",
               "2024-01-03",
               "2024-01-04",
           ]
       )
   )


   assert (
       result[
           "ANALYSED_DATE"
       ].tolist()
       == expected_dates
   )


   np.testing.assert_allclose(
       result[
           "ROLLING_WARNING_POSITION"
       ].to_numpy(),
       np.array(
           [
               0.0,
               0.5,
               1.5,
               2.5,
           ]
       ),
   )




def test_drift_not_applied_before_onset():
   cfg = MSConfig()


   df = pd.DataFrame(
       {
           "SCHEME_CODE": [
               "TEST",
               "TEST",
               "TEST",
           ],
           "ANALYTE_CODE": [
               "AG",
               "AG",
               "AG",
           ],
           "UNIT_CODE": [
               "PPM",
               "PPM",
               "PPM",
           ],
           "ANALYSED_DATE": (
               pd.to_datetime(
                   [
                       "2026-01-01",
                       "2026-01-02",
                       "2026-01-03",
                   ]
               )
           ),
       }
   )


   drift_summary = pd.DataFrame(
       {
           "SCHEME_CODE": [
               "TEST"
           ],
           "ANALYTE_CODE": [
               "AG"
           ],
           "UNIT_CODE": [
               "PPM"
           ],
           "warning_drift_detected": [
               True
           ],
           "warning_direction": [
               "Upper"
           ],
           "warning_drift_start": [
               pd.Timestamp(
                   "2026-01-02"
               )
           ],
           "failure_drift_detected": [
               False
           ],
           "failure_direction": [
               None
           ],
           "failure_drift_start": [
               pd.NaT
           ],
       }
   )


   result = _merge_drift_flags(
       df,
       drift_summary,
       cfg,
   )


   assert (
       result.loc[
           0,
           "DRIFT_FLAG",
       ]
       == "None"
   )


   assert (
       result.loc[
           1,
           "DRIFT_FLAG",
       ]
       == "Warning"
   )


   assert (
       result.loc[
           2,
           "DRIFT_FLAG",
       ]
       == "Warning"
   )


   assert pd.isna(
       result.loc[
           0,
           "DRIFT_ONSET",
       ]
   )


   assert (
       result.loc[
           1,
           "DRIFT_ONSET",
       ]
       == pd.Timestamp(
           "2026-01-02"
       )
   )




def test_drift_isolated_to_exact_scheme_analyte_unit_group():
   cfg = MSConfig()


   df = pd.DataFrame(
       {
           "SCHEME_CODE": [
               "SCH",
               "SCH",
               "SCH",
               "SCH",
           ],
           "ANALYTE_CODE": [
               "AG",
               "AG",
               "CU",
               "CU",
           ],
           "UNIT_CODE": [
               "PPM",
               "PPM",
               "PPM",
               "PPM",
           ],
           "ANALYSED_DATE": (
               pd.to_datetime(
                   [
                       "2024-01-01",
                       "2024-01-02",
                       "2024-01-01",
                       "2024-01-02",
                   ]
               )
           ),
       }
   )


   drift_summary = pd.DataFrame(
       {
           "SCHEME_CODE": [
               "SCH"
           ],
           "ANALYTE_CODE": [
               "AG"
           ],
           "UNIT_CODE": [
               "PPM"
           ],
           "warning_drift_detected": [
               True
           ],
           "warning_direction": [
               "Upper"
           ],
           "warning_drift_start": [
               pd.Timestamp(
                   "2024-01-02"
               )
           ],
           "failure_drift_detected": [
               False
           ],
           "failure_direction": [
               None
           ],
           "failure_drift_start": [
               pd.NaT
           ],
       }
   )


   result = _merge_drift_flags(
       df,
       drift_summary,
       cfg,
   )


   ag = (
       result[
           result[
               "ANALYTE_CODE"
           ]
           == "AG"
       ]
       .sort_values(
           "ANALYSED_DATE"
       )
   )


   cu = (
       result[
           result[
               "ANALYTE_CODE"
           ]
           == "CU"
       ]
       .sort_values(
           "ANALYSED_DATE"
       )
   )


   assert (
       ag[
           "DRIFT_FLAG"
       ].tolist()
       == [
           "None",
           "Warning",
       ]
   )


   assert (
       cu[
           "DRIFT_FLAG"
       ].tolist()
       == [
           "None",
           "None",
       ]
   )




# ---------------------------------------------------------------------------
# Final risk / detection method
# ---------------------------------------------------------------------------


def test_warning_with_failure_drift_is_high():
   risk = _assign_final_risk(
       rule_flag="Warning",
       if_anomaly=False,
       drift_flag="Failure",
       is_ignored=False,
   )


   assert risk == "High"




@pytest.mark.parametrize(
   (
       "rule_flag,"
       "if_anomaly,"
       "drift_flag,"
       "is_ignored,"
       "expected"
   ),
   [
       (
           "Failure",
           False,
           "None",
           False,
           "Critical",
       ),
       (
           "Warning",
           False,
           "Failure",
           False,
           "High",
       ),
       (
           "Warning",
           True,
           "None",
           False,
           "High",
       ),
       (
           "Warning",
           False,
           "Warning",
           False,
           "Medium",
       ),
       (
           "Pass",
           False,
           "Failure",
           False,
           "High",
       ),
       (
           "Pass",
           False,
           "Warning",
           False,
           "Medium",
       ),
       (
           "Pass",
           True,
           "None",
           False,
           "Medium",
       ),
       (
           "Pass",
           False,
           "None",
           False,
           "Low",
       ),
       (
           "Failure",
           True,
           "Failure",
           True,
           "Ignored",
       ),
   ],
)
def test_final_risk_matrix(
   rule_flag,
   if_anomaly,
   drift_flag,
   is_ignored,
   expected,
):
   assert (
       _assign_final_risk(
           rule_flag=rule_flag,
           if_anomaly=if_anomaly,
           drift_flag=drift_flag,
           is_ignored=is_ignored,
       )
       == expected
   )




def test_detection_method_contains_drift():
   method = (
       _assign_detection_method(
           rule_flag="Pass",
           if_anomaly=False,
           drift_flag="Failure",
           is_ignored=False,
       )
   )


   assert method == "Drift"




@pytest.mark.parametrize(
   (
       "rule_flag,"
       "if_anomaly,"
       "drift_flag,"
       "is_ignored,"
       "expected"
   ),
   [
       (
           "Pass",
           False,
           "None",
           False,
           "None",
       ),
       (
           "Warning",
           False,
           "None",
           False,
           "Rule-based",
       ),
       (
           "Pass",
           False,
           "Warning",
           False,
           "Drift",
       ),
       (
           "Pass",
           True,
           "None",
           False,
           "IsolationForest",
       ),
       (
           "Warning",
           True,
           "Failure",
           False,
           (
               "Rule-based + Drift "
               "+ IsolationForest"
           ),
       ),
       (
           "Failure",
           True,
           "Failure",
           True,
           "Ignored",
       ),
   ],
)
def test_detection_method_matrix(
   rule_flag,
   if_anomaly,
   drift_flag,
   is_ignored,
   expected,
):
   assert (
       _assign_detection_method(
           rule_flag=rule_flag,
           if_anomaly=if_anomaly,
           drift_flag=drift_flag,
           is_ignored=is_ignored,
       )
       == expected
   )




def test_ignored_row_remains_ignored():
   risk = _assign_final_risk(
       rule_flag="Failure",
       if_anomaly=True,
       drift_flag="Failure",
       is_ignored=True,
   )


   method = (
       _assign_detection_method(
           rule_flag="Failure",
           if_anomaly=True,
           drift_flag="Failure",
           is_ignored=True,
       )
   )


   assert risk == "Ignored"
   assert method == "Ignored"




# ---------------------------------------------------------------------------
# CCLAS comparison
# ---------------------------------------------------------------------------


def test_cclas_validation_matching_rows():
   df = pd.DataFrame(
       {
           "STANDARD_STATUS": [
               "Pass",
               "UpperWarning",
               "LowerWarning",
               "UpperFailure",
               "LowerFailure",
               "IgnoredUpperFailure",
               "IgnoredLowerFailure",
           ],
           "RULE_FLAG": [
               "Pass",
               "Warning",
               "Warning",
               "Failure",
               "Failure",
               "Ignored",
               "Ignored",
           ],
           "IS_IGNORED": [
               False,
               False,
               False,
               False,
               False,
               True,
               True,
           ],
       }
   )


   result = (
       validate_against_cclas(
           df
       )
   )


   assert (
       result[
           "CCLAS_MISMATCH"
       ].sum()
       == 0
   )




def test_unknown_cclas_status_stays_unknown_and_is_not_false_mismatch():
   df = pd.DataFrame(
       {
           "STANDARD_STATUS": [
               "UnexpectedStatus"
           ],
           "RULE_FLAG": [
               "Pass"
           ],
           "IS_IGNORED": [
               False
           ],
       }
   )


   result = (
       validate_against_cclas(
           df
       )
   )


   assert (
       result.loc[
           0,
           "CCLAS_STATUS_NORMALISED",
       ]
       == "UNKNOWN"
   )


   assert (
       result.loc[
           0,
           "CCLAS_MISMATCH",
       ]
       == False
   )




# ---------------------------------------------------------------------------
# Isolation Forest edge cases
# ---------------------------------------------------------------------------


def test_isolation_forest_small_dataset_skipped():
   cfg = MSConfig(
       iforest_min_train=20
   )


   data = {
       feature: [
           0.1,
           0.2,
       ]
       for feature in IF_FEATURES
   }


   data[
       "IS_IGNORED"
   ] = [
       False,
       False,
   ]


   df = pd.DataFrame(
       data
   )


   result = (
       run_isolation_forest(
           df,
           cfg,
       )
   )


   assert (
       result[
           "IF_STATUS"
       ]
       .eq(
           "SKIPPED"
       )
       .all()
   )


   assert (
       result[
           "IF_ANOMALY"
       ]
       .eq(
           False
       )
       .all()
   )




def test_isolation_forest_all_ignored_rows_skipped():
   cfg = MSConfig(
       iforest_min_train=1
   )


   data = {
       feature: [
           0.1,
           0.2,
       ]
       for feature in IF_FEATURES
   }


   data[
       "IS_IGNORED"
   ] = [
       True,
       True,
   ]


   df = pd.DataFrame(
       data
   )


   result = (
       run_isolation_forest(
           df,
           cfg,
       )
   )


   assert (
       result[
           "IF_STATUS"
       ]
       .eq(
           "SKIPPED"
       )
       .all()
   )


   assert (
       result[
           "IF_ANOMALY"
       ]
       .eq(
           False
       )
       .all()
   )




def test_isolation_forest_no_usable_features_skipped():
   cfg = MSConfig(
       iforest_min_train=2
   )


   data = {
       feature: [
           np.nan,
           np.nan,
       ]
       for feature in IF_FEATURES
   }


   data[
       "IS_IGNORED"
   ] = [
       False,
       False,
   ]


   df = pd.DataFrame(
       data
   )


   result = (
       run_isolation_forest(
           df,
           cfg,
       )
   )


   assert (
       result[
           "IF_STATUS"
       ]
       .eq(
           "SKIPPED"
       )
       .all()
   )


   assert (
       result[
           "IF_ANOMALY"
       ]
       .eq(
           False
       )
       .all()
   )




# ---------------------------------------------------------------------------
# Configuration wiring
# ---------------------------------------------------------------------------


def test_min_history_config_key_maps_to_iforest_min_train(
   tmp_path,
):
   config_path = (
       tmp_path
       / "matrix_spike_config.yaml"
   )


   config_path.write_text(
       """
matrix_spike:
 isolation_forest:
   min_history: 77
   contamination: 0.05
   n_estimators: 25
   random_state: 42
""".strip(),
       encoding="utf-8",
   )


   cfg = load_ms_config(
       str(
           config_path
       )
   )


   assert (
       cfg.iforest_min_train
       == 77
   )


   assert (
       cfg.iforest_n_estimators
       == 25
   )




def test_real_matrix_spike_config_file_is_wired_up():
   assert (
       DEFAULT_CONFIG_PATH.name
       == "matrix_spike_config.yaml"
   )


   assert (
       DEFAULT_CONFIG_PATH
       .parent
       .name
       == "config"
   )


   assert (
       DEFAULT_CONFIG_PATH.exists()
   )


   cfg = load_ms_config()


   assert (
       cfg.rolling_window
       == 10
   )


   assert (
       cfg.rolling_min_periods
       == 3
   )


   assert (
       cfg.mad_scale
       == pytest.approx(
           0.6745
       )
   )


   assert (
       cfg.robust_z_threshold
       == pytest.approx(
           3.0
       )
   )


   assert (
       cfg.drift_threshold
       == pytest.approx(
           0.5
       )
   )


   assert (
       cfg.extreme_score_threshold
       == pytest.approx(
           10.0
       )
   )


   assert (
       cfg.iforest_min_train
       == 20
   )


   assert (
       cfg.iforest_contamination
       == pytest.approx(
           0.05
       )
   )


   assert (
       cfg.iforest_n_estimators
       == 200
   )


   assert (
       cfg.iforest_random_state
       == 42
   )


   assert (
       cfg.drift_rolling_window
       == 10
   )


   assert (
       cfg.drift_consecutive_required
       == 3
   )


   assert (
       cfg.drift_boundary_level
       == pytest.approx(
           1.0
       )
   )


   assert (
       cfg.drift_trend_window
       == 10
   )


   assert (
       cfg.ts_rolling_window
       == 10
   )


   assert (
       cfg.ts_min_periods
       == 5
   )


   assert (
       cfg.ts_early_drift_fraction
       == pytest.approx(
           0.75
       )
   )


   assert (
       cfg.ts_persistence_windows
       == 4
   )


   assert (
       cfg.ts_strong_drift_fraction
       == pytest.approx(
           1.0
       )
   )


   assert (
       cfg.ts_strong_persistence_windows
       == 3
   )


   assert (
       cfg.output_dir
       == "ms_outputs"
   )




# ---------------------------------------------------------------------------
# End-to-end detector regression
# ---------------------------------------------------------------------------


def test_full_matrix_spike_pipeline_returns_stable_results():
   rows = []


   values = [
       99.0,
       100.0,
       101.0,
       100.5,
       99.5,
       100.0,
   ]


   for i, value in enumerate(
       values
   ):
       rows.append(
           _ms_row(
               JOB_CODE=(
                   "JOB_{}"
                   .format(
                       i + 1
                   )
               ),
               ANALYSED_DATE=(
                   pd.Timestamp(
                       "2024-01-01"
                   )
                   + pd.Timedelta(
                       days=i
                   )
               ),
               NUMERIC_FINAL_VALUE=value,
           )
       )


   # Same analytical type but MSD.
   # It must not enter the Matrix Spike input.
   rows.append(
       _ms_row(
           QC_TYPE="MSD",
           JOB_CODE="MSD_JOB",
           ANALYSED_DATE=(
               pd.Timestamp(
                   "2024-01-07"
               )
           ),
       )
   )


   raw = pd.DataFrame(
       rows
   )


   ms = filter_ms_records(
       raw
   )


   # Keep ML intentionally above this tiny
   # synthetic history so this regression
   # test remains deterministic.
   cfg = MSConfig(
       iforest_min_train=100
   )


   (
       results,
       drift_summary,
   ) = run_ms_detection(
       ms,
       cfg,
   )


   assert len(ms) == 6
   assert len(results) == 6


   assert (
       list(
           results.columns
       )
       == OUTPUT_COLUMNS
   )


   assert (
       len(
           drift_summary
       )
       == 1
   )


   assert (
       results[
           "CCLAS_MISMATCH"
       ].sum()
       == 0
   )


   assert (
       results[
           "RULE_FLAG"
       ]
       .eq(
           "Pass"
       )
       .all()
   )


   assert (
       results[
           "FINAL_RISK"
       ]
       .eq(
           "Low"
       )
       .all()
   )


   assert (
       results[
           "DETECTION_METHOD"
       ]
       .eq(
           "None"
       )
       .all()
   )


   assert (
       results[
           "IF_STATUS"
       ]
       .eq(
           "SKIPPED"
       )
       .all()
   )


   assert (
       drift_summary.loc[
           0,
           "drift_level",
       ]
       == "None"
   )

