"""
Tests for src/data_loader.py:
- load_qc_data()'s column_config.csv-driven renaming (uppercase source ->
  lowercase internal names), including the INSTRUMENT_ID/INSTRUMENT_CODE
  cross-file alias (ResultSet.csv vs QC_Sample_Data.csv name the same
  logical field differently).
- _derive_precision_metrics()'s rpd/mean_conc derivation, ported from
  notebooks/REP__distribution_analysis.ipynb /
  DUP__distribution_analysis.ipynb's build_replicate_model()/
  build_duplicate_model() (identical formula in both).
"""

from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_qc_data, _derive_precision_metrics


def _write_config(tmp_path: Path, rows: list) -> Path:
    config_path = tmp_path / "column_config.csv"
    pd.DataFrame(rows, columns=["source_column", "internal_column", "dtype", "required"]).to_csv(
        config_path, index=False
    )
    return config_path


def _write_csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# ── load_qc_data(): column mapping ──────────────────────────────────────────

def test_load_qc_data_renames_uppercase_source_to_lowercase_internal(tmp_path):
    config_path = _write_config(tmp_path, [
        ("ANALYTE_CODE", "analyte_code", "str", "true"),
        ("NUMERIC_FINAL_VALUE", "measured_value", "float", "true"),
        ("ANALYSED_DATE", "analysed_date", "datetime", "false"),
    ])
    raw = pd.DataFrame({
        "ANALYTE_CODE": ["Cu", "Zn"],
        "NUMERIC_FINAL_VALUE": [1.5, 2.5],
        "ANALYSED_DATE": ["2024-01-01", "2024-01-02"],
    })
    data_path = _write_csv(tmp_path, "raw.csv", raw)

    df = load_qc_data(data_path=data_path, config_path=config_path)

    assert list(df.columns) == ["analyte_code", "measured_value", "analysed_date"]
    assert df["measured_value"].tolist() == [1.5, 2.5]
    assert df["analyte_code"].tolist() == ["Cu", "Zn"]


def test_instrument_id_and_instrument_code_both_map_to_instrument_id(tmp_path):
    # ResultSet.csv names this field INSTRUMENT_ID; QC_Sample_Data.csv names
    # the same logical field INSTRUMENT_CODE. Both must land on the same
    # internal column, whichever is actually present in the source file.
    config_path = _write_config(tmp_path, [
        ("ANALYTE_CODE", "analyte_code", "str", "true"),
        ("INSTRUMENT_ID", "instrument_id", "str", "false"),
        ("INSTRUMENT_CODE", "instrument_id", "str", "false"),
    ])

    resultset_style = _write_csv(tmp_path, "resultset_style.csv", pd.DataFrame({
        "ANALYTE_CODE": ["Cu"], "INSTRUMENT_ID": ["INS-1"],
    }))
    df1 = load_qc_data(data_path=resultset_style, config_path=config_path)
    assert df1["instrument_id"].tolist() == ["INS-1"]

    qc_sample_style = _write_csv(tmp_path, "qc_sample_style.csv", pd.DataFrame({
        "ANALYTE_CODE": ["Zn"], "INSTRUMENT_CODE": ["INS-2"],
    }))
    df2 = load_qc_data(data_path=qc_sample_style, config_path=config_path)
    assert df2["instrument_id"].tolist() == ["INS-2"]


# ── _derive_precision_metrics(): rpd/mean_conc derivation ───────────────────

def test_derive_precision_metrics_matches_notebook_formula():
    # MEAN_CONC = (NUMERIC_FINAL_VALUE + PARENT_NUMERIC_FINAL_VALUE) / 2
    # RPD = |diff| / MEAN_CONC * 100, rounded to 2 d.p.
    df = pd.DataFrame({
        "measured_value": [110.0, 100.0],
        "parent_value": [100.0, 100.0],
    })

    out = _derive_precision_metrics(df)

    assert out["mean_conc"].tolist() == [105.0, 100.0]
    assert out["rpd"].iloc[0] == pytest.approx(9.52, abs=0.01)
    assert out["rpd"].iloc[1] == 0.0


def test_derive_precision_metrics_guards_zero_mean_conc():
    df = pd.DataFrame({"measured_value": [0.0], "parent_value": [0.0]})
    out = _derive_precision_metrics(df)
    assert out["mean_conc"].iloc[0] == 0.0
    assert pd.isna(out["rpd"].iloc[0])


def test_derive_precision_metrics_noop_without_parent_value_column():
    # Blank/Standard/Spike rows have no meaningful parent_value -- a frame
    # that lacks the column entirely must not raise and must not gain
    # rpd/mean_conc columns.
    df = pd.DataFrame({"measured_value": [1.0, 2.0]})
    out = _derive_precision_metrics(df)
    assert "rpd" not in out.columns
    assert "mean_conc" not in out.columns


def test_load_qc_data_derives_rpd_and_mean_conc_end_to_end(tmp_path):
    config_path = _write_config(tmp_path, [
        ("ANALYTE_CODE", "analyte_code", "str", "true"),
        ("NUMERIC_FINAL_VALUE", "measured_value", "float", "true"),
        ("PARENT_NUMERIC_FINAL_VALUE", "parent_value", "float", "false"),
    ])
    raw = pd.DataFrame({
        "ANALYTE_CODE": ["Cu", "Zn"],
        "NUMERIC_FINAL_VALUE": [110.0, 50.0],
        "PARENT_NUMERIC_FINAL_VALUE": [100.0, 50.0],
    })
    data_path = _write_csv(tmp_path, "raw.csv", raw)

    df = load_qc_data(data_path=data_path, config_path=config_path)

    assert df["mean_conc"].tolist() == [105.0, 50.0]
    assert df["rpd"].iloc[0] == pytest.approx(9.52, abs=0.01)
    assert df["rpd"].iloc[1] == 0.0
