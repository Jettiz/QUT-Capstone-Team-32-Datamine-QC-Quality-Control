from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.detectors import DetectionResultSet, get_detector, registered_detectors
from src.detectors.srms_detector import is_cclas_object_property_tsv, load_srms_config


def load_pipeline_input(path_str: str, sheet_name: str | None = None, max_rows: int | None = None) -> pd.DataFrame:
    path = Path(path_str).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False, nrows=max_rows)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False, nrows=max_rows)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name, nrows=max_rows)
    raise ValueError(f"Unsupported input type for common pipeline: {path.suffix}")


def run_common_pipeline(
    input_path: str,
    *,
    detector_name: str | None = None,
    history_path: str | None = None,
    sheet_name: str | None = None,
    config_path: str | None = None,
    max_rows: int | None = None,
) -> dict[str, DetectionResultSet]:
    config = load_srms_config(config_path)
    source_path = str(Path(input_path).expanduser().resolve())
    data = None if is_cclas_object_property_tsv(source_path) else load_pipeline_input(source_path, sheet_name, max_rows)
    detectors = [get_detector(detector_name, config=config)] if detector_name else registered_detectors(config=config)

    result_sets: dict[str, DetectionResultSet] = {}
    for detector in detectors:
        if detector.can_handle(data, source_path=source_path):
            result_sets[detector.name] = detector.detect(
                data,
                source_path=source_path,
                history_path=history_path,
                sheet_name=sheet_name,
                max_rows=max_rows,
            )
    return result_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Common QC anomaly detection pipeline.")
    parser.add_argument("--input", required=True, help="Path to CSV, TSV, XLS, or XLSX input.")
    parser.add_argument("--detector", default=None, help="Optional detector name. Defaults to all registered detectors.")
    parser.add_argument("--history", default=None, help="Optional historical source for detectors that use prior observations.")
    parser.add_argument("--sheet", default=None, help="Optional worksheet for Excel inputs and SRMS historical workbooks.")
    parser.add_argument("--config", default=None, help="Optional detector configuration YAML.")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_sets = run_common_pipeline(
        args.input,
        detector_name=args.detector,
        history_path=args.history,
        sheet_name=args.sheet,
        config_path=args.config,
        max_rows=args.max_rows,
    )
    if not result_sets:
        print("No registered detectors accepted the input.")
        return
    for name, result_set in result_sets.items():
        anomaly_count = sum(result.anomaly_detected for result in result_set.anomalies)
        print(f"{name}: processed {len(result_set.results):,} records; anomalies={anomaly_count:,}")


if __name__ == "__main__":
    main()
