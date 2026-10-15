"""Shared lossless CCLAS parsing, typed canonical records and conservative QC routing.

Source properties are retained verbatim as JSON; typed aliases never replace them.
Standard and Spike alone are insufficient evidence of a detector subtype.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

SCHEMA_VERSION = "cclas-qc-1"
PROPERTY_MAP = {'S_JobCode': 'Job',
 'S_JobName': 'JobName',
 'S_SampleCode': 'SampleCode',
 'S_SampleName': 'SampleName',
 'S_ClientSampleName': 'ClientSampleName',
 'S_DupOriginalClientSampleName': 'OriginalClientSampleName',
 'S_DupOriginalSampleCode': 'OriginalSampleCode',
 'S_DupOriginalSampleName': 'OriginalSampleName',
 'S_OriginalJobCode': 'OriginalJobCode',
 'S_OriginalSampleCode': 'SourceOriginalSampleCode',
 'S_PrimaryAnalyticalType': 'PrimaryType',
 'S_QcTypeCode': 'QCType',
 'S_SecondaryAnalyticalType': 'SecondaryAnalyticalType',
 'S_SecondaryQcTypeCode': 'SecondaryQcTypeCode',
 'S_SpikeOriginalSampleCode': 'SpikeOriginalSampleCode',
 'S_StandardCode': 'StandardCode',
 'S_StandardLotCode': 'StandardLotCode',
 'RSC_SchemeCode': 'Scheme',
 'RSC_SchemeName': 'SchemeName',
 'RSA_AnalyteCode': 'Analyte',
 'RSA_AnalyteName': 'AnalyteName',
 'SSA_UnitCode': 'Unit',
 'SSA_NumericFinalValue': 'Result',
 'PA_InternalMaxInclusive': 'UpperLimitInclusive',
 'PA_InternalMaxValue': 'UpperLimit',
 'PA_InternalMaxWarningInclusive': 'UpperWarningInclusive',
 'PA_InternalMaxWarningValue': 'UpperWarning',
 'PA_InternalMinInclusive': 'LowerLimitInclusive',
 'PA_InternalMinValue': 'LowerLimit',
 'PA_InternalMinWarningInclusive': 'LowerWarningInclusive',
 'PA_InternalMinWarningValue': 'LowerWarning',
 'PA_InternalTargetValue': 'Target',
 'SA_StatDetectionLimitDuplicateValue': 'DetectionLimitDuplicate',
 'SA_LimitRepeatabilityDuplicateValue': 'RepeatabilityLimitDuplicate',
 'SA_StatDetectionLimitValue': 'DetectionLimitRepeat',
 'SA_LimitRepeatabilityValue': 'RepeatabilityLimitRepeat',
 'SSA_LimitStatus': 'CCLAS_LimitStatus',
 'SSA_PrecisionStatus': 'CCLAS_PrecisionStatus',
 'SSA_SpecificationStatus': 'CCLAS_SpecificationStatus',
 'SSA_StandardStatus': 'CCLAS_StandardStatus',
 'SSA_Status': 'CCLAS_Status',
 'SSA_UserQcStatus': 'CCLAS_UserQcStatus',
 'SSA_UserWorkflowStatus': 'CCLAS_UserWorkflowStatus',
 'SSA_WorkflowStatus': 'CCLAS_WorkflowStatus',
 'SSA_InstrumentCode': 'InstrumentCode',
 'SSA_AnalysedByUserCode': 'AnalysedByUserCode',
 'RSC_SchemeVersion': 'SchemeVersion',
 'RSC_SchemeVersionCode': 'SchemeVersion',
 'SSA_AnalysedDate': 'AnalysisDate',
 'SSA_AnalysisDate': 'AnalysisDate',
 'SSA_ResultCode': 'ResultCode',
 'SSA_ResultId': 'ResultID',
 'SSA_SpikeAddedValue': 'SpikeAddedValue',
 'SSA_Recovery': 'Recovery'}
NUMERIC_COLUMNS = [
    "Result", "Target", "LowerLimit", "LowerWarning", "UpperWarning", "UpperLimit",
    "DetectionLimitDuplicate", "RepeatabilityLimitDuplicate", "DetectionLimitRepeat",
    "RepeatabilityLimitRepeat", "SpikeAddedValue", "Recovery", "ParentValue",
]
TEXT_COLUMNS = [
    "Job", "SampleCode", "PrimaryType", "QCType", "SecondaryAnalyticalType",
    "SecondaryQcTypeCode", "StandardCode", "StandardLotCode", "Scheme", "Analyte", "Unit",
    "OriginalSampleCode", "SpikeOriginalSampleCode", "SourceOriginalSampleCode",
    "LowerLimitInclusive", "UpperLimitInclusive", "LowerWarningInclusive", "UpperWarningInclusive",
    "CCLAS_StandardStatus", "CCLAS_LimitStatus", "CCLAS_PrecisionStatus",
]
QC_ALIASES = {
    "BLK": "BLANK", "BLANK": "BLANK", "SRMS": "SRMS", "CRM": "SRMS",
    "CERTIFIED_REFERENCE_MATERIAL": "SRMS", "STANDARD_REFERENCE_MATERIAL": "SRMS",
    "CONTROL_LCS": "CONTROL", "CONTROL": "CONTROL", "LCS": "CONTROL", "LABORATORY_CONTROL_SAMPLE": "CONTROL",
    "DUP": "DUPLICATE", "DUPLICATE": "DUPLICATE", "REP": "REPLICATE",
    "REPEAT": "REPLICATE", "REPLICATE": "REPLICATE", "MS": "MATRIX_SPIKE",
    "MATRIX_SPIKE": "MATRIX_SPIKE", "MSD": "MATRIX_SPIKE_DUPLICATE",
    "MATRIX_SPIKE_DUPLICATE": "MATRIX_SPIKE_DUPLICATE",
}


def token(value: object) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip().upper().replace(" ", "_").replace("/", "_")


def classify_qc(row: pd.Series) -> tuple[str, str, str]:
    fields = ["QCType", "PrimaryType", "SecondaryQcTypeCode", "SecondaryAnalyticalType"]
    evidence = [(key, token(row.get(key))) for key in fields]
    classes = {QC_ALIASES[value] for _, value in evidence if value in QC_ALIASES}
    if classes and token(row.get("QCType")) in {"UNK", "UNKNOWN"}:
        return "UNKNOWN", "UNRESOLVED", "Conflicting customer QC code and explicit detector subtype"
    # A generic parent type can coexist with an explicit subtype (e.g. STD + LCS).
    if len(classes) == 1:
        kind = classes.pop()
        generic = {value for _, value in evidence}
        incompatible = ((bool(generic & {"STD", "STANDARD"}) and kind not in {"SRMS", "CONTROL"})
                        or (bool(generic & {"SPIKE", "SPK"}) and kind not in {"MATRIX_SPIKE", "MATRIX_SPIKE_DUPLICATE"}))
        if incompatible:
            return "UNKNOWN", "UNRESOLVED", "Conflicting analytical type and QC subtype"
        return kind, "HIGH", "; ".join(f"{key}={value}" for key, value in evidence if value in QC_ALIASES)
    if len(classes) > 1:
        # Explicit MSD with primary Duplicate is compatible, but MS + DUP alone is ambiguous.
        if classes == {"MATRIX_SPIKE_DUPLICATE", "DUPLICATE"}:
            return "MATRIX_SPIKE_DUPLICATE", "HIGH", "Explicit MSD with Duplicate analytical type"
        return "UNKNOWN", "UNRESOLVED", "Conflicting QC metadata: " + "; ".join(f"{k}={v}" for k, v in evidence if v)
    return "UNKNOWN", "UNRESOLVED", "No explicit detector subtype: " + "; ".join(f"{k}={v}" for k, v in evidence if v)


# Confirmed CCLAS source families are separate from detector subtype inference.
SOURCE_TYPE_ALIASES = {
    "UNK": "UNK", "UNKNOWN": "UNK", "BLK": "BLK", "BLANK": "BLK",
    "STD": "STD", "STANDARD": "STD", "SPK": "SPK", "SPIKE": "SPK",
    "DUP": "DUP", "DUPLICATE": "DUP", "REP": "REP", "REPEAT": "REP", "REPLICATE": "REP",
    **{key: {"SRMS": "STD", "CONTROL": "STD", "BLANK": "BLK", "DUPLICATE": "DUP",
             "REPLICATE": "REP", "MATRIX_SPIKE": "SPK", "MATRIX_SPIKE_DUPLICATE": "SPK"}[value]
       for key, value in QC_ALIASES.items()},
}


def source_classification(row: pd.Series, subtype: str, subtype_reason: str) -> dict[str, str]:
    """Normalize a source family without treating a customer sample as unresolved QC.

    Empty QC code + explicit PrimaryType=UNKNOWN is UNK. Completely missing
    metadata remains unresolved. Original values are never replaced by this code.
    """
    qc, primary = token(row.get("QCType")), token(row.get("PrimaryType"))
    qc_family, primary_family = SOURCE_TYPE_ALIASES.get(qc), SOURCE_TYPE_ALIASES.get(primary)
    family = qc_family or primary_family
    reason = f"QCType={qc}" if qc_family else f"PrimaryType={primary}"
    subtype_family = SOURCE_TYPE_ALIASES.get(subtype) if subtype != "UNKNOWN" else None
    if subtype_family and (not family or (family == "UNK" and qc_family != "UNK")):
        family = subtype_family
        reason = "Explicit subtype metadata"
    conflict = subtype_reason.startswith("Conflicting") or (qc_family == "UNK" and subtype != "UNKNOWN")
    if qc_family and primary_family and qc_family != primary_family:
        # UNKNOWN primary type is a fallback when an explicit QC type is supplied.
        compatible = (primary_family == "UNK" and qc_family != "UNK") or (qc in {"MSD", "MATRIX_SPIKE_DUPLICATE"} and primary_family == "DUP")
        conflict |= not compatible
    if conflict:
        family, status, detector, reason = "UNRESOLVED", "CONFLICT", "UNRESOLVED", f"Conflicting source metadata (QCType={qc}, PrimaryType={primary}); {subtype_reason}"
    elif family == "UNK":
        status, detector = "CUSTOMER_SAMPLE", "NONE"
        reason += "; customer reference sample; no QC detector required"
    elif family in {"STD", "SPK"} and subtype == "UNKNOWN":
        status = "SUBTYPE_UNRESOLVED"
        detector = "STANDARD" if family == "STD" else "SPIKE"
        reason += "; confirmed source family; " + ("SRMS versus Control" if family == "STD" else "MS versus MSD") + " subtype not supplied"
    elif subtype != "UNKNOWN":
        status, detector = "CONFIRMED", subtype
        reason += "; " + subtype_reason
    else:
        family, status, detector = family or "UNRESOLVED", "UNRESOLVED", "UNRESOLVED"
        reason = "Missing or unrecognized source sample metadata; " + subtype_reason
    return {
        "original_cclas_sample_type": str(row.get("QCType", "")),
        "source_sample_type": family,
        "detector_type": detector,
        "detector_subtype": "NOT_APPLICABLE" if status == "CUSTOMER_SAMPLE" else ("UNRESOLVED" if subtype == "UNKNOWN" or conflict else subtype),
        "classification_status": status,
        "classification_reason": reason,
    }


def is_cclas_object_property_tsv(path_str: str | Path) -> bool:
    path = Path(path_str).expanduser()
    if path.suffix.lower() != ".tsv":
        return False
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return next(csv.reader(handle, delimiter="\t"), []) == ["OBJECT", "PROPERTY", "VALUE"]


def canonicalize(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    for col in TEXT_COLUMNS:
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    issues = [[] for _ in range(len(df))]
    for col in NUMERIC_COLUMNS + ["AnalysisDate"]:
        if col not in df:
            df[col] = ""
        raw = df.get(f"Raw_{col}", df[col])
        converted = pd.to_datetime(raw, errors="coerce", utc=True, format="mixed") if col == "AnalysisDate" else pd.to_numeric(raw, errors="coerce").astype(float)
        invalid = raw.notna() & raw.astype(str).str.strip().ne("") & converted.isna()
        if col != "AnalysisDate":
            invalid |= converted.notna() & ~np.isfinite(converted)
            converted = converted.mask(~np.isfinite(converted))
        for i, bad in enumerate(invalid):
            if bad:
                issues[i].append(f"malformed:{col}")
        # Preserve exact input string even for malformed values and subsequent TSV round trips.
        if f"Raw_{col}" not in df:
            df[f"Raw_{col}"] = raw.fillna("").astype(str)
        df[col] = converted
    for col in ["Job", "SampleCode", "Scheme", "Analyte", "Unit", "Result"]:
        missing = df[col].isna() | df[col].astype(str).str.strip().eq("")
        for i, bad in enumerate(missing):
            if bad:
                issues[i].append(f"missing:{col}")
    classification = [classify_qc(row) for _, row in df.iterrows()]
    for i, col in enumerate(["DetectorQCType", "ClassificationConfidence", "ClassificationReason"]):
        df[col] = [value[i] for value in classification]
    source_classes = [source_classification(row, value[0], value[2])
                      for (_, row), value in zip(df.iterrows(), classification)]
    for column in ["original_cclas_sample_type", "source_sample_type", "detector_type", "detector_subtype",
                   "classification_status", "classification_reason"]:
        df[column] = [value[column] for value in source_classes]
    conflict = df["classification_status"].eq("CONFLICT")
    df.loc[conflict, "DetectorQCType"] = "UNKNOWN"
    df.loc[conflict, "ClassificationConfidence"] = "UNRESOLVED"
    customer = df["classification_status"].eq("CUSTOMER_SAMPLE")
    df.loc[customer, "DetectorQCType"] = "NOT_APPLICABLE"
    df.loc[customer, "ClassificationConfidence"] = "HIGH"
    df["ClassificationReason"] = df["classification_reason"]
    # Ordinals are source traceability only, never laboratory chronology.
    df["SourceOrderMeaning"] = "FILE_ORDER_ONLY"
    df["SourceAnalyticalType"] = df["PrimaryType"]
    df["SourceQCType"] = df["QCType"]
    df["DetectionLimit"] = np.where(df["DetectorQCType"].eq("DUPLICATE"), df["DetectionLimitDuplicate"], df["DetectionLimitRepeat"])
    df["RepeatabilityLimit"] = np.where(df["DetectorQCType"].eq("DUPLICATE"), df["RepeatabilityLimitDuplicate"], df["RepeatabilityLimitRepeat"])
    previous = df.get("ValidationIssues", pd.Series("", index=df.index)).fillna("").astype(str)
    df["ValidationIssues"] = [";".join(dict.fromkeys([x for x in old.split(";") if x] + new)) for old, new in zip(previous, issues)]
    df["CanonicalSchemaVersion"] = SCHEMA_VERSION
    if "RecordID" in df and (df["RecordID"].fillna("").eq("").any() or df["RecordID"].duplicated().any()):
        raise ValueError("Duplicate canonical RecordID; refusing to merge or drop source results")
    return df


def parse_cclas_tsv(path_str: str | Path) -> pd.DataFrame:
    path = Path(path_str).expanduser().resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records, sample_rows, result_rows, metadata = [], [], [], []
    sample, result, scheme = {}, {}, {}
    sample_block, result_line = 0, 0

    def flush() -> None:
        if not result_rows:
            return
        if not sample_rows:
            raise ValueError(f"Result at line {result_line} has no VSAMPLE parent")
        records.append({**sample, **result, "SampleBlock": sample_block,
                        "ResultRow": len(records) + 1, "SourceLine": result_line,
                        "RecordID": f"{digest}:{len(records)+1}", "SourceFile": path.name,
                        "SourceSHA256": digest,
                        "RawSampleProperties": json.dumps(sample_rows, ensure_ascii=False),
                        "RawResultProperties": json.dumps(result_rows, ensure_ascii=False)})

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["OBJECT", "PROPERTY", "VALUE"]:
            raise ValueError("Expected OBJECT, PROPERTY, VALUE header")
        for row in reader:
            if None in row or any(row.get(key) is None for key in ("OBJECT", "PROPERTY", "VALUE")):
                raise ValueError(f"Malformed TSV at line {reader.line_num}")
            obj, prop = row["OBJECT"].strip().rstrip(":"), row["PROPERTY"].strip().rstrip("=")
            value = row["VALUE"]
            raw = {"line": reader.line_num, **row}
            mapped = PROPERTY_MAP.get(prop, prop)
            if obj == "VSAMPLE":
                if result_rows or (prop == "S_JobCode" and "Job" in sample) or (prop == "S_SampleCode" and "SampleCode" in sample):
                    flush()
                    sample, result, scheme, sample_rows, result_rows = {}, {}, {}, [], []
                    sample_block += 1
                if not sample_block:
                    sample_block = 1
                sample[mapped] = value
                sample_rows.append(raw)
            elif obj == "VSSA":
                if (prop == "RSC_SchemeCode" and "Scheme" in result and ("Analyte" in result or "Result" in result)) or (prop == "RSA_AnalyteCode" and "Analyte" in result):
                    flush()
                    result = {} if prop == "RSC_SchemeCode" else scheme.copy()
                    result_rows = []
                if not result_rows:
                    result_line = reader.line_num
                result[mapped] = value
                result_rows.append(raw)
                if prop.startswith("RSC_"):
                    scheme[mapped] = value
            else:
                metadata.append(raw)
    flush()
    for row in records:
        row["RawOtherProperties"] = json.dumps(metadata, ensure_ascii=False)
    return canonicalize(pd.DataFrame(records))


def load_qc_data(path_str: str | Path, sheet_name: str | None = None, max_rows: int | None = None) -> pd.DataFrame:
    """Read all rows before truncation so a max_rows limit cannot split a result block."""
    path = Path(path_str).expanduser().resolve()
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be nonnegative")
    if is_cclas_object_property_tsv(path):
        df = parse_cclas_tsv(path)
    else:
        if path.suffix.lower() in {".csv", ".tsv"}:
            df = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", dtype=str, keep_default_na=False)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0, dtype=str, keep_default_na=False)
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if len(members) != 1:
                    raise ValueError("Expected exactly one CSV in archive")
                with archive.open(members[0]) as handle:
                    df = pd.read_csv(handle, dtype=str, keep_default_na=False)
        else:
            raise ValueError(f"Unsupported QC input: {path.suffix}")
        if "CanonicalSchemaVersion" in df:
            if not df["CanonicalSchemaVersion"].eq(SCHEMA_VERSION).all():
                raise ValueError("Unsupported canonical schema version")
            required = {"RecordID", "SourceSHA256", "RawSampleProperties", "RawResultProperties"}
            if not required.issubset(df):
                raise ValueError(f"Missing normalized provenance fields: {sorted(required - set(df))}")
            # Structural ordinals have a fixed type after read_csv as well.
            for col in ["SampleBlock", "ResultRow", "SourceLine"]:
                if col in df:
                    df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
            df = canonicalize(df)
        elif {"ANALYTICAL_TYPE", "NUMERIC_FINAL_VALUE"}.issubset(df):
            # Historical ResultSet schema is retained for the existing SRMS backtest.
            # The shared loader adds canonical aliases without discarding source columns.
            aliases = {"ANALYTICAL_TYPE":"PrimaryType", "QC_TYPE":"QCType", "STD_CODE":"StandardCode",
                       "STD_LOT_CODE":"StandardLotCode", "JOB_CODE":"Job", "SAMPLE_CODE":"SampleCode",
                       "NUMERIC_FINAL_VALUE":"Result", "ANALYSED_DATE":"AnalysisDate", "SCHEME_CODE":"Scheme",
                       "ANALYTE_CODE":"Analyte", "UNIT_CODE":"Unit", "INTERNAL_TARGET_VALUE":"Target",
                       "INTERNAL_MIN_VALUE":"LowerLimit", "INTERNAL_MAX_VALUE":"UpperLimit",
                       "INTERNAL_MIN_WARNING_VALUE":"LowerWarning", "INTERNAL_MAX_WARNING_VALUE":"UpperWarning",
                       "INTERNAL_MIN_INCLUSIVE":"LowerLimitInclusive", "INTERNAL_MAX_INCLUSIVE":"UpperLimitInclusive",
                       "INTERNAL_MIN_WARNING_INCLUSIVE":"LowerWarningInclusive", "INTERNAL_MAX_WARNING_INCLUSIVE":"UpperWarningInclusive",
                       "STANDARD_STATUS":"CCLAS_StandardStatus", "SCHEME_VERSION":"SchemeVersion"}
            raw_rows = [json.dumps(row, ensure_ascii=False) for row in df.to_dict("records")]
            for old, new in aliases.items():
                if old in df:
                    df[new] = df[old]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            df["RecordID"] = [f"{digest}:{i+1}" for i in range(len(df))]
            df["SourceSHA256"], df["SourceFile"] = digest, path.name
            df["RawSampleProperties"], df["RawResultProperties"] = "[]", raw_rows
            df = canonicalize(df)
        elif {"SampleCode", "Scheme", "Analyte", "Result"}.issubset(df):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            df["RawResultProperties"] = [json.dumps(row) for row in df.to_dict("records")]
            df["RawSampleProperties"] = "[]"
            df["RecordID"] = [f"{digest}:{i+1}" for i in range(len(df))]
            df["SourceSHA256"], df["SourceFile"] = digest, path.name
            df = canonicalize(df)
        else:
            raise ValueError("Unsupported QC schema; expected CCLAS triples, normalized QC or ResultSet columns")
    return df.head(max_rows).copy() if max_rows is not None else df


def write_normalized(data: pd.DataFrame, output: str | Path, *, source: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    if destination == Path(source).expanduser().resolve():
        raise ValueError("Refusing to overwrite the source dataset")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, sep="\t", index=False, float_format="%.17g")
    return destination
