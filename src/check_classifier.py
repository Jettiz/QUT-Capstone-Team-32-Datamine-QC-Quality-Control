"""
check_classifier.py
===================
Classifies a loaded QC dataset by ANALYTICAL_TYPE and QC_TYPE, then
routes each subset to its dedicated anomaly detector.

Routing logic requires BOTH columns because ANALYTICAL_TYPE alone is
insufficient to distinguish all six sample types:

    ANALYTICAL_TYPE   QC_TYPE   -> Sample Type
    ─────────────────────────────────────────────
    Blank             BLK       -> blank_detector
    Standard          STD       -> standard_detector
    Spike             MS        -> matrix_spike_detector
    Replicate         REP       -> replicate_detector
    Duplicate         DUP       -> duplicate_detector
    Replicate         MSD       -> matrix_spike_dup_detector

Note: MSD records share ANALYTICAL_TYPE = Replicate with regular
replicates. QC_TYPE = MSD is the only way to distinguish them.

Usage
-----
    from check_classifier import classify_and_route
    from data_loader import load_qc_data

    df = load_qc_data("ResultSet.csv")
    results = classify_and_route(df)
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | check_classifier | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sample type identifiers
# ---------------------------------------------------------------------------

TYPE_BLANK              = "blank"
TYPE_STANDARD           = "standard"
TYPE_DUPLICATE          = "duplicate"
TYPE_REPLICATE          = "replicate"
TYPE_MATRIX_SPIKE       = "matrix_spike"
TYPE_MATRIX_SPIKE_DUP   = "matrix_spike_duplicate"

ALL_TYPES = [
    TYPE_BLANK,
    TYPE_STANDARD,
    TYPE_DUPLICATE,
    TYPE_REPLICATE,
    TYPE_MATRIX_SPIKE,
    TYPE_MATRIX_SPIKE_DUP,
]

# ---------------------------------------------------------------------------
# Routing rules
# Each rule is (analytical_type, qc_type) — both normalised to lowercase.
# qc_type = None means match any QC_TYPE value (used when QC_TYPE is absent).
# ---------------------------------------------------------------------------

ROUTING_RULES: dict[str, tuple[str, str | None]] = {
    TYPE_BLANK:            ("blank",     "blk"),
    TYPE_STANDARD:         ("standard",  "std"),
    TYPE_MATRIX_SPIKE:     ("spike",     "ms"),
    TYPE_REPLICATE:        ("replicate", "rep"),
    TYPE_DUPLICATE:        ("duplicate", "dup"),
    TYPE_MATRIX_SPIKE_DUP: ("replicate", "msd"),
}

# ---------------------------------------------------------------------------
# Grouping columns per sample type
# Detectors should call get_grouping_columns() rather than hardcoding these.
# ---------------------------------------------------------------------------

GROUPING_COLUMNS: dict[str, list[str]] = {
    TYPE_BLANK: [
        "SCHEME_CODE",
        "ANALYTE_CODE",
        "UNIT_CODE",
        "INSTRUMENT_ID",
    ],
    TYPE_STANDARD: [
        "STD_LOT_CODE",
        "SCHEME_CODE",
        "ANALYTE_CODE",
        "UNIT_CODE",
    ],
    TYPE_DUPLICATE: [
        "SCHEME_CODE",
        "ANALYTE_CODE",
        "UNIT_CODE",
        "PARENT_NUMERIC_FINAL_VALUE",
    ],
    TYPE_REPLICATE: [
        "SCHEME_CODE",
        "ANALYTE_CODE",
        "UNIT_CODE",
        "PARENT_NUMERIC_FINAL_VALUE",
    ],
    TYPE_MATRIX_SPIKE: [
        "SCHEME_CODE",
        "ANALYTE_CODE",
        "UNIT_CODE",
        "PARENT_NUMERIC_FINAL_VALUE",
        "STD_LOT_CODE",
    ],
    TYPE_MATRIX_SPIKE_DUP: [
        "SCHEME_CODE",
        "ANALYTE_CODE",
        "UNIT_CODE",
        "PARENT_NUMERIC_FINAL_VALUE",
        "STD_LOT_CODE",
    ],
}

# ---------------------------------------------------------------------------
# Required columns per sample type — validated before routing
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: dict[str, list[str]] = {
    TYPE_BLANK: [
        "NUMERIC_FINAL_VALUE",
        "ANALYSED_DATE",
        "SCHEME_CODE",
        "ANALYTE_CODE",
        "UNIT_CODE",
    ],
    TYPE_STANDARD: [
        "NUMERIC_FINAL_VALUE",
        "INTERNAL_TARGET_VALUE",
        "ANALYSED_DATE",
        "SCHEME_CODE",
        "ANALYTE_CODE",
        "UNIT_CODE",
        "STD_LOT_CODE",
    ],
    TYPE_DUPLICATE: [
        "NUMERIC_FINAL_VALUE",
        "PARENT_NUMERIC_FINAL_VALUE",
        "SCHEME_CODE",
        "ANALYTE_CODE",
        "UNIT_CODE",
    ],
    TYPE_REPLICATE: [
        "NUMERIC_FINAL_VALUE",
        "PARENT_NUMERIC_FINAL_VALUE",
        "SCHEME_CODE",
        "ANALYTE_CODE",
        "UNIT_CODE",
    ],
    TYPE_MATRIX_SPIKE: [
        "NUMERIC_FINAL_VALUE",
        "PARENT_NUMERIC_FINAL_VALUE",
        "INTERNAL_TARGET_VALUE",
        "SCHEME_CODE",
        "ANALYTE_CODE",
        "UNIT_CODE",
    ],
    TYPE_MATRIX_SPIKE_DUP: [
        "NUMERIC_FINAL_VALUE",
        "PARENT_NUMERIC_FINAL_VALUE",
        "SCHEME_CODE",
        "ANALYTE_CODE",
        "UNIT_CODE",
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_and_route(
    df: pd.DataFrame,
    detectors: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] | None = None,
) -> dict[str, pd.DataFrame | None]:
    """
    Classify a QC DataFrame by ANALYTICAL_TYPE + QC_TYPE and route each
    subset to its dedicated anomaly detector.

    Parameters
    ----------
    df:
        Cleaned QC DataFrame. Must contain ANALYTICAL_TYPE.
        QC_TYPE is used when present to disambiguate Replicate vs MSD
        and to confirm other sample type assignments.
    detectors:
        Optional dict mapping sample type strings to detector functions.
        Each function receives a filtered DataFrame and returns a results
        DataFrame. If None, filtered subsets are returned without detection.

    Returns
    -------
    dict mapping each sample type to its detector output, filtered subset,
    or None if the type was not present or failed validation.
    """
    _check_required_columns(df)

    df = df.copy()
    df["_analytical_type_norm"] = (
        df["ANALYTICAL_TYPE"].astype(str).str.strip().str.lower()
    )

    has_qc_type = "QC_TYPE" in df.columns
    if has_qc_type:
        df["_qc_type_norm"] = (
            df["QC_TYPE"].astype(str).str.strip().str.lower()
        )
        log.info("QC_TYPE column present — using dual-column routing.")
    else:
        log.warning(
            "QC_TYPE column not found. Routing by ANALYTICAL_TYPE only. "
            "Replicate and Matrix Spike Duplicate cannot be distinguished — "
            "all Replicate records will be routed to replicate_detector."
        )

    _log_type_summary(df, has_qc_type)

    results = {}

    for sample_type in ALL_TYPES:
        subset = _extract_subset(df, sample_type, has_qc_type)

        if subset is None:
            results[sample_type] = None
            continue

        if not _validate_columns(subset, sample_type):
            results[sample_type] = None
            continue

        log.info(
            "Routing %-30s %d rows | grouping by: %s",
            sample_type, len(subset), GROUPING_COLUMNS[sample_type],
        )

        clean_subset = subset.drop(
            columns=[c for c in ["_analytical_type_norm", "_qc_type_norm"]
                     if c in subset.columns]
        )

        if detectors and sample_type in detectors:
            try:
                result = detectors[sample_type](clean_subset)
                log.info(
                    "%s detector returned %d rows.", sample_type, len(result)
                )
                results[sample_type] = result
            except Exception as exc:
                log.error(
                    "%s detector raised an error: %s", sample_type, exc
                )
                results[sample_type] = None
        else:
            log.info(
                "No detector registered for %s — returning filtered subset.",
                sample_type,
            )
            results[sample_type] = clean_subset

    return results


def get_subset(df: pd.DataFrame, sample_type: str) -> pd.DataFrame | None:
    """
    Extract the subset for a single sample type without routing.
    Useful for testing individual detectors in isolation.
    """
    df = df.copy()
    df["_analytical_type_norm"] = (
        df["ANALYTICAL_TYPE"].astype(str).str.strip().str.lower()
    )
    has_qc_type = "QC_TYPE" in df.columns
    if has_qc_type:
        df["_qc_type_norm"] = (
            df["QC_TYPE"].astype(str).str.strip().str.lower()
        )
    subset = _extract_subset(df, sample_type, has_qc_type)
    if subset is not None:
        return subset.drop(
            columns=[c for c in ["_analytical_type_norm", "_qc_type_norm"]
                     if c in subset.columns]
        )
    return None


def get_grouping_columns(sample_type: str) -> list[str]:
    """
    Return the grouping columns for a given sample type.
    Detectors should call this rather than hardcoding their own lists.
    """
    if sample_type not in GROUPING_COLUMNS:
        raise ValueError(
            f"Unknown sample type: '{sample_type}'. "
            f"Expected one of: {ALL_TYPES}"
        )
    return GROUPING_COLUMNS[sample_type]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_required_columns(df: pd.DataFrame) -> None:
    if "ANALYTICAL_TYPE" not in df.columns:
        raise ValueError(
            "DataFrame is missing ANALYTICAL_TYPE column. "
            "Ensure data_loader.load_qc_data() was used to load the file."
        )


def _extract_subset(
    df: pd.DataFrame,
    sample_type: str,
    has_qc_type: bool,
) -> pd.DataFrame | None:
    analytical_type, qc_type = ROUTING_RULES[sample_type]

    if has_qc_type and qc_type is not None:
        mask = (
            (df["_analytical_type_norm"] == analytical_type) &
            (df["_qc_type_norm"] == qc_type)
        )
    else:
        mask = df["_analytical_type_norm"] == analytical_type
        if sample_type == TYPE_MATRIX_SPIKE_DUP and not has_qc_type:
            log.warning(
                "matrix_spike_duplicate: cannot route without QC_TYPE — skipped."
            )
            return None

    subset = df[mask].copy()

    if subset.empty:
        if sample_type in [TYPE_MATRIX_SPIKE, TYPE_MATRIX_SPIKE_DUP]:
            log.warning(
                "%s: no records found. Data expected from Datamine in Phase 2.",
                sample_type,
            )
        else:
            log.warning("%s: no records found in this export.", sample_type)
        return None

    return subset


def _validate_columns(df: pd.DataFrame, sample_type: str) -> bool:
    required = REQUIRED_COLUMNS.get(sample_type, [])
    missing = [c for c in required if c not in df.columns]
    entirely_null = [
        c for c in required
        if c in df.columns and df[c].isna().all()
    ]

    if missing:
        log.error(
            "%s: missing required columns %s — skipping.", sample_type, missing
        )
        return False

    if entirely_null:
        log.warning(
            "%s: columns %s are entirely null — detector results may be limited.",
            sample_type, entirely_null,
        )

    return True


def _log_type_summary(df: pd.DataFrame, has_qc_type: bool) -> None:
    if has_qc_type:
        counts = (
            df.groupby(["_analytical_type_norm", "_qc_type_norm"])
            .size()
            .reset_index(name="count")
        )
        log.info("Sample types found (ANALYTICAL_TYPE x QC_TYPE):")
        for _, row in counts.iterrows():
            log.info(
                "  %-15s x %-6s  %d rows",
                row["_analytical_type_norm"],
                row["_qc_type_norm"],
                row["count"],
            )
    else:
        counts = df["_analytical_type_norm"].value_counts()
        log.info("Sample types found (ANALYTICAL_TYPE only):")
        for t, n in counts.items():
            log.info("  %-20s %d rows", t, n)


from data_loader import load_qc_data

df = load_qc_data("ResultSet.csv")
results = classify_and_route(df)

# Access subsets
blank_df     = results["blank"]
standard_df  = results["standard"]
duplicate_df = results["duplicate"]
replicate_df = results["replicate"]
ms_df        = results["matrix_spike"]
msd_df       = results["matrix_spike_duplicate"]

# Print summary
for sample_type, result in results.items():
    rows   = len(result) if result is not None else 0
    status = "READY" if result is not None else "NOT PRESENT"
    print(f"{sample_type:<35} {rows:>8}   {status}")
