"""
Data Loader Module

Loads a raw QC export (CCLAS ResultSet.csv, QC_Sample_Data.csv, ...) and maps
it onto the product's INTERNAL column names, as defined by
config/column_config.csv (source_column -> internal_column, dtype, required).
Everything downstream (data_validator.py, the detectors that have been
migrated, the POC server) works on these internal names only.

Several source columns may map to the same internal column when different
exports name the same field differently, e.g.:
    JOB_NAME_ANON, JOB_CODE           -> job_code
    INSTRUMENT_CODE, INSTRUMENT_ID    -> instrument_id
The ROW ORDER of column_config.csv is the priority: per row, the first listed
source column that has a non-blank value wins, and the next one is only used
as a fallback where the preferred one is missing/blank.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH          = Path("QC_Sample_Data.csv")
DEFAULT_CONFIG_PATH        = Path("column_config.csv")
DEFAULT_SUPPLEMENTARY_PATH = Path("QC_Anomaly_Training_Data_v2.xlsx")

MS_SHEET  = "SPK(MS) Assessment"
UNNAMED_COLUMN_PATTERNS = {"", ".1", ".2", "UNNAMED"}

SUPPORTED_FORMATS = {
    ".csv":  "_read_csv",
    ".tsv":  "_read_tsv",
    ".xlsx": "_read_excel",
    ".xls":  "_read_excel",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | data_loader | %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_qc_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    supplementary_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load, validate, map, and transform a QC sample data file.

    Parameters
    ----------
    data_path:
        Path to the primary CCLAS export file (ResultSet.csv or equivalent).
        Contains Blank, Standard, Replicate, and Duplicate records.
    config_path:
        Path to column_config.csv defining source-to-internal column mappings,
        data types, and required/optional flags.
    supplementary_path:
        Optional path to QC_Anomaly_Training_Data_v2.xlsx.
        When provided, Matrix Spike records are loaded from the
        'SPK(MS) Assessment' sheet, then appended to the primary dataset.
        If the file does not exist, a warning is logged and loading continues
        without the supplementary data.
    """
    data_path   = Path(data_path)
    config_path = Path(config_path)

    # Load column configuration
    config = _load_config(config_path)

    # Load raw data file
    raw = _load_file(data_path)

    # Normalise source/config names. CSV and TSV exports can use different
    # spelling/separators for the same logical field, so downstream code must
    # see one stable set of internal names.
    raw.columns = [str(c).strip().upper() for c in raw.columns]
    config["source_column"] = config["source_column"].astype(str).str.strip().str.upper()
    config["internal_column"] = config["internal_column"].astype(str).str.strip()

    # Append supplementary Matrix Spike data before resolving mappings so the
    # same mapping rules are applied to both primary and supplementary records.
    if supplementary_path is not None:
        raw = _append_supplementary(raw, Path(supplementary_path))

    # Resolve CSV/TSV source names to one internal schema.  Multiple config rows
    # may point to the same internal_column (aliases); the first matching alias
    # is used.  Separator/case-only differences are also matched automatically.
    resolved = _resolve_column_mappings(raw, config)
    _validate_required_columns(config, resolved)

    # Only report an optional internal column as absent when NONE of its
    # source aliases is present (e.g. a missing JOB_NAME_ANON is fine when
    # JOB_CODE provides job_code).
    missing_optional = config[
        (~config["internal_column"].isin(available["internal_column"])) &
        (config["required"].astype(str).str.lower().isin(["false", "0", "no"]))
    ]
    if not missing_optional.empty:
        log.warning(
            "Optional columns not found, will be absent from output: %s",
            missing_optional["source_column"].tolist(),
        )

    df = _map_to_internal_columns(raw, available)

    missing_optional = _missing_optional_internal_columns(config, resolved)
    if missing_optional:
        log.warning(
            "Optional internal columns not found in this export: %s",
            missing_optional,
        )

    # Apply data types using the internal schema (not the source-file naming).
    df = _apply_dtypes(df, config)

    # Derive rpd/mean_conc (Replicate/Duplicate precision metrics) -- see
    # _derive_precision_metrics() docstring for why these can't just be a
    # column_config.csv mapping.
    df = _derive_precision_metrics(df)

    log.info(
        "Final dataset: %d rows, %d columns from '%s'%s.",
        len(df),
        len(df.columns),
        data_path.name,
        f" + supplementary '{Path(supplementary_path).name}'"
        if supplementary_path else "",
    )
    log.info("Internal columns: %s", df.columns.tolist())

    if "analytical_type" in df.columns:
        log.info("Sample type counts:")
        for t, n in df["analytical_type"].value_counts().items():
            log.info("  %-30s %d rows", t, n)

    return df


# ---------------------------------------------------------------------------
# Supplementary loader — MS and MSD from Excel
# ---------------------------------------------------------------------------

def _append_supplementary(
    raw: pd.DataFrame,
    supplementary_path: Path,
) -> pd.DataFrame:
    """
    Load Matrix Spike records from the supplementary Excel file
    and append them to the primary DataFrame.
    """
    if not supplementary_path.is_file():
        log.warning(
            "Supplementary file not found: '%s'. "
            "Matrix Spike and MSD records will not be included.",
            supplementary_path,
        )
        return raw

    frames = []

    # Matrix Spike
    try:
        ms = pd.read_excel(supplementary_path, sheet_name=MS_SHEET)
        ms.columns = [str(c).strip().upper() for c in ms.columns]
        ms = _drop_unnamed_columns(ms)
        log.info(
            "Loaded %d Matrix Spike rows from sheet '%s'.", len(ms), MS_SHEET
        )
        frames.append(ms)
    except Exception as exc:
        log.error(
            "Could not load Matrix Spike sheet '%s': %s — skipping.",
            MS_SHEET, exc,
        )

    if not frames:
        log.warning(
            "No supplementary data could be loaded — "
            "returning primary dataset only."
        )
        return raw

    combined = pd.concat([raw] + frames, ignore_index=True)

    log.info(
        "Appended supplementary: %d primary + %d supplementary = %d total rows.",
        len(raw),
        sum(len(f) for f in frames),
        len(combined),
    )

    return combined


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_to_internal_columns(raw: pd.DataFrame, available: pd.DataFrame) -> pd.DataFrame:
    """
    Build the internal-schema frame from the raw columns listed in
    `available` (the column_config.csv rows whose source column is present).

    A single source column is copied as-is. When several present source
    columns share one internal column, they are coalesced row by row in
    column_config.csv order: blank/whitespace-only values count as missing,
    and the first non-missing value wins (see the module docstring).
    """
    columns = {}
    for internal, sources in available.groupby("internal_column", sort=False)["source_column"]:
        sources = list(sources)
        if len(sources) == 1:
            columns[internal] = raw[sources[0]]
            continue

        merged = None
        for source in sources:
            values = raw[source]
            blank = values.isna() | values.astype(str).str.strip().eq("")
            values = values.mask(blank)
            merged = values if merged is None else merged.fillna(values)
        columns[internal] = merged

    return pd.DataFrame(columns, index=raw.index)


def _drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty or unnamed columns produced by Excel exports."""
    cols_to_drop = [
        c for c in df.columns
        if c.strip().upper() in UNNAMED_COLUMN_PATTERNS
        or c.startswith("UNNAMED")
    ]
    if cols_to_drop:
        log.info("Dropping unnamed/empty columns: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)
    return df


def _load_config(config_path: Path) -> pd.DataFrame:
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Column configuration file not found: '{config_path}'. "
            "Provide a valid path to column_config.csv."
        )

    config = pd.read_csv(config_path)
    config.columns = config.columns.str.strip().str.lower()

    required_config_cols = {"source_column", "internal_column", "dtype", "required"}
    missing = required_config_cols - set(config.columns)
    if missing:
        raise ValueError(
            f"column_config.csv is missing required config columns: {missing}"
        )

    log.info(
        "Loaded column configuration from '%s' — %d mappings defined.",
        config_path.name, len(config),
    )
    return config


def _load_file(data_path: Path) -> pd.DataFrame:
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Data file not found: '{data_path}'. "
            "Place the CCLAS export in the expected location."
        )

    suffix = data_path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported formats: {list(SUPPORTED_FORMATS.keys())}"
        )

    readers = {
        ".csv":  lambda p: pd.read_csv(p, low_memory=False),
        ".tsv":  lambda p: pd.read_csv(p, sep="\t", low_memory=False),
        ".xlsx": lambda p: pd.read_excel(p),
        ".xls":  lambda p: pd.read_excel(p),
    }

    raw = readers[suffix](data_path)
    log.info(
        "Read '%s' — %d rows, %d columns.", data_path.name, len(raw), len(raw.columns)
    )
    return raw


def _column_key(name: object) -> str:
    """
    Return a comparison key for source-column matching.

    This intentionally ignores case, spaces, underscores, hyphens and other
    punctuation. For example, ANALYTE_CODE, "Analyte Code" and "analyte-code"
    all resolve to the same key. This is useful because DataMine CSV and TSV
    exports can contain equivalent fields under different naming conventions.
    """
    return re.sub(r"[^A-Z0-9]+", "", str(name).strip().upper())


def _resolve_column_mappings(
    raw: pd.DataFrame,
    config: pd.DataFrame,
) -> dict[str, str]:
    """
    Resolve raw CSV/TSV columns to stable internal column names.

    `column_config.csv` may contain more than one source_column for the same
    internal_column. This allows explicit CSV/TSV aliases without changing any
    detector/visualiser code. Exact source-name matches take priority; if no
    exact match exists, a normalised-name match is attempted.

    Returns
    -------
    dict
        {internal_column: actual_source_column}
    """
    raw_columns = list(raw.columns)
    raw_set = set(raw_columns)

    # A normalised key can theoretically map to more than one raw column.
    # Keep all candidates so ambiguous mappings can be rejected safely.
    keyed_raw: dict[str, list[str]] = {}
    for col in raw_columns:
        keyed_raw.setdefault(_column_key(col), []).append(col)

    resolved: dict[str, str] = {}

    # Preserve config order: it defines alias priority.
    for internal, group in config.groupby("internal_column", sort=False):
        aliases = [str(v).strip().upper() for v in group["source_column"]]

        chosen: str | None = None

        # Prefer an exact configured source name.
        for alias in aliases:
            if alias in raw_set:
                chosen = alias
                break

        # Fall back to naming-convention normalisation.
        if chosen is None:
            for alias in aliases:
                candidates = keyed_raw.get(_column_key(alias), [])
                if len(candidates) == 1:
                    chosen = candidates[0]
                    break
                if len(candidates) > 1:
                    raise ValueError(
                        f"Ambiguous source-column mapping for internal column "
                        f"'{internal}': alias '{alias}' matches {candidates}."
                    )

        if chosen is not None:
            resolved[str(internal)] = chosen
            log.debug("Mapped source '%s' -> internal '%s'.", chosen, internal)

    return resolved


def _required_internal_columns(config: pd.DataFrame) -> list[str]:
    """Return required fields at the internal-schema level."""
    required_mask = config["required"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    return (
        config.loc[required_mask, "internal_column"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def _validate_required_columns(
    config: pd.DataFrame,
    resolved: dict[str, str],
) -> None:
    """
    Validate required logical fields after CSV/TSV alias resolution.

    A required internal field is valid when at least one configured source alias
    resolves to it. This avoids incorrectly requiring both a CSV name and a TSV
    name when they represent the same field.
    """
    required = _required_internal_columns(config)
    missing = [internal for internal in required if internal not in resolved]

    if missing:
        alias_details = {}
        for internal in missing:
            aliases = (
                config.loc[
                    config["internal_column"].astype(str) == internal,
                    "source_column",
                ]
                .astype(str)
                .tolist()
            )
            alias_details[internal] = aliases

        raise ValueError(
            "Required internal columns could not be mapped from the source "
            f"file: {missing}. Configured aliases: {alias_details}. "
            "Add the CSV/TSV source names to column_config.csv using the same "
            "internal_column for equivalent fields."
        )

    log.info("All required internal columns mapped successfully.")


def _missing_optional_internal_columns(
    config: pd.DataFrame,
    resolved: dict[str, str],
) -> list[str]:
    """Return optional logical fields for which no CSV/TSV alias was found."""
    required = set(_required_internal_columns(config))
    all_internal = config["internal_column"].astype(str).drop_duplicates().tolist()
    return [
        internal
        for internal in all_internal
        if internal not in required and internal not in resolved
    ]


def _derive_precision_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive `rpd` (relative percent difference) and `mean_conc` (mean
    concentration) from `measured_value`/`parent_value`, wherever both are
    present. Neither column has (or should have) a column_config.csv row --
    they are not present under any name in either raw source file
    (ResultSet.csv / QC_Sample_Data.csv); they are always computed.

    Formula ported verbatim from notebooks/REP__distribution_analysis.ipynb
    and notebooks/DUP__distribution_analysis.ipynb's build_replicate_model()
    / build_duplicate_model() (identical in both, feeding
    notebooks/REP__historical_reference_model.ipynb /
    DUP__historical_reference_model.ipynb downstream):
        MEAN_CONC = (NUMERIC_FINAL_VALUE + PARENT_NUMERIC_FINAL_VALUE) / 2
        RPD = |NUMERIC_FINAL_VALUE - PARENT_NUMERIC_FINAL_VALUE| / MEAN_CONC * 100
    RPD rounded to 2 d.p.; both NaN wherever `mean_conc == 0` (guards the
    division -- a genuinely zero-sum pair has no meaningful RPD) or either
    input is missing.

    `parent_value` is only ever populated for Replicate/Duplicate rows, so
    this is safe to call unconditionally on the full loaded dataset --
    Blank/Standard/Spike rows simply get NaN for both new columns.
    """
    if "measured_value" not in df.columns or "parent_value" not in df.columns:
        return df

    mean_conc = (df["measured_value"] + df["parent_value"]) / 2
    abs_diff = (df["measured_value"] - df["parent_value"]).abs()

    df["mean_conc"] = mean_conc
    df["rpd"] = np.where(mean_conc != 0, (abs_diff / mean_conc * 100).round(2), np.nan)

    return df


def _apply_dtypes(df: pd.DataFrame, config: pd.DataFrame) -> pd.DataFrame:
    dtype_map = dict(zip(config["internal_column"], config["dtype"]))

    for col, dtype in dtype_map.items():
        if col not in df.columns:
            continue

        dtype = str(dtype).strip().lower()

        if dtype == "str":
            df[col] = df[col].astype(str).where(df[col].notna(), other=pd.NA)

        elif dtype == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")

        elif dtype == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        elif dtype == "datetime":
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
            nat_count = df[col].isna().sum()
            if nat_count > 0:
                log.warning(
                    "Column '%s': %d values could not be parsed as datetime and were set to NaT.",
                    col, nat_count,
                )

        else:
            log.warning(
                "Column '%s': unknown dtype '%s' in config, left unchanged.", col, dtype,
            )

    return df


# ---------------------------------------------------------------------------
# Usage (only when run directly -- importing this module must not load data)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    qc_data = load_qc_data(
        data_path="data/raw/ResultSet.csv",
        config_path="./config/column_config.csv",
        supplementary_path="data/raw/QC_Anomaly_Training_Data_v2.xlsx"
    )

    # Display the first five rows
    print(qc_data.head())

    # Show information about the DataFrame
    print(qc_data.info())

    # Display the DataFrame dimensions (rows, columns)
    print(qc_data.shape)

    # See Matrix Spike rows
    print(qc_data[qc_data["analytical_type"] == "Spike"].head())

    # See all sample type counts
    print(qc_data["analytical_type"].value_counts())

    # See the last 5 rows (where appended data sits)
    print(qc_data.tail())
