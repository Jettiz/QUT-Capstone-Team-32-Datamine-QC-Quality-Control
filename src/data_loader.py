from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH   = Path("QC_Sample_Data.csv")
DEFAULT_CONFIG_PATH = Path("column_config.csv")

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
) -> pd.DataFrame:
    """
    Load, validate, map, and transform a QC sample data file.

    """
    data_path   = Path(data_path)
    config_path = Path(config_path)

    # Load column configuration
    config = _load_config(config_path)

    # Load raw data file
    raw = _load_file(data_path)

    # Normalise column names
    raw.columns    = raw.columns.str.strip().str.upper()
    config["source_column"] = config["source_column"].str.strip().str.upper()

    # 4. Validate required columns
    _validate_required_columns(raw, config)

    # 5. Select and rename to internal column names
    available = config[config["source_column"].isin(raw.columns)].copy()

    missing_optional = config[
        (~config["source_column"].isin(raw.columns)) &
        (config["required"].astype(str).str.lower().isin(["false", "0", "no"]))
    ]
    if not missing_optional.empty:
        log.warning(
            "Optional columns not found, will be absent from output: %s",
            missing_optional["source_column"].tolist(),
        )

    df = raw[available["source_column"]].copy()
    rename_map = dict(zip(available["source_column"], available["internal_column"]))
    df = df.rename(columns=rename_map)

    # 6. Apply data types
    df = _apply_dtypes(df, available)

    log.info("Loaded %d rows and %d columns from '%s'.", len(df), len(df.columns), data_path.name)
    log.info("Internal columns: %s", df.columns.tolist())

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
        raise ValueError(f"column_config.csv is missing required config columns: {missing}")

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
    log.info("Read '%s' — %d rows, %d columns.", data_path.name, len(raw), len(raw.columns))
    return raw


def _validate_required_columns(raw: pd.DataFrame, config: pd.DataFrame) -> None:
    required = config[
        config["required"].astype(str).str.lower().isin(["true", "1", "yes"])
    ]["source_column"]

    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(
            f"Required columns missing from source file: {missing}. "
            "Check the export is a valid CCLAS 6 QC file, "
            "or update column_config.csv to match the source column names."
        )

    log.info("All required columns present.")


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
            df[col] = pd.to_datetime(df[col], errors="coerce")
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
qc_data = load_qc_data(
    data_path="../data/raw/ResultSet.csv",
    config_path="src/config/column_config.csv"
)


# Display the first five rows
print(qc_data.head())

# Show information about the DataFrame
print(qc_data.info())

# Display the DataFrame dimensions (rows, columns)
print(qc_data.shape)

df = pd.read_csv("data/QC_Sample_Data.csv", nrows=5)

print(df.columns.tolist())