
"""
#From caller run the following...

from src.detectors.duplicate_detector import DuplicateDetector

detector = DuplicateDetector()
result = detector.detect(df, job_code=None)

ranked_candidates = pd.DataFrame(
    result.details["ranked_candidates"]
)

"""
# =============================================================================
# Duplicate QC Detector — Current Batch vs Historical Reference
# =============================================================================
#
# Author: Robert Clark
# Project: QUT Capstone — Team 32 / Datamine ICP-MS QC
#
# Purpose:
# Assess current-batch Duplicate QC observations against a previously
# modelled and persisted historical reference dataset.
#
# Reference model:
# data/reference/DUP_historical_reference_model.csv
#
# Processing:
# 1. Select Duplicate QC observations and prepare paired-result features.
# 2. Calculate mean concentration and Relative Percent Difference (RPD).
# 3. Load the persisted analyte-specific historical reference curves.
# 4. Interpolate reference thresholds using log mean concentration.
# 5. Identify exceedances of historical Q95 and allowable RPD.
# 6. Rank Job x Analyte candidates for chemist review.
# 7. Return a DuplicateAnomalyResult for reporting and visualisation.
#
# Notes:
# - Historical model fitting is performed separately.
# - The persisted reference model is read-only during detection.
# - Negative Q95 values are reported and excluded from historical scoring.
# - Updated 2026-10-05: defensive Q95 classification and review filtering.
# - Chart generation is handled by the Duplicate visualiser.
# - Flagged observations are review candidates requiring interpretation.
#
# =============================================================================


"""
Requires the historical reference model to be pre-built and ready in 'data/reference' folder.

Example (run from repository root):
    detector = DuplicateDetector()
    result = detector.detect(batch_df, job_code=None)

Optional YAML keys: reference_model_path, top_n. Relative model paths are
resolved against repository_root (default: inferred from module location).
Confidence is 0.0 because this method does not estimate probabilistic
confidence; details explicitly records that limitation. Severity is a review
priority mapping, not a calibrated scientific severity estimate.

"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from .base_detector import DuplicateAnomalyResult

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PAIR_COLUMNS = [
    'JOB_CODE', 'ANALYSED_DATE', 'ANALYTE_CODE', 'SCHEME_CODE',
    'PARENT_NUMERIC_FINAL_VALUE', 'NUMERIC_FINAL_VALUE',
    'STAT_DL_DUP_VALUE', 'LIM_REP_DUP_VALUE', 'PRECISION_STATUS', 'UNIT_CODE',
]
FILTER_COLUMNS = ['ANALYTICAL_TYPE', 'STD_LOT_CODE', 'STD_CODE']
CURVE_COLUMNS = ['Q25', 'Q50', 'Q75', 'Q95', 'ALLOWABLE_RPD']
REVIEW_CLASSES = ['HISTORICAL_HIGH', 'LIMIT_EXCEEDED', 'BOTH_EXCEEDED']
PRIORITY = {'INVALID_Q95_REFERENCE': 0, 'NORMAL': 0, 'HISTORICAL_HIGH': 1,
            'LIMIT_EXCEEDED': 2, 'BOTH_EXCEEDED': 3}
RANK_COLUMNS = ['MAX_CLASS_PRIORITY', 'BOTH_EXCEED_COUNT', 'ANOMALY_RATE',
                'MAX_ALLOWABLE_RATIO', 'MAX_Q95_RATIO', 'ANOMALY_COUNT']


def _require_columns(df, columns, label):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f'{label}: missing required columns {missing}')


def _records(df):
    """Convert pandas scalars/nulls/dates to JSON-compatible records."""
    import json
    return json.loads(df.to_json(orient='records', date_format='iso'))


def _load_reference(path):
    reference = pd.read_csv(path)
    _require_columns(reference, ['ANALYTE_CODE', 'LOG_MEAN_CONC'] + CURVE_COLUMNS,
                     'Duplicate reference')
    if reference.empty or reference['ANALYTE_CODE'].isna().any():
        raise ValueError('Duplicate reference is empty or has missing analyte keys')
    for col in ['LOG_MEAN_CONC'] + CURVE_COLUMNS:
        reference[col] = pd.to_numeric(reference[col], errors='coerce')
        if not np.isfinite(reference[col].to_numpy(dtype=float)).all():
            raise ValueError(f'Duplicate reference contains invalid {col} values')
    # This notebook uses analyte-only curves. Do not silently combine schemes.
    if 'SCHEME_CODE' in reference and (
        reference.groupby('ANALYTE_CODE')['SCHEME_CODE'].nunique(dropna=False) > 1
    ).any():
        raise ValueError('Multiple scheme curves per analyte require a scheme-aware detector')
    for analyte, group in reference.groupby('ANALYTE_CODE'):
        if len(group) < 2 or group['LOG_MEAN_CONC'].duplicated().any():
            raise ValueError(f'{analyte}: reference grid must have at least two unique points')
    return reference


def _prepare(df, job_code):
    _require_columns(df, FILTER_COLUMNS + PAIR_COLUMNS, 'Current batch')
    working = df.copy()
    if job_code is not None:
        working = working.loc[working['JOB_CODE'] == job_code]
    selected = working.loc[
        (working['ANALYTICAL_TYPE'] == 'Duplicate')
        & (working['STD_LOT_CODE'] == 'Sample')
        & (working['STD_CODE'] == 'Sample')
    ].copy()
    report = {'n_input': len(working), 'n_duplicate': len(selected),
              'n_missing_analyte': int(selected['ANALYTE_CODE'].isna().sum())}
    prepared = selected.loc[selected['ANALYTE_CODE'].notna(), PAIR_COLUMNS].copy()
    for col in ['NUMERIC_FINAL_VALUE', 'PARENT_NUMERIC_FINAL_VALUE']:
        prepared[col] = pd.to_numeric(prepared[col], errors='coerce')
    prepared['ABS_DIFF'] = abs(prepared['NUMERIC_FINAL_VALUE']
                               - prepared['PARENT_NUMERIC_FINAL_VALUE'])
    prepared['MEAN_CONC'] = (prepared['NUMERIC_FINAL_VALUE']
                             + prepared['PARENT_NUMERIC_FINAL_VALUE']) / 2
    prepared['RPD'] = (100 * prepared['ABS_DIFF']
                       / prepared['MEAN_CONC'].replace(0, np.nan)).round(2)
    eligible = (np.isfinite(prepared['MEAN_CONC'])
                & np.isfinite(prepared['RPD']) & (prepared['MEAN_CONC'] > 0))
    report.update(n_prepared=len(prepared), n_eligible=int(eligible.sum()),
                  n_ineligible=int((~eligible).sum()))
    # Retain current failures and zero RPDs; historical training filters do not apply.
    return prepared.loc[eligible].copy(), report


def _score(current, reference):
    tables = []
    for analyte, group in current.groupby('ANALYTE_CODE', sort=True):
        ref = reference.loc[reference['ANALYTE_CODE'] == analyte].sort_values('LOG_MEAN_CONC')
        scored = group.copy()
        scored['LOG_MEAN_CONC'] = np.log10(scored['MEAN_CONC'])
        x = ref['LOG_MEAN_CONC'].to_numpy()
        scored['WITHIN_REFERENCE_RANGE'] = scored['LOG_MEAN_CONC'].between(x.min(), x.max())
        for col in CURVE_COLUMNS:
            # Preserve notebook endpoint clamping outside the historical range.
            scored[f'REF_{col}'] = np.interp(scored['LOG_MEAN_CONC'], x, ref[col])
        tables.append(scored)
    if not tables:
        return pd.DataFrame()
    scored = pd.concat(tables, ignore_index=True)
    # Keep invalid model predictions visible; do not clip them to zero.
    scored['VALID_Q95_REFERENCE'] = (
        np.isfinite(scored['REF_Q95']) & (scored['REF_Q95'] >= 0)
    )
    scored['Q95_ASSESSABLE'] = (
        scored['WITHIN_REFERENCE_RANGE'] & scored['VALID_Q95_REFERENCE']
    )
    scored['EXCEEDS_Q95'] = (
        scored['Q95_ASSESSABLE'] & (scored['RPD'] > scored['REF_Q95'])
    )
    scored['EXCEEDS_ALLOWABLE_RPD'] = scored['RPD'] > scored['REF_ALLOWABLE_RPD']
    scored['Q95_EXCESS'] = (scored['RPD'] - scored['REF_Q95']).clip(
        lower=0
    ).where(scored['Q95_ASSESSABLE'])
    scored['ALLOWABLE_EXCESS'] = (scored['RPD'] - scored['REF_ALLOWABLE_RPD']).clip(lower=0)
    for name, threshold in [('Q95_RATIO', 'REF_Q95'), ('ALLOWABLE_RATIO', 'REF_ALLOWABLE_RPD')]:
        denominator = scored[threshold].where(scored[threshold] > 0)
        scored[name] = scored['RPD'] / denominator
    scored.loc[~scored['Q95_ASSESSABLE'], 'Q95_RATIO'] = np.nan
    scored['ANOMALY_CLASS'] = np.select(
        [scored['EXCEEDS_Q95'] & scored['EXCEEDS_ALLOWABLE_RPD'],
         scored['EXCEEDS_ALLOWABLE_RPD'], scored['EXCEEDS_Q95']],
        ['BOTH_EXCEEDED', 'LIMIT_EXCEEDED', 'HISTORICAL_HIGH'],
        default=np.where(scored['VALID_Q95_REFERENCE'],
                         'NORMAL', 'INVALID_Q95_REFERENCE'))
    scored['CLASS_PRIORITY'] = scored['ANOMALY_CLASS'].map(PRIORITY)
    return scored


def _rank(scored):
    if scored.empty:
        return pd.DataFrame()
    keys = ['JOB_CODE', 'ANALYTE_CODE']
    flagged = scored.loc[scored['ANOMALY_CLASS'].isin(REVIEW_CLASSES)]
    if flagged.empty:
        return pd.DataFrame()
    totals = scored.groupby(keys, dropna=False).size().rename('TOTAL_OBSERVATION_COUNT')
    summary = flagged.groupby(keys, dropna=False).agg(
        ANOMALY_COUNT=('ANOMALY_CLASS', 'size'),
        MAX_CLASS_PRIORITY=('CLASS_PRIORITY', 'max'), MAX_RPD=('RPD', 'max'),
        MAX_Q95_RATIO=('Q95_RATIO', 'max'), MAX_ALLOWABLE_RATIO=('ALLOWABLE_RATIO', 'max'),
        Q95_EXCEED_COUNT=('EXCEEDS_Q95', 'sum'),
        ALLOWABLE_EXCEED_COUNT=('EXCEEDS_ALLOWABLE_RPD', 'sum'),
        BOTH_EXCEED_COUNT=('ANOMALY_CLASS', lambda x: int((x == 'BOTH_EXCEEDED').sum())),
    ).join(totals).reset_index()
    summary['ANOMALY_RATE'] = summary['ANOMALY_COUNT'] / summary['TOTAL_OBSERVATION_COUNT']
    summary['MAX_ANOMALY_CLASS'] = summary['MAX_CLASS_PRIORITY'].map({v: k for k, v in PRIORITY.items()})
    summary = summary.sort_values(RANK_COLUMNS, ascending=False, kind='stable').reset_index(drop=True)
    summary['RANK'] = np.arange(1, len(summary) + 1)
    return summary


class DuplicateDetector:
    """Read-only reference assessment; outputs review candidates, not diagnoses."""

    def __init__(self, config_path: Optional[str] = None, debug: bool = False,
                 *, repository_root=None, reference_model_path=None, top_n=5):
        self.config = {}
        if config_path is not None:
            with open(config_path, encoding='utf-8') as handle:
                self.config = yaml.safe_load(handle) or {}
            if not isinstance(self.config, dict):
                raise ValueError('Detector configuration must be a YAML mapping')
        root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[2]
        model_path = Path(reference_model_path or self.config.get(
            'reference_model_path', 'data/reference/DUP_historical_reference_model.csv'))
        self.reference_model_path = model_path if model_path.is_absolute() else root / model_path
        self.top_n = self.config.get('top_n', top_n)
        if isinstance(self.top_n, bool) or not isinstance(self.top_n, int) or self.top_n <= 0:
            raise ValueError('top_n must be a positive integer')
        self.debug = debug

    def detect(self, df: pd.DataFrame, job_code: Optional[str] = None) -> DuplicateAnomalyResult:
        current, report = _prepare(df, job_code)
        # Missing/malformed reference is a configuration error, not a normal result.
        reference = _load_reference(self.reference_model_path)
        matched = current['ANALYTE_CODE'].isin(reference['ANALYTE_CODE'])
        missing = sorted(current.loc[~matched, 'ANALYTE_CODE'].unique().tolist(), key=str)
        report['n_missing_reference'] = int((~matched).sum())
        scored = _score(current.loc[matched], reference)
        ranked = _rank(scored)
        n_flagged = int((scored['ANOMALY_CLASS'].isin(REVIEW_CLASSES)).sum()) if not scored.empty else 0
        report['n_scored'] = len(scored)
        report['n_invalid_q95_reference'] = int((~scored['VALID_Q95_REFERENCE']).sum()) if not scored.empty else 0
        report['n_q95_assessable'] = int(scored['Q95_ASSESSABLE'].sum()) if not scored.empty else 0
        report['n_reference_negative_q95_points'] = int((reference['Q95'] < 0).sum())
        invalid_q95 = scored.loc[~scored['VALID_Q95_REFERENCE']] if not scored.empty else scored
        report['n_outside_reference_range'] = int((~scored['WITHIN_REFERENCE_RANGE']).sum()) if not scored.empty else 0
        max_priority = int(ranked['MAX_CLASS_PRIORITY'].max()) if not ranked.empty else 0
        severity = {0: 'low', 1: 'medium', 2: 'high', 3: 'high'}[max_priority]
        reason = (f'{n_flagged} of {len(scored)} assessed Duplicate observations flagged for chemist review'
                  if len(scored) else 'No assessable Duplicate observations for the given input and reference')
        if self.debug:
            logger.warning('Duplicate assessment: %s; validation=%s', reason, report)
        return DuplicateAnomalyResult(
            detected=n_flagged > 0, confidence=0.0, severity=severity,
            visualizable=not ranked.empty,
            details={
                'reason': reason, 'results': _records(scored),
                'ranked_candidates': _records(ranked),
                'top_candidates': _records(ranked.head(self.top_n)),
                'validation_report': report, 'missing_reference_analytes': missing,
                'reference_model_path': str(self.reference_model_path),
                'invalid_q95_observations': _records(invalid_q95),
                'invalid_q95_policy': 'Negative Q95 is unassessable; allowable RPD is evaluated independently. Zero Q95 can be assessed, but its ratio is undefined.',
                'confidence_assessed': False,
                'confidence_reason': 'Breach rules do not estimate probabilistic confidence.',
                'severity_basis': 'Provisional review priority: historical high=medium; limit breach=high.',
                'outside_range_policy': 'Suppress Q95 flags; clamp allowable RPD to nearest grid endpoint.',
            })

