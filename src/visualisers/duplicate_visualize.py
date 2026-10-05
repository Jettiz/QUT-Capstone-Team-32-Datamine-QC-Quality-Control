"""
#Code cell to run...

from src.visualizers.duplicate_visualizer import generate_duplicate_plots

plots = generate_duplicate_plots(
    result=result,  # your DuplicateDetector output
    output_dir=ROOT / "outputs" / "duplicate_plots",
    top_n=5,
)

from IPython.display import Image, display

for group, path in plots.items():
    print(group)
    display(Image(filename=str(path)))

"""

# =============================================================================
# DATAMINE — DUPLICATE QC VISUALISER
# Purpose: Render current duplicate QC observations against persisted references.
# Inputs: DuplicateAnomalyResult and the historical reference model CSV.
# Outputs: One PNG per selected (JOB_CODE, ANALYTE_CODE) candidate.
# =============================================================================
"""Duplicate QC distribution visualiser.

Rendering consumes completed detector output; it never calculates detection
flags or fits reference models. Unlike Control charts, concentration replaces
 time on the horizontal axis. Negative Q95 segments are omitted, not clipped.

Entry points:
    plot_duplicate_candidate: render one selected job/analyte group.
    generate_duplicate_plots: render ranked groups from DuplicateDetector.
"""
from hashlib import sha256
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ..detectors.base_detector import DuplicateAnomalyResult

_REVIEW_CLASSES = {'HISTORICAL_HIGH', 'LIMIT_EXCEEDED', 'BOTH_EXCEEDED'}
_COLORS = {
    'NORMAL': '#1a8a3d', 'HISTORICAL_HIGH': '#d9a300',
    'LIMIT_EXCEEDED': '#ec835a', 'BOTH_EXCEEDED': '#d03b3b',
    'INVALID_Q95_REFERENCE': '#7a7a7a',
}


def _require(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'{label} is missing columns: {sorted(missing)}')


def _plottable(frame):
    """Copy finite observations with positive concentrations for a log axis."""
    frame = frame.copy()
    for column in ('MEAN_CONC', 'RPD'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    return frame.loc[np.isfinite(frame['MEAN_CONC']) &
                     np.isfinite(frame['RPD']) & (frame['MEAN_CONC'] > 0) &
                     (frame['RPD'] >= 0)]


def plot_duplicate_candidate(
    current: pd.DataFrame, reference: pd.DataFrame,
    output_dir: Union[str, Path], *,
    candidate: Optional[Dict[str, Any]] = None,
    historical: Optional[pd.DataFrame] = None,
) -> Path:
    """Save one job/analyte chart and return its PNG path.

    current is the group's full scored result slice (including normal rows).
    reference is one analyte's persisted grid. historical, if supplied, is an
    already-selected analyte's prepared observations with MEAN_CONC and RPD.
    candidate optionally supplies RANK, ANOMALY_COUNT and ANOMALY_RATE.
    """
    _require(current, ['JOB_CODE', 'ANALYTE_CODE', 'MEAN_CONC', 'RPD',
                       'ANOMALY_CLASS'], 'Current results')
    if current.empty or len(current[['JOB_CODE', 'ANALYTE_CODE']].drop_duplicates()) != 1:
        raise ValueError('Select exactly one nonempty job/analyte group.')
    job, analyte = current.iloc[0][['JOB_CODE', 'ANALYTE_CODE']]
    _require(reference, ['ANALYTE_CODE', 'LOG_MEAN_CONC', 'Q25', 'Q50',
                         'Q75', 'Q95', 'ALLOWABLE_RPD'], 'Reference')
    ref = reference.copy()
    if ref.empty or not ref['ANALYTE_CODE'].eq(analyte).all():
        raise ValueError(f'Reference must contain only analyte {analyte}.')
    curves = ['Q25', 'Q50', 'Q75', 'Q95', 'ALLOWABLE_RPD']
    for column in ['LOG_MEAN_CONC'] + curves:
        ref[column] = pd.to_numeric(ref[column], errors='coerce')
    if not np.isfinite(ref[['LOG_MEAN_CONC'] + curves].to_numpy()).all():
        raise ValueError('Reference grid contains nonfinite values.')
    if len(ref) < 2 or ref['LOG_MEAN_CONC'].duplicated().any():
        raise ValueError('Reference needs at least two unique concentration grid points.')
    ref = ref.sort_values('LOG_MEAN_CONC')
    x = np.power(10.0, ref['LOG_MEAN_CONC'].to_numpy())
    if not (np.isfinite(x) & (x > 0)).all():
        raise ValueError('Reference concentrations must be finite and positive.')
    observations = _plottable(current)
    if observations.empty:
        raise ValueError('Selected group has no plottable observations.')
    hist = pd.DataFrame()
    if historical is not None and not historical.empty:
        _require(historical, ['MEAN_CONC', 'RPD'], 'Historical observations')
        hist = _plottable(historical)

    # Use an Agg canvas directly, avoiding changes to the caller's pyplot backend.
    fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    if not hist.empty:
        ax.scatter(hist['MEAN_CONC'], hist['RPD'], s=20, alpha=.25,
                   color='#4C78A8', label='Historical observations', zorder=1)
    low, high = ref['Q25'].to_numpy(), ref['Q75'].to_numpy()
    ax.fill_between(x, low, high, where=(low >= 0) & (high >= low),
                    color='#dce8f5', alpha=.8, label='Historical Q25–Q75', zorder=0)
    for column, color, style, label in [
        ('Q25', '#8a8a8a', ':', 'Q25'), ('Q50', '#234766', '-', 'Median (Q50)'),
        ('Q75', '#8a8a8a', ':', 'Q75'), ('Q95', '#d03b3b', '-', 'Historical Q95'),
        ('ALLOWABLE_RPD', '#88562d', '--', 'Allowable RPD'),
    ]:
        values = ref[column].to_numpy().copy()
        values[values < 0] = np.nan  # break invalid segments; never invent a limit
        ax.plot(x, values, color=color, linestyle=style, linewidth=1.5, label=label)
    for category, group in observations.groupby('ANOMALY_CLASS', sort=False):
        ax.scatter(group['MEAN_CONC'], group['RPD'], s=75,
                   color=_COLORS.get(category, '#7a7a7a'), edgecolors='black',
                   linewidths=.7, zorder=5,
                   label=f'Current: {str(category).replace("_", " ").title()}')
    title = f'{analyte} duplicate QC — job {job}'
    if candidate:
        title += f'\nRank {candidate.get("RANK", "?")} | '
        title += f'{candidate.get("ANOMALY_COUNT", "?")} flagged'
        rate = candidate.get('ANOMALY_RATE')
        if rate is not None and pd.notna(rate):
            title += f' ({float(rate):.1%})'
    notes = []
    if (ref['Q95'] < 0).any():
        notes.append('Negative Q95 segments omitted; Q95 assessment unavailable there.')
    outside = ~observations['MEAN_CONC'].between(x.min(), x.max())
    if outside.any():
        notes.append('Outside reference range: Q95 unassessed; allowable uses endpoint limits.')
        ax.scatter(observations.loc[outside, 'MEAN_CONC'], observations.loc[outside, 'RPD'],
                   s=150, facecolors='none', edgecolors='#252525', linewidths=1,
                   label='Outside reference range', zorder=6)
    ax.set(title=title, xlabel='Mean concentration (reported units; log scale)',
           ylabel='Relative percent difference (%)')
    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=.15)
    if notes:
        fig.text(.02, .015, '\n'.join(notes), fontsize=8)
    fig.tight_layout(rect=(0, .07 if notes else 0, 1, 1))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    identifiers = '_'.join(re.sub(r'[^A-Za-z0-9_.-]+', '_', str(v))[:70]
                           or 'unknown' for v in (job, analyte))
    digest = sha256(repr((str(job), str(analyte))).encode()).hexdigest()[:10]
    path = output / f'duplicate_{identifiers}_{digest}.png'
    fig.savefig(path, dpi=150)
    fig.clear()
    return path


def generate_duplicate_plots(
    result: DuplicateAnomalyResult,
    reference_path: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = 'outputs/duplicate_plots', *,
    only_flagged: bool = True, top_n: Optional[int] = 5,
    historical_path: Optional[Union[str, Path]] = None,
) -> Dict[Tuple[str, str], Path]:
    """Render ranked candidates; return (JOB_CODE, ANALYTE_CODE) -> PNG.

    Default: first five ranked flagged groups. top_n=None renders all selected
    groups. only_flagged=False includes all scored groups, ranked groups first.
    Reference defaults to result.details['reference_model_path'].
    Optional history CSV must have ANALYTE_CODE, MEAN_CONC, RPD and
    PRECISION_STATUS; only positive RPD and Pass/IgnoreFailure are displayed.
    No historical raw population is required to render reference curves.
    """
    if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 0):
        raise ValueError('top_n must be a nonnegative integer or None.')
    rows = result.details.get('results', [])
    if not rows or top_n == 0:
        return {}
    scored = pd.DataFrame(rows)
    _require(scored, ['JOB_CODE', 'ANALYTE_CODE', 'ANOMALY_CLASS'], 'Detector output')
    candidates = result.details.get('ranked_candidates', [])
    groups = {(row['JOB_CODE'], row['ANALYTE_CODE']): row
              for row in sorted(candidates, key=lambda row: row['RANK'])}
    if only_flagged:
        flagged_keys = set(scored.loc[scored['ANOMALY_CLASS'].isin(_REVIEW_CLASSES),
                                     ['JOB_CODE', 'ANALYTE_CODE']].itertuples(index=False, name=None))
        groups = {key: value for key, value in groups.items() if key in flagged_keys}
    else:
        for key in scored[['JOB_CODE', 'ANALYTE_CODE']].itertuples(index=False, name=None):
            groups.setdefault(key, {})
    selected = list(groups.items())[:top_n] if top_n is not None else list(groups.items())
    if not selected:
        return {}
    source = reference_path if reference_path is not None else result.details.get('reference_model_path')
    if not source:
        raise ValueError('Supply reference_path or detector reference_model_path.')
    reference = pd.read_csv(source, dtype={'ANALYTE_CODE': str})
    history = None
    if historical_path is not None:
        history = pd.read_csv(historical_path, dtype={'ANALYTE_CODE': str})
        _require(history, ['ANALYTE_CODE', 'MEAN_CONC', 'RPD', 'PRECISION_STATUS'], 'History')
        history = _plottable(history)
        history = history.loc[(history['RPD'] > 0) &
                              history['PRECISION_STATUS'].isin(['Pass', 'IgnoreFailure'])]
    plots = {}
    for (job, analyte), candidate in selected:
        current = scored.loc[scored['JOB_CODE'].eq(job) & scored['ANALYTE_CODE'].eq(analyte)]
        ref = reference.loc[reference['ANALYTE_CODE'].eq(analyte)]
        hist = history.loc[history['ANALYTE_CODE'].eq(analyte)] if history is not None else None
        plots[(job, analyte)] = plot_duplicate_candidate(
            current, ref, output_dir, candidate=candidate, historical=hist)
    return plots
