"""
Shared QC warning-state vocabulary.

Every QC method reports each analysed item as exactly one of three states.
Each method decides *how* its own analysis outcomes map onto these states
(e.g. LCS maps its DRIFT_STATUS values in
src/detectors/control_detector.py's WARNING_STATE_BY_DRIFT_STATUS); the
report/POC layer only ever knows these three values, so it never needs
per-method branching.

    FAIL    - the result has breached an acceptance limit; action required.
    WARNING - within limits, but the analysis found something that needs review.
    PASS    - no issue found.
"""

FAIL = "FAIL"
WARNING = "WARNING"
PASS = "PASS"

# Worst first -- this order drives sorting everywhere.
STATES = (FAIL, WARNING, PASS)

# Lower rank = worse. Unknown values sort after every real state.
STATE_RANK = {FAIL: 0, WARNING: 1, PASS: 2}
UNKNOWN_RANK = len(STATES)

STATE_LABELS = {FAIL: "Fail", WARNING: "Warning", PASS: "Pass"}


def worst_state(states):
    """Return the worst of an iterable of states (PASS if empty)."""
    ranked = [s for s in states if s in STATE_RANK]
    if not ranked:
        return PASS
    return min(ranked, key=STATE_RANK.__getitem__)
