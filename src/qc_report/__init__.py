"""
Report layer between the QC detectors (src/detectors) and the proof of
concept (poc/). Maps each QC method's detector output onto one shared result
contract (contract.py) so the browser never needs per-method code.

- contract.py  result objects + display helpers ("Job unknown", ...)
- registry.py  the QC methods the POC runs (currently LCS / Control)
- lcs_method.py  the LCS adapter, the reference implementation

See docs/QC_INTEGRATION_GUIDE.md for how to add another QC method.
"""

from .contract import (
    INSTRUMENT_UNKNOWN, JOB_UNKNOWN, QCItem, QCJob, QCMethodResult,
    display_instrument, display_instruments, display_job, is_missing,
)
from .registry import QC_METHODS, QCMethod, get_method, register

__all__ = [
    "INSTRUMENT_UNKNOWN", "JOB_UNKNOWN", "QCItem", "QCJob", "QCMethodResult",
    "display_instrument", "display_instruments", "display_job", "is_missing",
    "QC_METHODS", "QCMethod", "get_method", "register",
]
