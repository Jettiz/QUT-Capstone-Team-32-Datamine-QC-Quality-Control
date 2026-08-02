"""
Base Detector Module

Defines the abstract AnomalyResult class that all detectors return.
This ensures consistent structure across all anomaly types for easier
reporting, visualization, and analysis.

Classes:
- AnomalyResult: Base class for all anomaly detection results
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List


@dataclass
class AnomalyResult:
    """
    Standard result object for any anomaly detection.

    Attributes:
        anomaly_type: str - Type of QC sample ("blank", "lcs", "duplicate", "srm", "ms", "msd")
        detected: bool - Whether anomaly was detected
        confidence: float - Confidence level (0-100%)
        severity: str - "low", "medium", "high", "critical"
        details: Dict - Detailed information (reason, affected_samples, metrics, etc.)
        visualizable: bool - Can this result be visualized?
        timestamp: datetime - When detection was performed

    Methods:
        to_dict(): Convert to dictionary for reporting
        is_actionable(): Determine if anomaly requires immediate action
    """

    anomaly_type: str
    detected: bool
    confidence: float
    severity: str = "low"
    details: Dict[str, Any] = field(default_factory=dict)
    visualizable: bool = True
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.
        Useful for JSON serialization and reporting.
        """
        pass

    def is_actionable(self) -> bool:
        """
        Determine if this anomaly should trigger immediate action.
        Based on severity and confidence threshold.
        """
        pass


class BlankAnomalyResult(AnomalyResult):
    """Result object for blank sample anomaly detection."""
    def __init__(self, *args, **kwargs):
        kwargs['anomaly_type'] = 'blank'
        super().__init__(*args, **kwargs)


class LCSAnomalyResult(AnomalyResult):
    """Result object for LCS anomaly detection."""
    def __init__(self, *args, **kwargs):
        kwargs['anomaly_type'] = 'lcs'
        super().__init__(*args, **kwargs)


class DuplicateAnomalyResult(AnomalyResult):
    """Result object for duplicate sample anomaly detection."""
    def __init__(self, *args, **kwargs):
        kwargs['anomaly_type'] = 'duplicate'
        super().__init__(*args, **kwargs)


class SRMAnomalyResult(AnomalyResult):
    """Result object for SRM anomaly detection."""
    def __init__(self, *args, **kwargs):
        kwargs['anomaly_type'] = 'srm'
        super().__init__(*args, **kwargs)


class MatrixSpikeAnomalyResult(AnomalyResult):
    """Result object for matrix spike anomaly detection."""
    def __init__(self, *args, **kwargs):
        kwargs['anomaly_type'] = 'matrix_spike'
        super().__init__(*args, **kwargs)


class MatrixSpikeDuplicateAnomalyResult(AnomalyResult):
    """Result object for matrix spike duplicate anomaly detection."""
    def __init__(self, *args, **kwargs):
        kwargs['anomaly_type'] = 'matrix_spike_duplicate'
        super().__init__(*args, **kwargs)
