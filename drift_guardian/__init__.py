"""Data Drift Guardian — детекция дрейфа и контроль качества данных в продакшене."""
from .config import DriftConfig, Thresholds
from .contracts import DriftReport
from .guardian import DriftGuardian, analyze
from .timeline import run_timeline, run_timeline_from_frame, split_by_period

__all__ = [
    "DriftConfig", "Thresholds", "DriftReport", "DriftGuardian", "analyze",
    "run_timeline", "run_timeline_from_frame", "split_by_period",
]
__version__ = "0.1.0"
