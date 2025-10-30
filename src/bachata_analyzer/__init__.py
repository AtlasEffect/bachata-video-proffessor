"""Bachata Video Professor - CPU-only dance combination extraction pipeline."""

__version__ = "0.1.0"
__author__ = "Bachata Video Professor"

from .config import AnalysisConfig
from .analyzer import BachataAnalyzer

__all__ = ["AnalysisConfig", "BachataAnalyzer"]
