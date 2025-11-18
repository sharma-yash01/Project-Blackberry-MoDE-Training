"""
Reasoning Economics Visualization Package

This module exposes the main plotting classes:
- TrainingPlotter: Training progress and metric overview plots
- AnalysisPlotter: Detailed analysis (budget scatter, economic dashboard, etc)
- ClassificationPlotter: Confusion matrix and difficulty plots

Typical usage:
    from reasoning_economics.visualization import TrainingPlotter, AnalysisPlotter, ClassificationPlotter
"""

from .training_plots import TrainingPlotter
from .analysis_plots import AnalysisPlotter
from .classification_plots import ClassificationPlotter

__all__ = [
    "TrainingPlotter",
    "AnalysisPlotter",
    "ClassificationPlotter",
]
