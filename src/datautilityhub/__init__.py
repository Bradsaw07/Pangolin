"""
DataUtilityHub - Una biblioteca para simplificar el análisis exploratorio de datos y machine learning
"""

from .eda import DataLoader, NullAnalyzer, DuplicateHandler
from .visualization import DataVisualizer
from .preprocessing import DataCleaner, FeatureEngineer
from .ml import ModelEvaluator
from .utils import LibraryInstaller

__version__ = "0.1.0"

__all__ = [
    'DataLoader',
    'NullAnalyzer',
    'DuplicateHandler',
    'DataVisualizer',
    'DataCleaner',
    'FeatureEngineer',
    'ModelEvaluator',
    'LibraryInstaller'
]
