"""
education-spending-analytics · src
===================================
Public re-exports so notebooks can do:
    from src import load_raw, clean, engineer_features, plots
"""

from .data_cleaning import load_raw, clean
from .feature_engineering import engineer_features
from . import plotting as plots

__all__ = ["load_raw", "clean", "engineer_features", "plots"]
