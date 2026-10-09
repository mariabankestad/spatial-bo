"""spatialbo: spatial weight fields for the exploration term of GP Bayesian optimization.

Companion code for "Understanding Hidden Spatial Preferences in Bayesian
Optimization".
"""
from spatialbo.fields import Field, load_field, reference_lengthscale
from spatialbo.functions import FUNCS, standardized

__all__ = ["Field", "load_field", "reference_lengthscale", "FUNCS", "standardized"]
__version__ = "0.1.0"
