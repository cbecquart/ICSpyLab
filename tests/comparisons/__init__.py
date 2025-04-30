"""
Initialization file for the comparisons package.
Imports necessary modules and functions for comparison tests.
"""

import logging
import pytest
import numpy as np
from tests.utils import calculate_similarity_percentage
from pyics import ICS
from sklearn.datasets import load_iris
from tests.fixtures import load_data, run_r_ics, run_py_ics
from tests.settings import datasets, params_sets, params_sets2, decimal_precisions_for_r, decimal_precisions, algorithm, center, fix_signs

__all__ = [
    'logging', 'pytest', 'np', 'calculate_similarity_percentage', 'ICS', 'load_iris',
    'load_data', 'run_r_ics', 'run_py_ics',
    'datasets', 'params_sets', 'params_sets2',
    'decimal_precisions_for_r', 'decimal_precisions', 'algorithm', 'center', 'fix_signs'
]
