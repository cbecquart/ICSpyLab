"""
Initialization file for the fixtures package.
This makes the fixtures directory a Python package and imports fixtures for easy access.
"""

from .fixtures import load_data, run_r_ics, run_py_ics

__all__ = ['load_data', 'run_r_ics', 'run_py_ics']
