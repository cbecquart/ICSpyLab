"""
This file contains configuration settings and parameters used in the tests.
"""

from sklearn.datasets import load_iris, load_wine, load_digits, load_diabetes, load_breast_cancer
from icspylab import cov, covW, cov4, covAxis
import numpy as np

# Define datasets to be tested
datasets = ['iris', 'diabetes', 'wine']

# A list of decimal precisions to test
decimal_precisions_for_r = [8]
decimal_precisions = [12]

# Define algorithms to be tested
algorithm = ['standard', 'whiten', 'QR']
algorithm = ['standard', 'whiten']

# Define center options to be tested
center = [True, False]

# Define fix_signs options to be tested
fix_signs = ['scores', 'W']

# Define parameter sets to be tested
params_sets = [
    # Set 0: Default covariance and weighted covariance
    (
        {'S1': 'ICS_cov', 'S2': 'ICS_covW', 'S1_args': {}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': cov, 'S2': covW, 'S1_args': {}, 'S2_args': {}}
    ),
    # Set 1: Covariance and weighted covariance with specific alpha and cf values
    (
        {'S1': 'ICS_cov', 'S2': 'ICS_covW', 'S1_args': {}, 'S2_args': {'alpha': 2, 'cf': 3}, 'na_action': 'na.fail'},
        {'S1': cov, 'S2': covW, 'S1_args': {}, 'S2_args': {'alpha': 2, 'cf': 3}}
    ),
    # Set 2: Covariance and fourth-order covariance
    (
        {'S1': 'ICS_cov', 'S2': 'ICS_cov4', 'S1_args': {}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': cov, 'S2': cov4, 'S1_args': {}, 'S2_args': {}}
    ),
    # Set 3: Covariance and Tyler shape matrix
    (
        {'S1': 'ICS_cov', 'S2': 'ICS_covAxis', 'S1_args': {}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': cov, 'S2': covAxis, 'S1_args': {}, 'S2_args': {}}
    ),
    # Set 4: Weighted covariance with specific alpha and cf values
    (
        {'S1': 'ICS_covW', 'S2': 'ICS_covW', 'S1_args': {'alpha': 2, 'cf': 3}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': covW, 'S2': covW, 'S1_args': {'alpha': 2, 'cf': 3}, 'S2_args': {}}
    ),
    # Set 5: Weighted covariance with specific alpha and cf values for S2
    (
        {'S1': 'ICS_covW', 'S2': 'ICS_covW', 'S1_args': {}, 'S2_args': {'alpha': 2, 'cf': 3}, 'na_action': 'na.fail'},
        {'S1': covW, 'S2': covW, 'S1_args': {}, 'S2_args': {'alpha': 2, 'cf': 3}}
    ),
    # Set 6: Weighted covariance and covariance
    (
        {'S1': 'ICS_covW', 'S2': 'ICS_cov', 'S1_args': {}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': covW, 'S2': cov, 'S1_args': {}, 'S2_args': {}}
    ),
    # Set 7: Weighted covariance with specific alpha and cf values and covariance
    (
        {'S1': 'ICS_covW', 'S2': 'ICS_cov', 'S1_args': {'alpha': 2, 'cf': 3}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': covW, 'S2': cov, 'S1_args': {'alpha': 2, 'cf': 3}, 'S2_args': {}}
    ),
    # Set 8: Weighted covariance and Tyler shape matrix
    (
        {'S1': 'ICS_covW', 'S2': 'ICS_covAxis', 'S1_args': {}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': covW, 'S2': covAxis, 'S1_args': {}, 'S2_args': {}}
    ),
    # Set 9: Weighted covariance with specific alpha and cf values and Tyler shape matrix
    (
        {'S1': 'ICS_covW', 'S2': 'ICS_covAxis', 'S1_args': {'alpha': 2, 'cf': 3}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': covW, 'S2': covAxis, 'S1_args': {'alpha': 2, 'cf': 3}, 'S2_args': {}}
    ),
    # Set 10: Fourth-order covariance and weighted covariance with specific alpha and cf values
    (
        {'S1': 'ICS_cov4', 'S2': 'ICS_covW', 'S1_args': {}, 'S2_args': {'alpha': 2, 'cf': 3}, 'na_action': 'na.fail'},
        {'S1': cov4, 'S2': covW, 'S1_args': {}, 'S2_args': {'alpha': 2, 'cf': 3}}
    ),
    # Set 11: Fourth-order covariance and covariance
    (
        {'S1': 'ICS_cov4', 'S2': 'ICS_cov', 'S1_args': {}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': cov4, 'S2': cov, 'S1_args': {}, 'S2_args': {}}
    ),
    # Set 12: Fourth-order covariance and Tyler shape matrix
    (
        {'S1': 'ICS_cov4', 'S2': 'ICS_covAxis', 'S1_args': {}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': cov4, 'S2': covAxis, 'S1_args': {}, 'S2_args': {}}
    ),
    # Set 13: Tyler shape matrix and fourth-order covariance
    (
        {'S1': 'ICS_covAxis', 'S2': 'ICS_cov4', 'S1_args': {}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': covAxis, 'S2': cov4, 'S1_args': {}, 'S2_args': {}}
    ),
    # Set 14: Tyler shape matrix and covariance
    (
        {'S1': 'ICS_covAxis', 'S2': 'ICS_cov', 'S1_args': {}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': covAxis, 'S2': cov, 'S1_args': {}, 'S2_args': {}}
    ),
]

params_sets2 = [
    # Set 0: Default covariance and weighted covariance
    (
        {'S1': 'ICS_cov', 'S2': 'ICS_covW', 'S1_args': {}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': cov, 'S2': covW, 'S1_args': {}, 'S2_args': {}}
    ),
    # Set 1: Covariance and weighted covariance with specific alpha and cf values
    (
        {'S1': 'ICS_cov', 'S2': 'ICS_cov4', 'S1_args': {}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': cov, 'S2': cov4, 'S1_args': {}, 'S2_args': {}}
    ),
    # Set 2: Covariance and fourth-order covariance
    (
        {'S1': 'ICS_cov', 'S2': 'ICS_covAxis', 'S1_args': {}, 'S2_args': {}, 'na_action': 'na.fail'},
        {'S1': cov, 'S2': covAxis, 'S1_args': {}, 'S2_args': {}}
    )
]




