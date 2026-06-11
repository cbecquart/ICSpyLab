"""
This file contains fixture functions used in the tests.
"""

# import os
# conda_prefix = os.environ.get("CONDA_PREFIX")
# os.environ["R_HOME"] = os.path.join(conda_prefix, "Lib", "R")
# os.environ["PATH"] += os.pathsep + os.path.join(conda_prefix, "Lib", "R", "bin", "x64")

import pytest
import pandas as pd
import numpy as np
from tests.utils import load_dataset
from icspylab.ics import ICS, cov, covW

HAS_RPY2 = False
HAS_R_ICS = False

try:
    from rpy2.rinterface_lib.sexp import NULLType
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr, PackageNotInstalledError
    from rpy2.robjects.vectors import ListVector, StrVector
    from rpy2.robjects.conversion import localconverter, get_conversion

    HAS_RPY2 = True

    try:
        utils = importr("utils")
        ICS_R = importr("ICS")
        HAS_R_ICS = True
    except PackageNotInstalledError:
        ICS_R = None

except ImportError:
    ro = None


@pytest.fixture(scope="module")
def load_data():
    """
    Fixture to load different datasets.

    This fixture provides a function to load datasets by their name. It ensures
    that the loaded dataset does not contain any missing values.

    Returns:
        function: A function that takes a dataset name and returns the dataset (X, y).

    Raises:
        ValueError: If the dataset contains missing values.
    """
    def _load_data(dataset_name):
        X, y = load_dataset(dataset_name)
        if np.isnan(X).any():
            raise ValueError(f"Dataset {dataset_name} contains missing values.")
        return X, y
    return _load_data


@pytest.fixture(scope="module")
def run_r_ics():
    """
    Fixture to perform ICS in R.

    This fixture provides a function to run the ICS algorithm using R's ICS package.
    It converts the input data to an R-compatible format, runs the ICS algorithm in R,
    and returns the results.

    Parameters:
        X (np.ndarray): The input data matrix.
        S1 (str, optional): The first scatter matrix function in R. Default is 'ICS_cov'.
        S2 (str, optional): The second scatter matrix function in R. Default is 'ICS_cov4'.
        S1_args (dict, optional): Additional arguments for S1. Default is None.
        S2_args (dict, optional): Additional arguments for S2. Default is None.
        algorithm (str, optional): The algorithm to use. Default is 'whiten'.
        center (bool, optional): Whether to center the data. Default is False.
        fix_signs (str, optional): Method to fix signs. Default is 'scores'.
        na_action (str, optional): Action for handling NA values. Default is 'na.fail'.

    Returns:
        dict: A dictionary with the results, including the transformation matrix,
              generalized kurtosis, skewness, and transformed data.
    """

    if not HAS_RPY2 or not HAS_R_ICS:
        pytest.skip("rpy2 or R ICS not available")

    def _run_r_ics(X, S1='ICS_cov', S2='ICS_cov4', S1_args=None, S2_args=None, algorithm='whiten', center=False,
                   fix_signs='scores', na_action='na.fail'):

        S1_args = S1_args if S1_args is not None else {}
        S2_args = S2_args if S2_args is not None else {}

        # Convert the numpy array to a pandas DataFrame
        df = pd.DataFrame(X)

        # Convert the pandas DataFrame to an R data frame
        with localconverter(ro.default_converter + pandas2ri.converter):
            X_r = get_conversion().py2rpy(df)

        ro.globalenv['X'] = X_r
        ro.globalenv['S1_args'] = ListVector(S1_args)
        ro.globalenv['S2_args'] = ListVector(S2_args)

        ro.r(f'''
        ics_result <- ICS(X, S1 = {S1}, S2 = {S2}, S1_args = S1_args, S2_args = S2_args, algorithm = "{algorithm}", center = {str(center).upper()}, fix_signs = "{fix_signs}", na.action = {na_action})
        ''')
        transformation_matrix = ro.r('ics_result$W')
        kurtosis = ro.r('ics_result$gen_kurtosis')
        skewness = ro.r('ics_result$gen_skewness') if not isinstance(ro.r('ics_result$gen_skewness'),
                                                                     NULLType) else None
        transformed_data = ro.r('ics_result$scores')

        # Convert the results back to pandas DataFrame for better precision control in Python
        with localconverter(ro.default_converter + pandas2ri.converter):
            transformation_matrix_df = get_conversion().rpy2py(transformation_matrix)
            transformed_data_df = get_conversion().rpy2py(transformed_data)

        return {
            'transformation_matrix': transformation_matrix_df,
            'kurtosis': kurtosis,
            'skewness': skewness,
            'transformed_data': transformed_data_df,
        }

    return _run_r_ics


@pytest.fixture(scope="module")
def run_py_ics():
    """
    Fixture to perform ICS in Python.

    This fixture provides a function to run the ICS algorithm using the Python implementation.
    It creates an ICS object, fits and transforms the data, and returns the results.

    Parameters:
        X (np.ndarray): The input data matrix.
        S1 (function, optional): The first scatter matrix function. Default is cov.
        S2 (function, optional): The second scatter matrix function. Default is covW.
        algorithm (str, optional): The algorithm to use. Default is 'whiten'.
        center (bool, optional): Whether to center the data. Default is False.
        fix_signs (str, optional): Method to fix signs. Default is 'scores'.
        S1_args (dict, optional): Additional arguments for S1. Default is {}.
        S2_args (dict, optional): Additional arguments for S2. Default is {}.

    Returns:
        dict: A dictionary with the results, including the transformation matrix,
              generalized kurtosis, skewness, and transformed data.
    """
    def _run_py_ics(X, S1=cov, S2=covW, algorithm='whiten', center=False, fix_signs='scores', S1_args={}, S2_args={}):
        ics = ICS(S1=S1, S2=S2, algorithm=algorithm, center=center, fix_signs=fix_signs, S1_args=S1_args, S2_args=S2_args)
        X_new = ics.fit_transform(X)
        return {
            'transformation_matrix': ics.components_,
            'kurtosis': ics.kurtosis_,
            'skewness': ics.skewness_,
            'transformed_data': X_new,
        }
    return _run_py_ics
