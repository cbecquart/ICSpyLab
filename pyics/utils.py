"""
Module containing utility functions for the ICS algorithm.

This module provides various utility functions used in the computation of scatter matrices
and transformations within the Invariant Coordinate Selection (ICS) algorithm. Functions
included in this module perform operations such as sorting eigenvalues and eigenvectors,
and computing the square root or inverse square root of symmetric matrices.
"""

import numpy as np
import warnings
from numpy.linalg import multi_dot


def sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors):
    """
    Sort eigenvalues and eigenvectors in descending order of eigenvalues.

    Parameters:
        eigenvalues (np.ndarray): Array of eigenvalues.
        eigenvectors (np.ndarray): Corresponding eigenvectors.

    Returns:
        tuple: A tuple containing:
            - eigenvalues (numpy.ndarray): 1D array of eigenvalues sorted in descending order.
            - eigenvectors (numpy.ndarray): 2D array of eigenvectors sorted to match the order of sorted_eigenvalues.
    """
    # Get the indices that would sort the eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    # Sort the eigenvalues and eigenvectors
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


def sqrt_symmetric_matrix(A, inverse=False):
    """
    Compute the square root or inverse square root of a symmetric matrix.

    Parameters:
        A (np.ndarray): Symmetric matrix to compute the square root or inverse square root of.
        inverse (bool): (default: False) If True, compute the inverse square root. Otherwise, compute the square root.

    Returns:
        np.ndarray: The (inverse) square root of the matrix.
    """
    # Compute the eigenvalues and eigenvectors of the matrix
    A_eigenval, A_eigenvect = np.linalg.eig(A)
    # Sort the eigenvalues and eigenvectors
    A_eigenval, A_eigenvect = sort_eigenvalues_eigenvectors(A_eigenval, A_eigenvect)
    # Compute the power for the eigenvalues (inverse square root if inverse is True, otherwise square root)
    power = -0.5 if inverse else 0.5
    # Compute the (inverse) square root matrix
    A_sqrt = multi_dot([A_eigenvect, np.diag(A_eigenval ** power), A_eigenvect.T])
    return A_sqrt


def _check_gen_kurtosis(gen_kurtosis):
    """
    Check the gen_kurtosis array for NA, infinite, and complex values.

    Parameters:
        gen_kurtosis (np.ndarray): Array of kurtosis values.
    """
    if not np.all(np.isfinite(gen_kurtosis)):
        warnings.warn("Some generalized kurtosis values are infinite")

    if np.any(np.iscomplex(gen_kurtosis)):
        warnings.warn("Some generalized kurtosis values are complex")

    if np.any(np.isnan(gen_kurtosis)):
        warnings.warn("Some generalized kurtosis values are NA (Not Available)")


def _sign_max(row):
    """
    Determine the sign of the maximum absolute value in a row.

    This function checks if the maximum value in the row is the same as the maximum absolute value.
    If it is, the function returns 1, indicating that the maximum value is positive.
    Otherwise, it returns -1, indicating that the maximum absolute value is negative.

    Parameters
    ----------
    row : array-like
        The input row of numerical values.

    Returns
    -------
    int
        1 if the maximum value is positive, -1 if the maximum absolute value is negative.

    Examples
    --------
    >>> _sign_max([1, -3, 2])
    -1
    >>> _sign_max([-1, -2, -3])
    -1
    >>> _sign_max([0, 2, -2])
    1
    """
    return 1 if max(row) == np.max(np.abs(row)) else -1
