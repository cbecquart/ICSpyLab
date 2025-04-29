"""
Module containing scatter matrix calculations and the Scatter class.

This module provides various functions to compute scatter matrices, which are essential for the
ICS algorithm. The scatter matrices implemented include the
covariance matrix, weighted covariance matrix, and the one-step Tyler shape matrix. These
scatter matrices are encapsulated in the Scatter class, which includes information about
the location (mean) and a label describing the type of scatter matrix. If you want to use ICS with other scatter
matrices than the ones provided in this module, you would need to create Scatter object. The S1 and S2 arguments are
functions returning Scatter objects.

Most scatters come from the `R package ICS <https://cran.r-project.org/web/packages/ICS/index.html>`_.
"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from numpy.linalg import multi_dot


class Scatter:
    """
    A class to represent the scatter matrix and its related data.

    Attributes:
        location (np.ndarray): The mean location of the data.
        scatter (np.ndarray): The scatter matrix.
        label (str): A label describing the scatter matrix.
    """

    def __init__(self, location, scatter, label):
        """
        Initialize the Scatter object with specified parameters.

        Parameters:
            location (np.ndarray): The mean location of the data.
            scatter (np.ndarray): The scatter matrix.
            label (str): A label describing the scatter matrix.
        """
        self.location = location
        self.scatter = scatter
        self.label = label


def cov(X, location=True):
    """
    Compute the covariance matrix.

    Parameters:
        X (numpy.ndarray): The data matrix.
        location (bool): (default: True) Whether to include the mean location.

    Returns:
        Scatter: An object containing the location and scatter matrix.
    """
    # Compute the covariance matrix
    scatter_ = np.cov(X.T)

    # Compute hte mean location if required
    location_ = X.mean(0) if location else None
    return Scatter(location_, scatter_, "Covariance")


def covW(X, location=True, alpha=1, cf=1):
    """
    Estimates the scatter matrix based on one-step M-estimator using mean and covariance matrix as starting point.
    For more details, check the R documentation of the package ICS (function covW).

    Parameters:
        X (numpy.ndarray): The data matrix.
        location (bool): (default: True) Whether to include the mean location.
        alpha (float): (default: 1) Parameter of the one-step M-estimator.
        cf (float): (default: 1) Consistency factor of the one-step M-estimator.

    Returns:
        Scatter: An object containing the location and weighted scatter matrix.

    Details:
        It is given for a :math:`n` x :math:`p` matrix :math:`X` by:
        :math:`CovW(X) = (1/n) cf \sum_{i=1}^{n} w(D^2(x_i)) (x_i - \overline{x})^T (x_i - \overline{x})`
    where:
        - :math:`n` is the number of observations,
        - :math:`x_i` is the i-th observation vector,
        - :math:`\overline{x}` is the mean vector of all observations,
        - :math:`w(d)= d^{α}` is a non-negative and continuous weight function applied to the squared Mahalanobis distance :math:`D^2(x_i)`.
        - :math:`cf` is a consistency factor

    """
    n, p = X.shape
    if pd.isnull(X).any():
        raise ValueError("Missing values are not allowed in X")
    if p <= 1:
        raise ValueError("X must be at least bi-variate")

    # Calculate the mean location and covariance matrix
    X_means = X.mean(axis=0)
    X_cov = np.cov(X.T)

    # Compute the Mahalanobis distance, square it, and apply the exponent alpha
    distance_maha = np.apply_along_axis(mahalanobis, 1, X, X_means, np.linalg.inv(X_cov))
    distance_maha_square = np.power(distance_maha, 2)
    distance_maha_square_alpha = np.power(distance_maha_square, alpha)

    # Center the data and compute the weighted covariance matrix
    X_centered = X - X_means
    X_covW = cf / n * multi_dot([X_centered.T, np.diag(distance_maha_square_alpha), X_centered])

    location_ = X.mean(0) if location else None
    return Scatter(location_, X_covW, "Weighted Covariance")


def covAxis(X, location=True):
    """
    Compute the one-step Tyler shape matrix which internally uses covW with alpha=-1 and cf=p.

    Parameters:
        X (numpy.ndarray): The data matrix.
        location (bool): (default: True) Whether to include the mean location.

    Returns:
        Scatter: An object containing the location and scatter matrix.
    """
    X = np.asarray(X)
    p = X.shape[1]
    # Directly call covW with given parameters
    covaxis_scatter = covW(X, location, alpha=-1, cf=p)
    covaxis_scatter.label = "CovAxis"

    return covaxis_scatter


def cov4(X, location=True):
    """
    Compute a custom weighted covariance matrix (cov4) which internally uses covW with alpha=1 and cf=(1 / (p + 2)).

    Parameters:
        X (numpy.ndarray): The data matrix.
        location (bool): (default: True) Whether to include the mean location.

    Returns:
        Scatter: An object containing the location and custom weighted scatter matrix.
    """
    X = np.asarray(X)
    p = X.shape[1]
    location_ = X.mean(0) if location else None

    # Directly call covW with given parameters
    cov4_scatter = covW(X, location, alpha=1, cf=(1 / (p + 2)))
    cov4_scatter.label = "Cov4"
    return cov4_scatter
