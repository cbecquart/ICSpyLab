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
from sklearn.covariance import MinCovDet
from numba import njit
import warnings
try:
    from icspylab.tcov import tcov_module
except ImportError:
    tcov_module = None


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
            location (np.ndarray or None): The mean location of the data.
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
    cov_cov = np.cov(X.T)

    # Compute the mean location if required
    cov_loc = X.mean(0) if location else None

    return Scatter(cov_loc, cov_cov, "Covariance")


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


def mcd(X, reweighted=True, **kwargs):
    """
    Wrapper function around scikit learn's implementation of the MCD (Minimum Covariance Determinant) scatter, using
    the FastMCD algorithm. MCD is a robust estimator of covariance. The idea is to find a given proportion of
    non-outlying observations and compute their empirical covariance matrix. It is then rescaled to account for the
    selection of observations ("consistency step"). Once the MCD estimator is computed, the observations can be weighted
    by their Mahalanobis distance. The resulting estimator is called the reweighted MCD. The "reweighting step" is
    performed by default. To access the raw estimators of the MCD, call the raw_location_ and raw_covariance_ attributes
    of a MinCovDet object. information, check out scikit learn's `documentation <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html>`_.
    Parameters:
        X (numpy.ndarray): The data matrix.

    Returns:
        Scatter: An object containing the location and scatter matrix.

    References
    - P. J. Rousseeuw. Least median of squares regression. J. Am Stat Ass, 79:871, 1984.
    - A Fast Algorithm for the Minimum Covariance Determinant Estimator, 1999, American Statistical Association and the American Society for Quality, TECHNOMETRICS.
    """
    mcd_fit = MinCovDet(**kwargs).fit(X)
    if reweighted:
        mcd_loc = mcd_fit.location_
        mcd_cov = mcd_fit.covariance_
    else:
        mcd_loc = mcd_fit.raw_location_
        mcd_cov = mcd_fit.raw_covariance_

    return Scatter(location=mcd_loc, scatter=mcd_cov, label="MCD")


@njit
def _tcov_numba(X, cov_inv, b):
    """Loop over pairs of observations and add their weighted contribution."""

    n, p = X.shape
    V = np.zeros((p, p))
    denominator = 0.0

    for i in range(1, n):
        xi = X[i]
        for j in range(i):
            # Compute difference of current pair of observations
            diff = xi - X[j]
            # Compute squared pairwise Mahalanobis distance r_sq = diff^T @ cov_inv @ diff
            tmp = cov_inv @ diff
            r_sq = np.dot(diff, tmp)
            # Compute weight for current pair of observations
            w = np.exp(b * r_sq)
            # Add weighted contribution of current pair of observations
            for k in range(p):
                for l in range(p):
                    V[k, l] += w * diff[k] * diff[l]
            denominator += w

    return V / denominator


def _tcov_py(X, beta):
    """Python implementation of tcov, optimized with Numba. In the paper, we have w(x) = exp(-x/2). But since we always
    call w(beta * r^2), we instead set b = -beta/2 and use w(x) = exp(x)."""

    # Initialize b and the inverse covariance
    b = -beta / 2.0
    cov_inv = np.linalg.inv(np.cov(X, rowvar=False))

    return _tcov_numba(X, cov_inv, b)


def tcov(X, beta=2, use_cpp=True):
    """
    Computes a pairwise one-step M-estimate of scatter with weights based on pairwise Mahalanobis distances. Note that
    this estimator is based on pairwise differences and therefore no location estimate is returned.

    Parameters:
        X (numpy.ndarray):  data
        beta (int or float > 0, default=2): positive numeric value specifying the tuning parameter of the tcov estimator
        use_cpp (bool, default=True): whether to use the C++ implementation. If `use_cpp=True` (default), the code calls
        Andreas Alfons' C++ implementation. It is faster but requires a Python module compiled from the C++ source.
        Precompiled modules are included in ICSpyLab for Windows and Python versions 3.10–3.14. If the module is not
        available, the code issues a warning and continues with `use_cpp=False`, which calls a Numba routine instead.
        For help building the module, see `icspylab/tcov/README.md`.
    Returns:
        Scatter: An object containing the location(=None) and scatter matrix.

    References
    - Caussinus, H. and Ruiz-Gazen, A. (1993) Projection Pursuit and Generalized Principal Component Analysis. In
    Morgenthaler, S., Ronchetti, E., Stahel, W.A. (eds.) New Directions in Statistical Data Analysis and Robustness,
    35-46. Monte Verita, Proceedings of the Centro Stefano Franciscini Ascona Series. Springer-Verlag.
    - Caussinus, H. and Ruiz-Gazen, A. (1995) Metrics for Finding Typical Structures by Means of Principal Component
    Analysis. In Data Science and its Applications, 177-192. Academic Press.
    """

    # Check types
    X = np.asarray(X, dtype=np.float64)
    X = np.ascontiguousarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    beta = float(beta)

    # If use_cpp=True, tcov_module can't be None. Otherwise, proceed with use_cpp=False.
    if use_cpp and (tcov_module is not None):
        tcov_X = tcov_module.tcov_cpp(X, beta)
    else:
        if use_cpp:
            raise ImportError('Requires tcov_module which is not available. For help building the module, see '
                              '`icspylab/tcov/README.md`. Proceeding with `use_cpp=False`.')
        warnings.warn('Use the C++ implementation for faster computations (`use_cpp=True`).')
        tcov_X = _tcov_py(X, beta)         

    return Scatter(location=None, scatter=tcov_X, label="TCOV")


def _norm_mu_V(a, B, A):
    """
    Computes the norm used to define convergence as in Arslan et al. (1995).
    """

    A_inv = np.linalg.inv(A)
    BA_inv = B @ A_inv
    square = a.T @ A_inv @ a + np.trace(BA_inv @ BA_inv)
    return np.sqrt(square)


def _alg3(X, df, mu_init, V_init, eps, maxiter):
    """
    Computes tM location and scatter according to Algorithm 3 in Kent and Tyler
    :param X: array (n,p): data
    :param df: integer >= 1: assumed degrees of freedom of the t-distribution. Default is 1 which corresponds
    to the Cauchy distribution.
    :param mu_init: array(p): initial value of the location mu
    :param V_init: array(p,p): initial value of the scatter V
    :param eps: float: convergence tolerance
    :param maxiter: integer: maximum number of iterations
    :return: tuple: location mu, scatter V, and number of iterations iter
    """

    n, p = X.shape

    # Init parameters
    mu_i = mu_init
    V_i = V_init
    gamma = 1
    differ = np.inf
    iter_count = 0

    while differ > eps:
        iter_count += 1

        # Update weights
        d = np.apply_along_axis(mahalanobis, 1, X, mu_i, np.linalg.inv(V_i))
        w = (df + p) / (df - 1 + (1 / gamma) + (1 / gamma) * (d ** 2))
        # Update estimate mu
        gamma_new = np.mean(w)
        mu_new = np.average(X, axis=0, weights=w) / gamma_new
        # Update estimate V
        X_centered = X - mu_new
        V_new = ((X_centered.T * w) @ X_centered / n) / gamma_new

        # Compute difference
        differ = _norm_mu_V(a=mu_new - mu_i, B=V_new - V_i, A=V_new)
        mu_i, V_i = mu_new, V_new

        if iter_count >= maxiter:
            raise RuntimeError("maxiter reached without convergence")

    return mu_new, V_new, iter_count


def tM(X, df=1, mu_init=None, V_init=None, eps=1e-6, maxiter=100):
    """
    Computes the location and scatter for a multivariate t-distribution for a given degree of freedom.
    This function implements the third EM algorithm described in Kent et al. (1994). The norm used to define convergence
    is as in Arslan et al. (1995).
    :param X: array (n,p): data
    :param df: integer >= 1: assumed degrees of freedom of the t-distribution. Default is 1 which corresponds
    to the Cauchy distribution.
    :param mu_init: array(p): initial value of the location mu
    :param V_init: array(p,p): initial value of the scatter V
    :param eps: float: convergence tolerance
    :param maxiter: integer: maximum number of iterations
    :return: tuple: _alg3 output

    References:
    - Kent, J.T., Tyler, D.E. and Vardi, Y. (1994), A curious likelihood identity for the multivariate tdistribution,
    Communications in Statistics, Simulation and Computation, 23, 441–453. <doi:10.1080/03610919408813180>.
    - Arslan, O., Constable, P.D.L. and Kent, J.T. (1995), Convergence behaviour of the EM algorithm for
    the multivariate t-distribution, Communications in Statistics, Theory and Methods, 24, 2981–3000.
    <doi:10.1080/03610929508831664>.
    """

    X = np.asarray(X)

    # Initialize values for location mu and scatter V if necessary
    if mu_init is None:
        mu_init = X.mean(axis=0)
    if V_init is None:
        V_init = np.cov(X, rowvar=False)

    mu, V, iter = _alg3(X, df, mu_init, V_init, eps, maxiter)

    return Scatter(location=mu, scatter=V, label="tM")


@njit
def _tcov2_numba(X, cov_inv):
    """Loop over pairs of observations and add their weighted contribution."""

    n, p = X.shape
    V = np.zeros((p, p))

    for i in range(1, n):
        xi = X[i]
        for j in range(i):
            # Compute difference of current pair of observations
            diff = xi - X[j]
            # Compute squared pairwise Mahalanobis distance r_sq = diff^T @ cov_inv @ diff
            tmp = cov_inv @ diff
            r_sq = np.dot(diff, tmp)

            V += np.outer(diff, diff) / r_sq ** 2

    return 2 / (n * (n-1)) * V

def tcov2(X):
    """Python implementation of tcov, optimized with Numba. In the paper, we have w(x) = exp(-x/2). But since we always
    call w(beta * r^2), we instead set b = -beta/2 and use w(x) = exp(x)."""

    # Check types
    X = np.asarray(X, dtype=np.float64)
    X = np.ascontiguousarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    cov_inv = np.linalg.inv(np.cov(X, rowvar=False))
    tcov_X = _tcov2_numba(X, cov_inv)

    return Scatter(location=None, scatter=tcov_X, label="TCOV2")