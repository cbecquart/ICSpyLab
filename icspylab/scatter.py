import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from numpy.linalg import multi_dot
from sklearn.covariance import MinCovDet
from sklearn.utils.validation import check_array
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
        location (ndarray): The mean location of the data.
        scatter (ndarray): The scatter matrix.
        label (str): A label describing the scatter matrix.
    """

    def __init__(self, location, scatter, label):
        """
        Initialize the Scatter object with specified parameters.

        Parameters:
            location (ndarray or None): The mean location of the data.
            scatter (ndarray): The scatter matrix.
            label (str): A label describing the scatter matrix.
        """
        self.location = location
        self.scatter = scatter
        self.label = label


def cov(X, location=True):
    """
    Compute the covariance matrix.

    Parameters:
        X (array-like): The data matrix.
        location (bool, default=True) Whether to include the mean location.

    Returns:
        Scatter: An object containing the location and scatter matrix.

    Refernce:
        Refer to numpy.cov

    Example:
        >>> import numpy as np
        >>> from icspylab.scatter import cov
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> cov_X = cov(X)
        >>> print(cov_X.scatter)
        [[5.6 3.6]
         [3.6 2.4]]
    """

    # Check inputs
    X = np.asarray(X)
    if X.ndim > 2:
        raise ValueError("X has more than 2 dimensions")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    p = X.shape[1]

    # Compute the covariance matrix
    cov_cov = np.cov(X, rowvar=False).reshape(p, p)

    # Compute the mean location if required
    cov_loc = X.mean(0) if location else None

    return Scatter(cov_loc, cov_cov, "Covariance")


def covW(X, location=True, alpha=1, cf=1):
    """
    Estimates the scatter matrix based on one-step M-estimator using mean and covariance matrix as starting point.
    For more details, check the R documentation of the package ICS (function covW).

    Parameters:
        X (array-like): Data matrix, must be at least bi-variate.
        location (bool, default=True) Whether to include the mean location.
        alpha (float, default=1) Parameter of the one-step M-estimator.
        cf (float, default=1) Consistency factor of the one-step M-estimator.

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

    References:
        - Tukey, J. W. (1977). Exploratory data analysis (Vol. 2, pp. 131-160). Reading, MA: Addison-wesley.
        - Archimbaud, A., Drmac, Z., Nordhausen, K., Radojicic, U. and Ruiz-Gazen, A. (2023). SIAM Journal on Mathematics of Data Science (SIMODS), Vol.5(1):97–121. doi:10.1137/22M1498759.
        - Nordhausen, K., Oja, H., & Tyler, D. E. (2008). Tools for exploring multivariate data: The package ICS. Journal of Statistical Software, 28, 1-31.

    Example:
        >>> import numpy as np
        >>> from icspylab.scatter import covW
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> covW_X = covW(X)
        >>> print(covW_X.scatter)
        [[7.77777778 5.        ]
         [5.         3.33333333]]
    """

    # Check inputs
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n, p = X.shape
    if np.isnan(X).any():
        raise ValueError("X must not contain NaN values")
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
        X (array-like): The data matrix.
        location (bool, default=True) Whether to include the mean location.

    Returns:
        Scatter: An object containing the location and scatter matrix.

    References:
        - Critchley , F., Pires, A. and Amado, C. (2006), Principal axis analysis, Technical Report, 06/14, The Open University Milton Keynes.
        - Tyler, D.E., Critchley, F., Dumbgen, L. and Oja, H. (2009), Invariant co-ordinate selection, Journal of the Royal Statistical Society,Series B, 71, 549–592. <doi:10.1111/j.1467-9868.2009.00706.x>.
        - Nordhausen, K., Oja, H., & Tyler, D. E. (2008). Tools for exploring multivariate data: The package ICS. Journal of Statistical Software, 28, 1-31.

    Example:
        >>> import numpy as np
        >>> from icspylab.scatter import covAxis
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> covAxis_X = covAxis(X)
        >>> print(covAxis_X.scatter)
        [[5.6 3.6]
         [3.6 2.4]]
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
        X (array-like): The data matrix.
        location (bool, default=True) Whether to include the mean location.

    Returns:
        Scatter: An object containing the location and custom weighted scatter matrix.

    References:
        - Cardoso, J.F. (1989), Source separation using higher order moments, in Proc. IEEE Conf. on Acoustics, Speech and Signal Processing (ICASSP’89), 2109–2112. <doi:10.1109/ICASSP.1989.266878>.
        - Oja, H., Sirkia, S. and Eriksson, J. (2006), Scatter matrices and independent component analysis, Austrian Journal of Statistics, 35, 175–189.
        - Nordhausen, K., Oja, H., & Tyler, D. E. (2008). Tools for exploring multivariate data: The package ICS. Journal of Statistical Software, 28, 1-31.

    Example:
        >>> import numpy as np
        >>> from icspylab.scatter import cov4
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> cov4_X = cov4(X)
        >>> print(cov4_X.scatter)
        [[1.94444444 1.25      ]
         [1.25       0.83333333]]
    """

    X = np.asarray(X)
    p = X.shape[1]
    location_ = X.mean(0) if location else None

    # Directly call covW with given parameters
    cov4_scatter = covW(X, location, alpha=1, cf=(1 / (p + 2)))
    cov4_scatter.label = "Cov4"
    return cov4_scatter


def mcd(X, support_fraction=0.25, reweighted=False, **kwargs):
    """
    Wrapper function around scikit learn's implementation of the MCD (Minimum Covariance Determinant) estimator,
    computed using the FastMCD algorithm. The MCD is a robust estimator of location and covariance. It is based on
    selecting a subset of observations (of given size) whose empirical covariance matrix has minimal determinant.
    The covariance estimate is then rescaled to ensure consistency. An optional reweighting step can be applied,
    where observations are weighted according to their Mahalanobis distance from the estimated location. This yields
    the reweighted MCD estimator. In scikit-learn's MinCovDet, both raw (non-reweighted) and reweighted estimates are
    available through the attributes `raw_location_`, `raw_covariance_`, `location_`, and `covariance_`.
    By default, this function returns the raw estimates.
    Fore more information, check out scikit learn's `documentation <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html>`_.

    Parameters:
        X (array-like): The data matrix.
        support_fraction (float or None, default=0.25): The proportion of observations to be included in the support of the raw MCD estimate. This corresponds to the `support_fraction` parameter of sklearn.covariance.MinCovDet. By default, it is set to 0.25, which differs from the default used in sklearn.covariance.MinCovDet. If None, the original MinCovDet default value is used: (n_samples + n_features + 1) / (2 * n_samples).
        reweighted (bool, default=False) If True, use the reweighted version of the MCD estimator. Default is False.

    Returns:
        Scatter: An object containing the location and scatter matrix.

    References:
        - Refer to sklearn.covariance.MinCovDet
        - Rousseeuw, P.J. (1984) Least median of squares regression. J. Am Stat Ass, 79:871.
        - A Fast Algorithm for the Minimum Covariance Determinant Estimator, 1999, American Statistical Association and the American Society for Quality, TECHNOMETRICS.

    Example:
        >>> import numpy as np
        >>> from icspylab.scatter import mcd
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> mcd_X = mcd(X, support_fraction=None)
        >>> print(mcd_X.scatter)
        [[4.66666667 3.        ]
         [3.         2.        ]]
    """

    # Check inputs
    X = np.asarray(X)
    if X.ndim > 2:
        raise ValueError("X has more than 2 dimensions")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = X.shape[0]
    if n < 2:
        raise ValueError("X must have at least 2 observations")

    mcd_fit = MinCovDet(support_fraction=support_fraction, **kwargs).fit(X)

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


def tcov(X, beta=2):
    """
    Computes a pairwise one-step M-estimate of scatter with weights based on pairwise Mahalanobis distances.
    It can be interpreted as a `local` covariance matrix. Without the exponential term (or when beta=0), :math:`T_n` is
    proportional to the classical covariance matrix. The weighting scheme increases the importance of close sample pairs
    and decreases the contribution of pairs that are far apart.
    Note that this estimator is based on pairwise differences and therefore no location estimate is returned.

    Parameters:
        X (array-like):  data
        beta (int or float > 0, default=2): positive numeric value specifying the tuning parameter of the tcov estimator. The optimal value depends on the data but it usually is between 1.5 and 2.5.

    Returns:
        Scatter: An object containing the location(=None) and scatter matrix.

    Details:
        It is given for a :math:`n` x :math:`p` matrix :math:`X` by:

    .. math::

        T_n(\\beta) =
        \\frac{
            \\sum_{i=1}^{n} \\sum_{j=i+1}^{n}
            \\exp\\left(
                -\\frac{\\beta}{2}
                \\lVert X_i - X_j \\rVert_{V_n^{-1}}^2
            \\right)
            (X_i - X_j)(X_i - X_j)^\\top
        }{
            \\sum_{i=1}^{n} \\sum_{j=i+1}^{n}
            \\exp\\left(
                -\\frac{\\beta}{2}
                \\lVert X_i - X_j \\rVert_{V_n^{-1}}^2
            \\right)
        }

    where

    .. math::

        V_n = \\frac{1}{n}
        \\sum_{i=1}^{n}
        (X_i - \\bar X_n)(X_i - \\bar X_n)^\\top

    .. math::

        \\bar X_n = \\frac{1}{n} \\sum_{i=1}^{n} X_i

    .. math::

        \\lVert x \\rVert_M^2 = x^\\top M x


    The computation is optimized with Numba. In the paper, we have :math:`w(x) = exp(-x/2)`. But since we always
    call :math:`w(beta * r^2)`, we instead set :math:`b = -beta/2` and use :math:`w(x) = exp(x)`.

    References:
        - Caussinus, H. and Ruiz-Gazen, A. (1993) Projection Pursuit and Generalized Principal Component Analysis. In Morgenthaler, S., Ronchetti, E., Stahel, W.A. (eds.) New Directions in Statistical Data Analysis and Robustness, 35-46. Monte Verita, Proceedings of the Centro Stefano Franciscini Ascona Series. Springer-Verlag.
        - Caussinus, H. and Ruiz-Gazen, A. (1995) Metrics for Finding Typical Structures by Means of Principal Component Analysis. In Data Science and its Applications, 177-192. Academic Press.

    Example:
        >>> import numpy as np
        >>> from icspylab.scatter import tcov
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> tcov_X = tcov(X)
        >>> print(tcov_X.scatter)
        [[5.03250611 3.2351825 ]
         [3.2351825  2.15678833]]
    """

    # Check types
    X = np.asarray(X, dtype=np.float64)
    if X.ndim > 2:
        raise ValueError("X has more than 2 dimensions")
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    X = np.ascontiguousarray(X)

    n = X.shape[0]
    if n < 2:
        raise ValueError("X must have at least 2 observations")

    beta = float(beta)

    # Initialize b and the inverse covariance
    b = -beta / 2.0
    cov_inv = np.linalg.inv(np.cov(X, rowvar=False))

    tcov_X = _tcov_numba(X, cov_inv, b)

    return Scatter(location=None, scatter=tcov_X, label="Tcov")


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

    Parameters:
        X (np.ndarray(n,p)): data
        df (int >= 1): assumed degrees of freedom of the t-distribution. Default is 1 which corresponds
        to the Cauchy distribution.
        mu_init (np.ndarray(p)): initial value of the location mu
        V_init (np.ndarray(p,p)): initial value of the scatter V
        eps (float): convergence tolerance
        maxiter (int): maximum number of iterations

    Returns:
        tuple: location mu, scatter V, and number of iterations iter
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

    Parameters:
        X (array-like): data matrix, must be at least bi-variate
        df (int >= 1, default=1) assumed degrees of freedom of the t-distribution. Default is 1 which corresponds
        to the Cauchy distribution.
        mu_init (ndarray(p) or None, default=None) initial value of the location mu
        V_init (ndarray(p,p) or None, default=None) initial value of the scatter V
        eps (float, default=1e-6) convergence tolerance
        maxiter (int, default=100) maximum number of iterations

    Returns:
        tuple: _alg3 output

    References:
        - Kent, J.T., Tyler, D.E. and Vardi, Y. (1994), A curious likelihood identity for the multivariate tdistribution, Communications in Statistics, Simulation and Computation, 23, 441–453. <doi:10.1080/03610919408813180>.
        - Arslan, O., Constable, P.D.L. and Kent, J.T. (1995), Convergence behaviour of the EM algorithm for the multivariate t-distribution, Communications in Statistics, Theory and Methods, 24, 2981–3000. <doi:10.1080/03610929508831664>.
        - Nordhausen, K., Oja, H., & Tyler, D. E. (2008). Tools for exploring multivariate data: The package ICS. Journal of Statistical Software, 28, 1-31.

    Example:
        >>> import numpy as np
        >>> from icspylab.scatter import tM
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> tM_X = tM(X)
        >>> print(tM_X.scatter)
        [[4.66666667 3.        ]
         [3.         2.        ]]
    """

    # Check inputs
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    n, p = X.shape
    if np.isnan(X).any():
        raise ValueError("X must not contain NaN values")
    if p <= 1:
        raise ValueError("X must be at least bi-variate")

    if df < 1:
        raise ValueError("df (degrees of freedom) must be >= 1")
    if mu_init is not None and mu_init.shape[0] != X.shape[1]:
        raise ValueError("mu_init must have length equal to the number of features in X")
    if V_init is not None and V_init.shape != (X.shape[1], X.shape[1]):
        raise ValueError("V_init must be square with shape (n_features, n_features)")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if maxiter < 1:
        raise ValueError("maxiter must be >= 1")

    # Initialize values for location mu and scatter V if necessary
    if mu_init is None:
        mu_init = X.mean(axis=0)
    if V_init is None:
        V_init = np.cov(X, rowvar=False)

    mu, V, iter = _alg3(X, df, mu_init, V_init, eps, maxiter)

    return Scatter(location=mu, scatter=V, label="tM")


@njit
def _tcovAxis_numba(X, cov_inv):
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

def tcovAxis(X):
    """
    Computes a pairwise one-step M-estimate of scatter, similar to tcov, but using the same weights as covAxis.

    Parameters:
        X (array-like): data matrix, must be at least bi-variate.

    Returns:
        Scatter: An object containing the location(=None) and scatter matrix.

    Details:
        It is given for a :math:`n` x :math:`p` matrix :math:`X` by:

    .. math::

        T_n =
        \\frac{2}{n(n-1)}
        \\sum_{i=1}^{n-1} \\sum_{j=i+1}^{n}
        \\frac{
            (y_i - y_j)(y_i - y_j)^\\top
        }{
            \\left(
                (y_i - y_j)^\\top S_n^{-1} (y_i - y_j)
            \\right)^2
        }

    Reference:
        - Tyler, D.E., Critchley, F., Dumbgen, L. and Oja, H. (2009), Invariant co-ordinate selecetion, Journal of the Royal Statistical Society,Series B, 71, 549–592. <doi:10.1111/j.1467-9868.2009.00706.x>.

    Example:
        >>> import numpy as np
        >>> from icspylab.scatter import tcovAxis
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> tcovAxis_X = tcovAxis(X)
        >>> print(tcovAxis_X.scatter)
        [[0.98 0.63]
         [0.63 0.42]]
    """

    # Check types
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    X = np.ascontiguousarray(X)

    n, p = X.shape
    if n < 2:
        raise ValueError("X must have at least 2 observations")
    if p <= 1:
        raise ValueError("X must be at least bi-variate")

    cov_inv = np.linalg.inv(np.cov(X, rowvar=False))
    tcov_X = _tcovAxis_numba(X, cov_inv)

    return Scatter(location=None, scatter=tcov_X, label="TcovAxis")
