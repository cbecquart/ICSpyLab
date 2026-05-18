import numpy as np
from numpy.linalg import eigh
from scipy.stats import gamma, multivariate_t
import warnings


# --- Multivariate distributions ---

def unifsphere(n, p):
    """
    Generate n vectors uniformly distributed on the unit sphere in dimension p.

    Parameters:
        n (int): Number of observations.
        p (int): Dimension of the sphere.

    Returns:
        ndarray (n, p)

    Example:
        >>> from icspylab.distributions import unifsphere
        >>> X = unifsphere(n=100, p=2)
        >>> print(X.shape)
    (100, 2)
    """
    x = np.random.normal(size=(n, p))
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / norm


def multivariate_powerexp(n, scatter, location=None, beta=1):
    """
    Generate n observations from a multivariate power exponential distribution.

    Parameters:
        n (int): Number of observations.
        scatter (array-like): Symmetric positive definite scatter matrix (p x p).
        location (array-like): Mean vector of dimension p.
        beta (float):  Shape parameter (> 0). beta = 1 corresponds to the multivariate normal distribution, beta < 1 corresponds to heavier tails.

    Returns:
        ndarray (n, p)

    References:
        - Oja, H. (2010), Multivariate Nonparametric Methods with R, Springer.
        - Nordhausen, K., & Oja, H. (2011). Multivariate L1 statistical methods: The package MNM. Journal of Statistical Software, 43, 1-28.

    Example:
        >>> from icspylab.distributions import multivariate_powerexp
        >>> X = multivariate_powerexp(n=100, scatter=np.eye(3), beta=4)
        >>> print(X.shape)
        (100, 3)
    """

    scatter = np.asarray(scatter)
    p = scatter.shape[0]

    if location is None:
        location = np.zeros(p)
    else:
        location = np.asarray(location)

    # check symmetry
    if not np.allclose(scatter, scatter.T, atol=np.sqrt(np.finfo(float).eps)):
        raise ValueError("scatter must be a symmetric matrix")

    # check dimensions
    if len(location) != p:
        raise ValueError("location and scatter have non-conforming size")

    # check beta
    if beta <= 0:
        raise ValueError("beta must be positive")

    # eigen decomposition of scatter matrix
    eigenvalues, eigenvectors = eigh(scatter)

    # check numerical positive definiteness
    if not np.all(eigenvalues >= -np.sqrt(np.finfo(float).eps) * abs(eigenvalues[-1])):
        warnings.warn("warning: scatter is numerically not positive definite")

    # matrix square root of scatter
    scattersqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

    # generate radial component
    radius = gamma.rvs(a=p / (2 * beta), scale=2, size=n) ** (1 / (2 * beta))

    # generate directions uniformly on unit sphere
    un = unifsphere(n, p)

    # construct multivariate samples
    mvpowerexp = (radius[:, None] * un) @ scattersqrt

    # add location vector
    mvpowerexp += location

    return mvpowerexp


# --- Elliptical mixtures ---

def generate_gaussian_mixture(eps, mu, sigma, n, p):
    """
    Generates a Gaussian Mixture Model (GMM) with the given parameters.

    Parameters:
        eps (list of float): Proportions of points assigned to each cluster (must sum to 1).
        mu (list of np.ndarray): List of mean vectors (centroids) for each cluster (size k).
        sigma (list of np.ndarray): List of covariance matrices (size k).
        n (int): Total number of data points to generate.
        p (int): Dimension of the data, including noise.

    Returns:
        tuple: A tuple containing:
            data_with_noise (ndarray): Matrix (n, p) of generated data points.
            labels (ndarray): Array of cluster labels (size n).

    Example:
        >>> eps = [0.5, 0.5]
        >>> mu = [np.ones(2), np.ones(2)*10]
        >>> sigma = [np.eye(2) for _ in range(2)]
        >>> X, labels = generate_gaussian_mixture(eps, mu, sigma, n=1000, p=6)
    """
    # Validate inputs
    assert np.isclose(sum(eps), 1.0), "The elements of eps must sum to 1."
    assert len(eps) > 0, "Proportions (eps) must contain at least one group."
    assert len(eps) == len(sigma), "Proportions (eps) and sigma must have the same length."
    assert len(eps) == len(mu), "Proportions (eps) and mu must have the same length."

    # Number of clusters k
    k = len(eps)

    # Determine the number of points for each group based on eps
    points_per_group = (np.array(eps) * n).astype(int)
    # Adjust to ensure the total points add up to n
    points_per_group[0] = n - np.sum(points_per_group[1:])

    # Generate data
    data = []

    for i in range(k):
        # Generate points from a multivariate normal distribution
        group_points = np.random.default_rng().multivariate_normal(mu[i], sigma[i], points_per_group[i])
        data.append(group_points)

    # Combine data points into a single array (n x r)
    data = np.vstack(data)

    # Add noise
    p_noise = p - data.shape[1]
    noise = np.random.normal(loc=0, scale=1, size=(n, p_noise))
    data_with_noise = np.hstack((data, noise))

    # Save group label for each point
    group_labels = ["Group_" + str(i + 1) for i in range(k)]
    labels = [val for val, count in zip(group_labels, points_per_group) for _ in range(count)]
    labels = np.array(labels)

    return data_with_noise, labels


def generate_student_mixture(eps, mu, sigma, df, n, p):
    """
    Generates a Student-t Mixture Model (SMM) with the given parameters.

    Parameters:
        eps (list of float): Proportions of points assigned to each cluster (must sum to 1).
        mu (list of ndarray): List of mean vectors (centroids) for each cluster (size k).
        sigma (list of ndarray): List of covariance matrices (size k).
        df (int or list of int): Degrees of freedom (size k if list). Must be strictly positive integers.
        n (int): Total number of data points to generate.
        p (int): Dimension of the data, including noise.

    Returns:
        tuple: A tuple containing:
            data_with_noise (ndarray): Matrix (n, p) of generated data points.
            labels (ndarray): Array of cluster labels (size n).

    Example:
        >>> eps = [0.5, 0.5]
        >>> mu = [np.ones(2), np.ones(2)*10]
        >>> sigma = [np.eye(2) for _ in range(2)]
        >>> X, labels = generate_student_mixture(eps, mu, sigma, df=2, n=1000, p=6)
    """

    # Number of clusters k
    k = len(eps)

    if isinstance(df, int):
        df = [df for _ in range(k)]

    # Validate inputs
    assert np.isclose(sum(eps), 1.0), "The elements of eps must sum to 1."
    assert len(eps) > 0, "Proportions (eps) must contain at least one group."
    assert len(eps) == len(sigma), "Proportions (eps) and sigma must have the same length."
    assert len(eps) == len(mu), "Proportions (eps) and mu must have the same length."
    assert len(eps) == len(df), "Proportions (eps) and df must have the same length."

    # Determine the number of points for each group based on eps
    points_per_group = (np.array(eps) * n).astype(int)
    # Adjust to ensure the total points add up to n
    points_per_group[0] = n - np.sum(points_per_group[1:])

    # Generate data
    data = []

    for i in range(k):
        # Generate points from a multivariate Student-t distribution
        group_points = multivariate_t.rvs(loc=mu[i], shape=sigma[i], df=df[i], size=points_per_group[i])
        data.append(group_points)

    # Combine data points into a single array (n x r)
    data = np.vstack(data)

    # Add noise
    p_noise = p - data.shape[1]
    noise = np.random.normal(loc=0, scale=1, size=(n, p_noise))
    data_with_noise = np.hstack((data, noise))

    # Save group label for each point
    group_labels = ["Group_" + str(i + 1) for i in range(k)]
    labels = [val for val, count in zip(group_labels, points_per_group) for _ in range(count)]
    labels = np.array(labels)

    return data_with_noise, labels


def generate_powerexp_mixture(eps, mu, sigma, beta, n, p):
    """
    Generates a mixture of multivariate power exponential distribution (PEM) with the given parameters.

    Parameters:
        eps (list of float): Proportions of points assigned to each cluster (must sum to 1).
        mu (list of np.ndarray): List of mean vectors (centroids) for each cluster (size k).
        sigma (list of np.ndarray): List of covariance matrices (size k).
        beta (float or list of float): Shape parameters (size k if list).
        n (int): Total number of data points to generate.
        p (int): Dimension of the data, including noise.

    Returns:
        tuple: A tuple containing:
            data_with_noise (ndarray): Matrix (n, p) of generated data points.
            labels (ndarray): Array of cluster labels (size n).

    Example:
        >>> eps = [0.5, 0.5]
        >>> mu = [np.ones(2), np.ones(2)*10]
        >>> sigma = [np.eye(2) for _ in range(2)]
        >>> X, labels = generate_powerexp_mixture(eps, mu, sigma, beta=0.8, n=1000, p=6)
    """

    # Number of clusters k
    k = len(eps)

    if isinstance(beta, float):
        beta = [beta for _ in range(k)]

    # Validate inputs
    assert np.isclose(sum(eps), 1.0), "The elements of eps must sum to 1."
    assert len(eps) > 0, "Proportions (eps) must contain at least one group."
    assert len(eps) == len(sigma), "Proportions (eps) and sigma must have the same length."
    assert len(eps) == len(mu), "Proportions (eps) and mu must have the same length."
    assert len(eps) == len(beta), "Proportions (eps) and beta must have the same length."

    # Determine the number of points for each group based on eps
    points_per_group = (np.array(eps) * n).astype(int)
    # Adjust to ensure the total points add up to n
    points_per_group[0] = n - np.sum(points_per_group[1:])

    # Generate data
    data = []

    for i in range(k):
        # Generate points from a multivariate normal distribution
        group_points = multivariate_powerexp(n=points_per_group[i], location=mu[i], scatter=sigma[i], beta=beta[i])
        data.append(group_points)

    # Combine data points into a single array (n x r)
    data = np.vstack(data)

    # Add noise
    p_noise = p - data.shape[1]
    noise = np.random.normal(loc=0, scale=1, size=(n, p_noise))
    data_with_noise = np.hstack((data, noise))

    # Save group label for each point
    group_labels = ["Group_" + str(i + 1) for i in range(k)]
    labels = [val for val, count in zip(group_labels, points_per_group) for _ in range(count)]
    labels = np.array(labels)

    return data_with_noise, labels


# --- RANDU ---

def generate_randu(n=400, seed=1):
    """
    Generate a synthetic dataset based on the classical RANDU pseudo-random number generator.

    RANDU is an obsolete linear congruential generator that is widely used as a benchmark example of poor randomness properties.

    The implementation follows the standard definition described in the R datasets package manual.

    Parameters:
        n (int, default=400): Number of data points to generate.
        seed (int, default=1): Seed of the generator.

    Returns:
        ndarray (n, 3)

    References:
        - Fortran Language Reference Manual (1999), Compaq.
        - R Core Team (datasets package), https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/randu.html

    Example:
        >>> from icspylab.distributions import generate_randu
        >>> X = generate_randu(n=100)
        >>> print(X.shape)
        (100, 3)
    """

    def randu():
        nonlocal seed
        seed = ((2 ** 16 + 3) * seed) % (2 ** 31)
        return seed / (2 ** 31)

    x = np.empty((n, 3), dtype=float)

    for i in range(n):
        U = np.array([randu() for _ in range(5)])
        x[i, :] = np.round(U[:3], 6)

    return x
