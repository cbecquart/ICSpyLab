"""
Unit tests for the distributions functions in the ICSpyLab package.
"""

import logging
import numpy as np
import pytest
from icspylab.distributions import unifsphere, multivariate_powerexp, generate_gaussian_mixture, generate_student_mixture, generate_rmvpowerexp_mixture

logger = logging.getLogger(__name__)

def test_unifsphere_shape_and_norm():
    """Tests the shape and norm from the unifsphere function."""
    n, p = 10, 3
    x = unifsphere(n, p)

    assert x.shape == (n, p)

    # norms ≈ 1
    norms = np.linalg.norm(x, axis=1)
    assert np.allclose(norms, 1, atol=1e-6)


def test_multivariate_powerexp_basic():
    """Tests the multivariate_powerexp function with the scatter argument and default location."""
    n, p = 5, 2
    scatter = np.eye(p)

    x = multivariate_powerexp(n, scatter=scatter)

    assert x.shape == (n, p)


def test_multivariate_powerexp_location():
    """Tests the multivariate_powerexp function with the scatter and location arguments."""
    n, p = 5, 2
    scatter = np.eye(p)
    location = np.ones(p)

    x = multivariate_powerexp(n, location=location, scatter=scatter)

    assert x.shape == (n, p)


def test_multivariate_powerexp_errors():
    """Tests the error handling of the function multivariate_powerexp."""
    p = 2
    scatter = np.eye(p)

    # missing scatter
    with pytest.raises(TypeError):
        multivariate_powerexp(5)

    # asymmetric scatter
    bad_scatter = np.array([[1, 2], [0, 1]])
    with pytest.raises(ValueError):
        multivariate_powerexp(5, scatter=bad_scatter)

    # wrong dimension
    with pytest.raises(ValueError):
        multivariate_powerexp(5, location=[1, 2, 3], scatter=scatter)


def _basic_params():
    eps = [0.5, 0.5]
    mu = [np.zeros(2), np.ones(2)]
    sigma = [np.eye(2), np.eye(2)]
    return eps, mu, sigma


def test_generate_gaussian_mixture_basic():
    eps, mu, sigma = _basic_params()
    n, p = 10, 4

    x, labels = generate_gaussian_mixture(eps, mu, sigma, n, p)

    assert x.shape == (n, p)
    assert len(labels) == n


def test_generate_student_mixture_basic():
    eps, mu, sigma = _basic_params()
    n, p = 10, 4

    x, labels = generate_student_mixture(eps, mu, sigma, df=5, n=n, p=p)

    assert x.shape == (n, p)
    assert len(labels) == n


def test_generate_rmvpowerexp_mixture_basic():
    eps, mu, sigma = _basic_params()
    n, p = 10, 4

    x, labels = generate_rmvpowerexp_mixture(eps, mu, sigma, beta=1.0, n=n, p=p)

    assert x.shape == (n, p)
    assert len(labels) == n


def test_labels_content():
    eps, mu, sigma = _basic_params()
    n, p = 10, 2

    _, labels = generate_gaussian_mixture(eps, mu, sigma, n, p)

    assert set(labels).issubset({"Group_1", "Group_2"})


def test_invalid_eps_sum():
    eps = [0.3, 0.3]
    mu = [np.zeros(2), np.ones(2)]
    sigma = [np.eye(2), np.eye(2)]

    with pytest.raises(AssertionError):
        generate_gaussian_mixture(eps, mu, sigma, n=10, p=2)
