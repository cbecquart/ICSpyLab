import pytest
import numpy as np
import pandas as pd
from icspylab import Scatter, cov, covW, covAxis, cov4, mcd, tcov, tcovAxis, tM
import logging

logger = logging.getLogger(__name__)

def test_scatter_class():


    """
    Test the initialization of the Scatter class and check its attributes.

    This test verifies that the Scatter class correctly initializes with the provided
    location, scatter matrix, and label. It asserts that the attributes match the
    expected values.

    Attributes:
        location (np.ndarray): The location vector.
        scatter (np.ndarray): The scatter matrix.
        label (str): The label for the scatter matrix.
    """
    location = np.array([1.0, 2.0])
    scatter = np.array([[1.0, 0.5], [0.5, 0.1]])
    label = "Test Scatter"

    scatter_instance = Scatter(location, scatter, label)

    assert np.array_equal(scatter_instance.location, location)
    assert np.array_equal(scatter_instance.scatter, scatter)
    assert scatter_instance.label == label
    assert isinstance(scatter_instance, Scatter)

def test_cov():
    """
    Test the cov function for calculating the covariance matrix.

    This test verifies that the cov function correctly calculates the covariance matrix
    of the given data matrix. It asserts that the scatter matrix has the correct shape
    and that the location vector has the correct length.
    """
    X = np.random.rand(20, 5)
    scatter = cov(X)

    assert scatter.label == "Covariance"
    assert scatter.scatter.shape == (5, 5)
    assert scatter.location.shape == (5,)
    assert isinstance(scatter, Scatter)

def test_covW():
    """
    Test the covW function for calculating the weighted covariance matrix.

    This test verifies that the covW function correctly calculates the weighted covariance
    matrix of the given data matrix. It asserts that the scatter matrix has the correct
    shape and that the location vector has the correct length.
    """
    X = np.random.randn(20, 5)
    scatter = covW(X, alpha=1, cf=1)

    assert scatter.label == "Weighted Covariance"
    assert scatter.scatter.shape == (5, 5)
    assert scatter.location.shape == (5,)
    assert isinstance(scatter, Scatter)

def test_covAxis():
    """
    Test the covAxis function for calculating the Tyler shape matrix.

    This test verifies that the covAxis function correctly calculates the Tyler shape matrix
    of the given data matrix. It asserts that the scatter matrix has the correct shape
    and that the location vector has the correct length.
    """
    X = np.random.randn(20, 5)
    scatter = covAxis(X)

    assert scatter.label == "CovAxis"
    assert scatter.scatter.shape == (5, 5)
    assert scatter.location.shape == (5,)
    assert isinstance(scatter, Scatter)

def test_cov4():
    """
    Test the cov4 function for calculating the fourth-order covariance matrix.

    This test verifies that the cov4 function correctly calculates the fourth-order covariance
    matrix of the given data matrix. It asserts that the scatter matrix has the correct shape
    and that the location vector has the correct length.
    """
    X = np.random.randn(20, 5)
    scatter = cov4(X)

    assert scatter.label == "Cov4"
    assert scatter.scatter.shape == (5, 5)
    assert scatter.location.shape == (5,)
    assert isinstance(scatter, Scatter)

def test_mcd_reweighted():
    """
    Test the mcd function for calculating the MCD, with reweighted=True.

    This test verifies that the mcd function correctly implements the MCD scatter of the given data matrix. It asserts
    that the scatter matrix has the correct shape and that the location vector has the correct length.
    """
    X = np.random.randn(20, 5)
    scatter = mcd(X)

    assert scatter.label == "MCD"
    assert scatter.scatter.shape == (5, 5)
    assert scatter.location.shape == (5,)
    assert isinstance(scatter, Scatter)

def test_mcd_raw():
    """
    Test the mcd function for calculating the MCD, with reweighted=False.

    This test verifies that the mcd function correctly implements the MCD scatter of the given data matrix. It asserts
    that the scatter matrix has the correct shape and that the location vector has the correct length.
    """
    X = np.random.randn(20, 5)
    scatter = mcd(X, reweighted=False)

    assert scatter.label == "MCD"
    assert scatter.scatter.shape == (5, 5)
    assert scatter.location.shape == (5,)
    assert isinstance(scatter, Scatter)

def test_tcov():
    """
    Test the tcov function for calculating Tcov.

    This test verifies that the tcov function correctly implements the Tcov scatter of the given data matrix.
    It asserts that the scatter matrix has the correct shape and that the location vector is None.
    """
    X = np.random.randn(20, 5)
    scatter = tcov(X)

    assert scatter.label == "Tcov"
    assert scatter.scatter.shape == (5, 5)
    assert scatter.location == None
    assert isinstance(scatter, Scatter)

def test_tcovAxis():
    """
    Test the tcovAxis function for calculating TcovAxis.

    This test verifies that the tcovAxis function correctly implements the TcovAxis scatter of the given data matrix.
    It asserts that the scatter matrix has the correct shape
    and that the location vector is None.
    """
    X = np.random.randn(20, 5)
    scatter = tcovAxis(X)

    assert scatter.label == "TcovAxis"
    assert scatter.scatter.shape == (5, 5)
    assert scatter.location == None
    assert isinstance(scatter, Scatter)

def test_tM():
    """
    Test the tM function for calculating the tM scatter.

    This test verifies that the tM function correctly implements the tM scatter of the given data matrix.
    It asserts that the scatter matrix has the correct shape
    and that the location vector has the correct length.
    """
    X = np.random.randn(20, 5)
    scatter = tM(X)

    assert scatter.label == "tM"
    assert scatter.scatter.shape == (5, 5)
    assert scatter.location.shape == (5,)
    assert isinstance(scatter, Scatter)

def test_scatters_dataframe():
    X_df = pd.DataFrame(np.random.randn(10, 3), columns=["A", "B", "C"])
    # All functions should accept pd.DataFrame
    for func in [cov, covW, covAxis, cov4, mcd, tcov, tcovAxis, tM]:
        s = func(X_df)
        assert isinstance(s, Scatter)

def test_scatters_1D():
    X_1d = np.random.randn(10)
    # All functions should accept pd.DataFrame
    for func in [cov, mcd, tcov]:
        s = func(X_1d)
        print(s.scatter)
        assert isinstance(s, Scatter)
        assert s.scatter.shape == (1, 1)
