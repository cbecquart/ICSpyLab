from . import *

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

    Attributes:
        data_matrix (np.ndarray): The data matrix to be used for covariance calculation.
    """
    X = np.random.rand(100, 5)
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

    Attributes:
        data_matrix (np.ndarray): The data matrix to be used for weighted covariance calculation.
    """
    X = np.random.randn(100, 5)
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

    Attributes:
        data_matrix (np.ndarray): The data matrix to be used for Tyler shape matrix calculation.
    """
    X = np.random.randn(100, 5)
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

    Attributes:
        data_matrix (np.ndarray): The data matrix to be used for fourth-order covariance calculation.
    """
    X = np.random.randn(100, 5)
    scatter = cov4(X)

    assert scatter.label == "Cov4"
    assert scatter.scatter.shape == (5, 5)
    assert scatter.location.shape == (5,)
    assert isinstance(scatter, Scatter)
