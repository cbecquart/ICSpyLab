"""
Unit tests for the ICS class in the ICSpyLab package.
"""

import logging
import pytest
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from icspylab import ICS, cov, covW, covAxis, cov4, normal_crit, median_crit, unimodal_crit
from tests.fixtures import run_py_ics
from tests.settings import algorithm, center, fix_signs

logger = logging.getLogger(__name__)


def test_ics_sklearn_compatible():
    check_estimator(ICS())


# Section: Initialization Tests
def test_initialization():
    """
    Test the initialization of the ICS class.

    This test verifies that the ICS class correctly initializes with default parameters
    and that all attributes are set to their expected initial values.
    """
    ics = ICS()
    assert isinstance(ics, ICS)
    assert ics.S1 is not None
    assert ics.S2 is not None
    assert ics.algorithm == 'eigh'
    assert ics.center is False
    assert ics.fix_signs == 'scores'
    assert ics.S1_args is None
    assert ics.S2_args is None
    assert ics.method_select is None
    assert ics.select_args is None


def test_S1_as_matrix():
    """
    Test the type of S1 argument.

    The test verifies that a TypeError is raised if S1 a numpy array.

    """
    X = np.random.randn(100, 5)
    cov_matrix = np.cov(X, rowvar=False)
    ics = ICS(S1=cov_matrix)

    with pytest.raises(TypeError):
        ics.fit(X)


def test_S1_as_string():
    """
    Test the type of S1 argument.

    The test verifies that a TypeError is raised if S1 an unknown character string.

    """
    X = np.random.randn(10, 3)
    ics = ICS(S1="coconut")

    with pytest.raises(ValueError):
        ics.fit(X)


def test_S2_as_matrix():
    """
    Test the type of S2 argument.

    The test verifies that a TypeError is raised if S2 a numpy array.

    """
    X = np.random.randn(100, 5)
    cov_matrix = np.cov(X, rowvar=False)
    ics = ICS(S2=cov_matrix)

    with pytest.raises(TypeError):
        ics.fit(X)


def test_S2_as_string():
    """
    Test the type of S2 argument.

    The test verifies that a TypeError is raised if S2 an unknown character string.

    """
    X = np.random.randn(10, 3)
    ics = ICS(S2="coconut")

    with pytest.raises(ValueError):
        ics.fit(X)


def test_method_select_as_string():
    """
    Test the type of method_select argument.

    The test verifies that a TypeError is raised if method_select an unknown character string.

    """
    X = np.random.randn(10, 3)
    ics = ICS(method_select="coconut")

    with pytest.raises(ValueError):
        ics.fit(X)


def test_invalid_scatters_for_QR():
    """
    Test for invalid scatters for QR warning.

    This test verifies that ICS initialization raises a warning when the algorithm is "QR" and the scatter matrices are
    invalid for "QR", then is checks that the code continues with algorithm = "standard".
    """
    X = np.random.randn(100, 5)
    ics = ICS(S1=covW, S2=cov, algorithm="QR")
    with pytest.warns(UserWarning, match="QR algorithm is not applicable; proceeding with the standard algorithm"):
        ics.fit(X)
    assert ics.algorithm == 'standard'

def test_invalid_algorithm_error():
    """
    Test for invalid algorithm error.

    This test verifies that the ICS initialization raises a ValueError when an invalid algorithm name is provided.
    """
    X = np.random.randn(10, 3)
    ics = ICS(algorithm="coconut")

    with pytest.raises(ValueError, match="must be a str among"):
        ics.fit(X)


def test_invalid_fix_signs_error():
    """
    Test for invalid fix_signs error.

    This test verifies that the ICS initialization raises a ValueError when an invalid fix_signs value is provided.
    """
    X = np.random.randn(10, 3)
    ics = ICS(fix_signs="coconut")

    with pytest.raises(ValueError, match="must be a str among"):
        ics.fit(X)


# Section: Fit Method Tests
def test_fit_method():
    """
    Test the fit method of the ICS class.

    This test verifies that the fit method processes the input data correctly,
    and sets the transformation matrix W, and kurtosis attributes.
    """
    ics = ICS()
    X = np.random.randn(100, 5)
    ics.fit(X)
    assert isinstance(ics, ICS)
    assert ics.components_ is not None
    assert ics.n_components_ is not None
    assert ics.component_names_ is not None
    assert ics.kurtosis_ is not None

    # Compare fit results with method_select = None
    ics2 = ICS()
    ics2.fit(X)
    assert isinstance(ics2, ICS)
    np.testing.assert_array_equal(ics.components_, ics2.components_)
    assert ics.n_components_ == ics2.n_components_
    assert ics.component_names_ == ics2.component_names_
    np.testing.assert_array_equal(ics.kurtosis_, ics2.kurtosis_)


def test_fit_method_normal_crit_str():
    """
    Test the fit method of the ICS class with method_select = "normal".

    This test verifies that the fit method processes the input data correctly,
    and sets the transformation matrix W, and kurtosis attributes.
    """
    ics = ICS(method_select="normal")
    X = np.random.randn(100, 5)
    ics.fit(X)
    assert isinstance(ics, ICS)
    assert ics.components_ is not None
    assert ics.n_components_ is not None
    assert ics.component_names_ is not None
    assert ics.kurtosis_ is not None
    assert ics.n_components_ <= X.shape[1]
    assert ics.n_components_ == ics.components_.shape[0]
    assert ics.criteria_out_ is not None


def test_fit_method_normal_crit():
    """
    Test the fit method of the ICS class with method_select = normal_crit.

    This test verifies that the fit method processes the input data correctly,
    and sets the transformation matrix W, and kurtosis attributes.
    """
    ics = ICS(method_select=normal_crit)
    X = np.random.randn(100, 5)
    ics.fit(X)
    assert isinstance(ics, ICS)
    assert ics.components_ is not None
    assert ics.n_components_ is not None
    assert ics.component_names_ is not None
    assert ics.kurtosis_ is not None
    assert ics.n_components_ <= X.shape[1]
    assert ics.n_components_ == ics.components_.shape[0]
    assert ics.criteria_out_ is not None


def test_fit_method_unimodal_crit_str():
    """
    Test the fit method of the ICS class with method_select = "unimodal".

    This test verifies that the fit method processes the input data correctly,
    and sets the transformation matrix W, and kurtosis attributes.
    """
    ics = ICS(method_select="unimodal")
    X = np.random.randn(100, 5)
    ics.fit(X)
    assert isinstance(ics, ICS)
    assert ics.components_ is not None
    assert ics.n_components_ is not None
    assert ics.component_names_ is not None
    assert ics.kurtosis_ is not None
    assert ics.n_components_ <= X.shape[1]
    assert ics.n_components_ == ics.components_.shape[0]
    assert ics.criteria_out_ is not None


def test_fit_method_unimodal_crit():
    """
    Test the fit method of the ICS class with method_select = unimodal_crit.

    This test verifies that the fit method processes the input data correctly,
    and sets the transformation matrix W, and kurtosis attributes.
    """
    ics = ICS(method_select=unimodal_crit)
    X = np.random.randn(100, 5)
    ics.fit(X)
    assert isinstance(ics, ICS)
    assert ics.components_ is not None
    assert ics.n_components_ is not None
    assert ics.component_names_ is not None
    assert ics.kurtosis_ is not None
    assert ics.n_components_ <= X.shape[1]
    assert ics.n_components_ == ics.components_.shape[0]
    assert ics.criteria_out_ is not None


def test_fit_method_median_crit_str():
    """
    Test the fit method of the ICS class with method_select = "median".

    This test verifies that the fit method processes the input data correctly,
    and sets the transformation matrix W, and kurtosis attributes.
    """
    ics = ICS(method_select="median")
    X = np.random.randn(100, 5)
    ics.fit(X)
    assert isinstance(ics, ICS)
    assert ics.components_ is not None
    assert ics.n_components_ is not None
    assert ics.component_names_ is not None
    assert ics.kurtosis_ is not None
    assert ics.n_components_ < X.shape[1]
    assert ics.n_components_ == ics.components_.shape[0]
    assert ics.criteria_out_ is not None


def test_fit_method_median_crit():
    """
    Test the fit method of the ICS class with method_select = median_crit.

    This test verifies that the fit method processes the input data correctly,
    and sets the transformation matrix W, and kurtosis attributes.
    """
    ics = ICS(method_select=median_crit)
    X = np.random.randn(100, 5)
    ics.fit(X)
    assert isinstance(ics, ICS)
    assert ics.components_ is not None
    assert ics.n_components_ is not None
    assert ics.component_names_ is not None
    assert ics.kurtosis_ is not None
    assert ics.n_components_ < X.shape[1]
    assert ics.n_components_ == ics.components_.shape[0]
    assert ics.criteria_out_ is not None


# Section: Transform Method Tests
def test_transform_method():
    """
    Test the transform method of the ICS class.

    This test verifies that the transform method correctly transforms the input data
    using the fitted ICS model, and raises an error if the model is not fitted.
    """
    ics = ICS()
    X = np.random.randn(100, 5)
    ics.fit(X)
    transformed_data = ics.transform(X)
    assert isinstance(ics, ICS)
    assert transformed_data.shape == X.shape
    with pytest.raises(NotFittedError):
        ics_unfitted = ICS()
        ics_unfitted.transform(X)


# Section: Fit-Transform Method Tests
def test_fit_transform_method():
    """
    Test the fit_transform method of the ICS class.

    This test verifies that the fit_transform method correctly fits the ICS model to the
    input data and transforms the data in a single step.
    """
    ics = ICS()
    X = np.random.randn(100, 5)
    transformed_data = ics.fit_transform(X)
    assert isinstance(ics, ICS)
    assert transformed_data.shape == X.shape


def test_empty_dataset():
    """
    Test the fit method with an empty dataset.

    This test verifies that the fit method raises a ValueError when the input data is empty.
    """
    ics = ICS()
    X = np.array([]).reshape(0, 5)
    with pytest.raises(ValueError):
        ics.fit(X)


def test_large_dataset():
    """
    Test the fit_transform method with a large dataset.

    This test verifies that the fit_transform method correctly processes a large dataset.
    """
    ics = ICS()
    X = np.random.randn(10000, 10)
    X_new = ics.fit_transform(X)
    assert ics.components_ is not None
    assert X_new is not None
    assert ics.kurtosis_ is not None


# Section: Error Handling Tests
@pytest.mark.parametrize("algorithm", algorithm)
@pytest.mark.parametrize("center", center)
@pytest.mark.parametrize("fix_signs", fix_signs)
def test_single_variable_error(run_py_ics, algorithm, center, fix_signs):
    """
    Test for single variable dataset error.

    This test verifies that the fit_transform method raises a ValueError when the input data has only one feature.
    """
    X_single_var = np.random.randn(100, 1)  # 100 samples, 1 feature
    params = {}
    with pytest.raises(ValueError, match="minimum of 2"):
        run_py_ics(X=X_single_var, algorithm=algorithm, center=center, fix_signs=fix_signs, **params)


@pytest.mark.parametrize("algorithm", algorithm)
@pytest.mark.parametrize("center", center)
@pytest.mark.parametrize("fix_signs", fix_signs)
def test_missing_values_error(run_py_ics, algorithm, center, fix_signs):
    """
    Test for dataset with missing values error.

    This test verifies that the fit_transform method raises a ValueError when the input data contains missing values.
    """
    X_missing_values = np.random.randn(100, 5)
    X_missing_values[0, 0] = np.nan
    params = {}
    with pytest.raises(ValueError, match="NaN"):
        run_py_ics(X=X_missing_values, algorithm=algorithm, center=center, fix_signs=fix_signs, **params)


@pytest.mark.parametrize("algorithm", algorithm)
@pytest.mark.parametrize("fix_signs", fix_signs)
def test_cannont_center_S1_Location_is_false(run_py_ics, algorithm, fix_signs):
    """
    Test for not being able to center warning when location is S1 is set to False

    This test verifies  if the correct warning message is raised when location is S1 is set to False.
    """
    X = np.random.randn(100, 5)
    with pytest.warns(UserWarning, match="Location component in S1 is required for centering the data. Proceeding without centering"):
        run_py_ics(X=X, S1=cov, algorithm=algorithm, center=True, fix_signs=fix_signs, S1_args={'location':False})

    with pytest.warns(UserWarning, match="Location component in S1 is required for centering the data. Proceeding without centering"):
        run_py_ics(X=X, S1=cov4, algorithm=algorithm, center=True, fix_signs=fix_signs, S1_args={'location':False})

    with pytest.warns(UserWarning, match="Location component in S1 is required for centering the data. Proceeding without centering"):
        run_py_ics(X=X, S1=covW, algorithm=algorithm, center=True, fix_signs=fix_signs, S1_args={'location':False})

    with pytest.warns(UserWarning, match="Location component in S1 is required for centering the data. Proceeding without centering"):
        run_py_ics(X=X, S1=covAxis, algorithm=algorithm, center=True, fix_signs=fix_signs, S1_args={'location':False})
