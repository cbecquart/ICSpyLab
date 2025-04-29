"""
Unit tests for the ICS class in the ICSpy package.
"""
import numpy as np

from . import *
logger = logging.getLogger(__name__)


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
    assert ics.algorithm == 'whiten'
    assert ics.center is False
    assert ics.fix_signs == 'scores'
    assert ics.S1_args == {}
    assert ics.S2_args == {}
    assert ics.W is None
    assert ics.scores is None
    assert ics.kurtosis is None
    assert ics.skewness is None
    assert ics.feature_names_in_ is None
    assert ics.S1_X_ is None


def test_S1_as_matrix():
    """
    Test the type of S1 argument.

    The test verifies that an error is raised if S1 a numpy array.

    """
    X = np.random.randn(100, 5)
    cov_matrix = np.cov(X.T)
    with pytest.raises(AssertionError, match="S1 must be a function returning a Scatter object."):
        ICS(S1=cov_matrix)


def test_S1_as_string():
    """
    Test the type of S1 argument.

    The test verifies that an error is raised if S1 a character string.

    """
    with pytest.raises(AssertionError, match="S1 must be a function returning a Scatter object."):
        ICS(S1="cov")


def test_S2_as_matrix():
    """
    Test the type of S2 argument.

    The test verifies that an error is raised if S2 a numpy array.

    """
    X = np.random.randn(100, 5)
    cov_matrix = np.cov(X.T)
    with pytest.raises(AssertionError, match="S2 must be a function returning a Scatter object."):
        ICS(S2=cov_matrix)


def test_S2_as_string():
    """
    Test the type of S2 argument.

    The test verifies that an error is raised if S2 a character string.

    """
    with pytest.raises(AssertionError, match="S2 must be a function returning a Scatter object."):
        ICS(S2="cov")


def test_invalid_scatters_for_QR():
    """
    Test for invalid scatters for QR warning.

    This test verifies that ICS initialization raises a warning when the algorithm is "QR" and the scatter matrices are
    invalid for "QR", then is checks that the code continues with algorithm = "standard".
    """
    with pytest.warns(UserWarning, match="QR algorithm is not applicable; proceeding with the standard algorithm"):
        ics = ICS(S1=covW, S2=cov, algorithm="QR")
    assert ics.algorithm == 'standard'

def test_invalid_algorithm_error():
    """
    Test for invalid algorithm error.

    This test verifies that the ICS initialization raises a ValueError when an invalid algorithm name is provided.
    """
    invalid_algorithm = ''.join(random.choices(string.ascii_letters + string.digits, k=np.random.randint(1, 10)))
    while invalid_algorithm in ['whiten', 'standard', 'QR']:  # ensure the random string is not a valid algorithm name
        invalid_algorithm = ''.join(random.choices(string.ascii_letters + string.digits, k=np.random.randint(1, 10)))
    with pytest.raises(AssertionError, match="algorithm must be one of \['whiten', 'standard', 'QR'\]"):
        ICS(algorithm=invalid_algorithm)


def test_invalid_fix_signs_error():
    """
    Test for invalid fix_signs error.

    This test verifies that the ICS initialization raises a ValueError when an invalid fix_signs value is provided.
    """
    invalid_fix_signs = ''.join(random.choices(string.ascii_letters + string.digits, k=np.random.randint(1, 5)))
    while invalid_fix_signs in ['scores', 'W']:  # ensure the random string is not a valid fix_signs name
        invalid_fix_signs = ''.join(random.choices(string.ascii_letters + string.digits, k=np.random.randint(1, 10)))
    with pytest.raises(AssertionError, match="fix_signs must be one of \['scores', 'W'\]"):
        ICS(fix_signs=invalid_fix_signs)


# Section: Fit Method Tests
def test_fit_method():
    """
    Test the fit method of the ICS class.

    This test verifies that the fit method processes the input data correctly,
    and sets the transformation matrix (W), and kurtosis attributes, but not the scores.
    """
    ics = ICS()
    X = np.random.randn(100, 5)
    ics.fit(X)
    assert isinstance(ics, ICS)
    assert ics.W is not None
    assert ics.kurtosis is not None
    assert ics.scores is None


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
    with pytest.raises(ValueError):
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


# Section: Describe Method Tests
# def test_describe_method(capsys):
#     """
#     Test the describe method of the ICS class.
#
#     This test verifies that the describe method correctly prints a summary of the ICS model,
#     including the algorithm, centering option, sign fixing method, generalized kurtosis,
#     transformation matrix, transformed data, and skewness.
#     """
#     ics = ICS()
#     X = np.random.randn(100, 5)
#     ics.fit_transform(X)
#     ics.describe()
#     captured = capsys.readouterr()
#     assert "ICS Summary" in captured.out
#     assert "Algorithm" in captured.out
#     assert "Generalized Kurtosis" in captured.out
#     assert "Transformation Matrix (W)" in captured.out
#     assert "Transformed Data (Scores)" in captured.out


# Section: Edge Case Tests
# def test_empty_dataset():
#     """
#     Test the fit method with an empty dataset.
#
#     This test verifies that the fit method raises a ValueError when the input data is empty.
#     """
#     ics = ICS()
#     X = np.array([]).reshape(0, 5)
#     with pytest.raises(ValueError):
#         ics.fit(X)


def test_large_dataset():
    """
    Test the fit_transform method with a large dataset.

    This test verifies that the fit_transform method correctly processes a large dataset.
    """
    ics = ICS()
    X = np.random.randn(10000, 10)
    ics.fit_transform(X)
    assert ics.W is not None
    assert ics.scores is not None
    assert ics.kurtosis is not None


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
    with pytest.raises(ValueError, match="X must be at least bi-variate"):
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
    with pytest.raises(ValueError, match="Missing values are not allowed in X"):
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
