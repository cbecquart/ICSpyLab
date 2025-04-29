"""
Comparison tests for Python and R implementations of ICS.
"""
from . import *

logger = logging.getLogger(__name__)
@pytest.mark.parametrize("dataset_name", datasets)
@pytest.mark.parametrize("decimal", decimal_precisions_for_r)
@pytest.mark.parametrize("r_params, py_params", params_sets)
@pytest.mark.parametrize("algorithm", algorithm)
@pytest.mark.parametrize("center", center)
@pytest.mark.parametrize("fix_signs", fix_signs)
def test_ICS(load_data, run_r_ics, run_py_ics, dataset_name, r_params, py_params, decimal, algorithm, center, fix_signs):
    """
    Test for comparing ICS implementation in Python and R.

    This test verifies that the Python implementation of ICS produces results similar to the R implementation
    for the same dataset and parameters.

    Parameters:
        dataset_name (str): The name of the dataset.
        r_params (dict): Parameters for the R implementation.
        py_params (dict): Parameters for the Python implementation.
        decimal (int): The decimal precision for comparison.
        algorithm (str): The algorithm used for ICS.
        center (bool): Whether to center the data.
        fix_signs (str): The method for fixing the signs of the components.
    """
    # logger.info(f"Testing ICS for dataset {dataset_name} with algorithm {algorithm}, center {center}, fix_signs {fix_signs}")

    # Load the dataset
    X, y = load_data(dataset_name)

    # Run ICS in R
    r_results = run_r_ics(X=X,
                          algorithm=algorithm,
                          center=center,
                          fix_signs=fix_signs,
                          **r_params)

    # Run ICS in Python
    py_results = run_py_ics(X=X,
                            algorithm=algorithm,
                            center=center,
                            fix_signs=fix_signs,
                            **py_params)

    # Validate results
    np.testing.assert_almost_equal(py_results['transformation_matrix'],
                                   r_results['transformation_matrix'],
                                   decimal=decimal,
                                   err_msg="Transformation Matrix does not match")

    np.testing.assert_almost_equal(py_results['kurtosis'],
                                   r_results['kurtosis'],
                                   decimal=decimal,
                                   err_msg="Kurtosis does not match")

    if py_results['skewness'] is None and r_results['skewness'] is None:
        pass  # Both are None, no need to compare
    elif py_results['skewness'] is None or r_results['skewness'] is None:
        raise AssertionError("One of the skewness results is None while the other is not.")
    else:
        np.testing.assert_almost_equal(py_results['skewness'],
                                       r_results['skewness'],
                                       decimal=decimal,
                                       err_msg="Skewness does not match")

    np.testing.assert_almost_equal(py_results['transformed_data'],
                                   r_results['transformed_data'],
                                   decimal=decimal,
                                   err_msg="Transformed Data does not match")
