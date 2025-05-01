"""
Comparison tests for running ICSpyLab multiple times on the same dataset to make sure we get consistent results.
"""

from . import *

logger = logging.getLogger(__name__)
@pytest.mark.parametrize("dataset_name", datasets)
@pytest.mark.parametrize("algorithm", algorithm)
@pytest.mark.parametrize("center", center)
@pytest.mark.parametrize("fix_signs", fix_signs)
@pytest.mark.parametrize("r_params, py_params", params_sets)
def test_python_ICS_numerical_stability(load_data, run_py_ics, dataset_name, algorithm, center, fix_signs, py_params, r_params):
    """
    Test the consistency of the transformation_matrix output of ICS implementation.

    This test runs the ICS algorithm 10 times on the same dataset with the same parameters
    and checks that the transformation matrices produced in each run are identical.

    Parameters:
        load_data (function): Function to load the dataset.
        run_py_ics (function): Function to run the Python implementation of ICS.
        dataset_name (str): The name of the dataset.
        algorithm (str): The algorithm used for ICS.
        center (bool): Whether to center the data.
        fix_signs (str): The method for fixing the signs of the components.
        py_params (dict): Parameters for the Python implementation.
        r_params (dict): Parameters for the R implementation.
    """
    X, y = load_data(dataset_name)

    transformation_matrices = []
    for _ in range(10):
        py_results = run_py_ics(X=X,
                                algorithm=algorithm,
                                center=center,
                                fix_signs=fix_signs,
                                **py_params)
        transformation_matrices.append(py_results['transformation_matrix'])

    for i in range(1, 10):
        np.testing.assert_array_equal(transformation_matrices[0],
                                      transformation_matrices[i],
                                      err_msg=f"Transformation Matrix does not match in run {i}")


@pytest.mark.parametrize("dataset_name", datasets)
@pytest.mark.parametrize("center", center)
@pytest.mark.parametrize("fix_signs", fix_signs)
@pytest.mark.parametrize("r_params, py_params", params_sets)
@pytest.mark.parametrize("decimal", decimal_precisions)
def test_standard_whiten_consistency(load_data, run_py_ics, dataset_name, center, fix_signs, py_params, r_params, decimal):
    """
    Test the consistency of "standard" and "whiten" algorithms.

    This test verifies that the ICS implementation produces the same results with "standard" and "whiten" algorithms
    for the same dataset and parameters.

    Parameters:
        load_data (function): Function to load the dataset.
        run_py_ics (function): Function to run the Python implementation of ICS.
        dataset_name (str): The name of the dataset.
        center (bool): Whether to center the data.
        fix_signs (str): The method for fixing the signs of the components.
        py_params (dict): Parameters for the Python implementation.
        r_params (dict): Parameters for the R implementation.
        decimal (int): The decimal precision for comparison.
    """
    X, y = load_data(dataset_name)

    standard_results = run_py_ics(X=X,
                                  algorithm="standard",
                                  center=center,
                                  fix_signs=fix_signs,
                                  **py_params)
    whiten_results = run_py_ics(X=X,
                                algorithm="whiten",
                                center=center,
                                fix_signs=fix_signs,
                                **py_params)

    # Validate results
    np.testing.assert_allclose(standard_results['transformation_matrix'],
                               whiten_results['transformation_matrix'],
                               rtol=decimal,
                               err_msg="Transformation Matrix does not match")

    np.testing.assert_allclose(standard_results['kurtosis'],
                               whiten_results['kurtosis'],
                               rtol=decimal,
                               err_msg="Kurtosis does not match")

    if standard_results['skewness'] is None and whiten_results['skewness'] is None:
        pass  # Both are None, no need to compare
    elif standard_results['skewness'] is None or whiten_results['skewness'] is None:
        raise AssertionError("One of the skewness results is None while the other is not.")
    else:
        np.testing.assert_allclose(standard_results['skewness'],
                                   whiten_results['skewness'],
                                   rtol=decimal,
                                   err_msg="Skewness does not match")

    np.testing.assert_allclose(standard_results['transformed_data'],
                               whiten_results['transformed_data'],
                               rtol=decimal,
                               err_msg="Transformed Data does not match")
