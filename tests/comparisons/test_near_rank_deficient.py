"""
Comparison tests for using the QR algorithm with near rank-deficient datasets to ensure
numerical stability between the QR implementations in R and Pyhon
"""
from . import *

logger = logging.getLogger(__name__)
@pytest.mark.parametrize("r_params, py_params", params_sets2)
@pytest.mark.parametrize("center", center)
@pytest.mark.parametrize("fix_signs", fix_signs)
def test_QR_near_rank_deficient(load_data, run_r_ics, run_py_ics, r_params, py_params, center, fix_signs):
    """
    Test for comparing ICS implementation in Python and R for near rank-deficient data.

    This test verifies that the Python implementation of ICS produces results similar to the R implementation
    for the same dataset and parameters.

    Parameters:
        r_params (dict): Parameters for the R implementation.
        py_params (dict): Parameters for the Python implementation.
        decimal (int): The decimal precision for comparison.
        center (bool): Whether to center the data.
        fix_signs (str): The method for fixing the signs of the components.
    """
    # Load the dataset
    data = load_iris()
    X = data.data
    scales = np.array([10**(-14), 10**(-3), 1, 10**14])
    X_rank_deficient = X * scales

    # Run ICS in R
    r_results = run_r_ics(X=X_rank_deficient,
                          algorithm='QR',
                          center=center,
                          fix_signs=fix_signs,
                          **r_params)

    # Run ICS in Python
    py_results = run_py_ics(X=X_rank_deficient,
                            algorithm='QR',
                            center=center,
                            fix_signs=fix_signs,
                            **py_params)

    similarity_percentage = calculate_similarity_percentage(r_results['transformation_matrix'],
                                                            py_results['transformation_matrix'])


    # Validate results
    assert np.all(similarity_percentage >= 99.9999999999), f"Similarity percentage too low: {similarity_percentage}"


    # The whiten algorithm breaks down in this case
    with pytest.raises(ValueError):
        # Run ICS in Python with the whiten algorithm
        py_results = run_py_ics(X=X_rank_deficient,
                                algorithm='whiten',
                                center=center,
                                fix_signs=fix_signs,
                                **py_params)