Testing
=======

Comparison with the R implementation
------------------------------------

To ensure functional equivalence with the original R implementation of ICS, ICSpyLab is validated against the reference
package through automated tests.

The testing workflow consists of the following steps:

1. **Setup**: Install necessary dependencies, including `pytest` and `rpy2` for interfacing with R.
2. **Data Loading**: Load datasets such as iris, wine, and diabetes using `scikit-learn`.
3. **ICS execution**: Perform the ICS (Invariant Coordinate Selection) algorithm in both R (using `rpy2`) and Python.
4. **Validation**: Outputs from the Python implementation are compared against those obtained from the R package to assess numerical consistency.

**Fixtures and Parameters**

To streamline the testing process, fixtures are used to load data and run the ICS algorithm in both R and Python.
Parameters for the ICS algorithm, such as covariance estimators and transformation settings, are defined and tested
across different datasets and configurations.

**Validation**

The results with algorithms 'standard' and 'whiten' from the Python implementation are validated against the R implementation by comparing:

- Transformation matrices
- Kurtosis values
- Skewness values (if available)
- Transformed data

Comparisons are performed using numerical tolerance thresholds to account for floating-point implementation differences
between R and Python. The method is `numpy.testing.assert_almost_equal` with 8 decimals.
This ensures that icspylab reproduces the behavior of the reference implementation with a high level of consistency.


Unit tests
----------

Additional unit tests cover:

- validation of the ICS estimator against scikit-learn estimator checks (sklearn.utils.estimator_checks.check_estimator),
- object initialization and parameter validation,
- error handling and invalid input scenarios,
- consistency between 'standard' and 'whiten' algorithms,
- affine invariance properties of the transformed coordinates.

The complete testing suite is available in the `tests` directory of the source repository.
