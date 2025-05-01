Testing
=======

Comparison with R
-----------------

This section explains the logic behind testing the ICSpyLab package to ensure it matches the functionality
of the original R package.

The testing approach involves the following steps:

1. **Setup**: Install necessary dependencies, including `pytest` and `rpy2` for interfacing with R.
2. **Data Loading**: Load datasets such as iris, wine, and diabetes using `scikit-learn`.
3. **Running ICS**: Perform the ICS (Invariant Coordinate Selection) algorithm in both R (using `rpy2`) and Python.
4. **Comparison**: Compare the results from the R implementation and the Python implementation.

**Fixtures and Parameters**

To streamline the testing process, fixtures are used to load data and run the ICS algorithm in both R and Python.
Parameters for the ICS algorithm, such as covariance estimators and transformation settings, are defined and tested
across different datasets and configurations.

**Main Testing Files**

The testing logic is organized into the following main files:

.. automodule:: tests.fixtures
   :members:
   :undoc-members:
   :show-inheritance:

**Validation**

The results with algorithms 'standard' and 'whiten' from the Python implementation are validated against the R implementation by comparing:

- Transformation matrices
- Kurtosis values
- Skewness values (if available)
- Transformed data

This comparison ensures that the Python package produces results consistent with the R package.

Unit tests
----------

Other tests include:

- Initialization tests
- Error Handling Tests
- Test with a large dataset: 10000 rows and 10 columns
- Consistency of 'standard' and 'whiten' algorithm
- Specific tests for QR algorithm (pending)

For more details and to view the full testing code, please refer to the `tests` directory in the source repository.
