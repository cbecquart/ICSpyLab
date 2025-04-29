"""
Module containing the main Invariant Coordinate Selection (ICS) Class and associated methods.

The ICS class provides methods to fit the ICS model from the data, transform data using the model,
and provide a detailed summary of the results. This module relies on scatter matrices
(defined in the Scatter page). The ICS class supports three different
algorithms for applying ICS to data: ('standard', 'whiten', and 'QR'), which can be specified
as parameters during instantiation. Additional options such as the choice of scatter matrices,
centering the data, and fixing the signs can also be defined.

This implementation is based on the function ICS-S3 from the R package `ICS <https://cran.r-project.org/web/packages/ICS/index.html>`_.
For more details about the supported algorithms and 'fix_signs' argument, see the R package
`documentation <https://cran.r-project.org/web/packages/ICS/ICS.pdf>`_ (function ICS-S3).
"""


import numpy as np
import pandas as pd
import warnings
from scipy.linalg import qr
from numpy.linalg import multi_dot

from .scatter import Scatter, cov, covW, covAxis, cov4
from .utils import sort_eigenvalues_eigenvectors, sqrt_symmetric_matrix, _sign_max, _check_gen_kurtosis
from .plot import plot_scores, _plot_kurtosis



class ICS:
    """
    Invariant Coordinate Selection (ICS) Class and associated methods.

    This class implements the ICS algorithm: it transforms the data, via the simultaneous diagonalization of two scatter
    matrices, into an invariant coordinate system or independent components, depending on the underlying assumptions.
    It supports various scatter matrix calculations and offers multiple algorithms for applying ICS.

    Parameters:
        S1 (function returning a scatter object): (default: cov) Function to compute the first scatter matrix.
        S2 (function returning a scatter object): (default: covW) Function to compute the second scatter matrix.
        algorithm (str): (default: 'whiten') The algorithm used for transformation ('standard', 'whiten', 'QR').
        center (bool): (default: False): a logical indicating whether the invariant coordinates should be centered with respect to the first locattion or not. Centering is only applicable if the first scatter object contains a location component, otherwise this is set to False. Note that this only affects the scores of the invariant components (output scores), but not the generalized kurtosis values (output kurtosis).
        fix_signs(str): (default: 'scores') How to fix the signs of the invariant coordinates. Possible values are "scores" to fix the signs based on (generalized) skewness values of the coordinates, or "w" to fix the signs based on the coefficient matrix of the linear transformation.
        S1_args (dict): Additional arguments for S1.
        S2_args (dict): Additional arguments for S2.

    Attributes:
        W (np.ndarray): Transformation matrix in which each row contains the coefficients of the linear transformation to the corresponding invariant coordinate.
        scores (np.ndarray): Transformed matrix in which each column contains the scores of the corresponding invariant coordinate.
        kurtosis (np.ndarray): Generalized kurtosis values.
        skewness (np.ndarray): Skewness values.
        feature_names_in_ (np.ndarray): Names of features seen during fit. Defined only when X has feature names that are all strings.
        S1_X_ (np.ndarray): Fitted scatter S1. Defined only when center=True.

    Supported Algorithms:
        1. standard: performs the spectral decomposition of the symmetric matrix :math:`S_1(X)^{-1/2}S_2(X)S_1(X)^{-1/2}`
        2. whiten: whitens the data with respect to the first scatter matrix before computing the second scatter matrix.
        3. QR: numerically stable algorithm based on the QR algorithm for a common family of scatter pairs: if S1 is cov(), and if S2 is one of cov4, covW, or covAxis. See Archimbaud et al. (2023) for details.

    Examples:
        >>> from sklearn.datasets import load_iris
        >>> from icspy import ICS
        >>> iris = load_iris()
        >>> X = iris.data
        >>> ICS = ICS()
        >>> ICS.fit(X)
    """

    def __init__(self, S1=cov, S2=covW, algorithm='whiten', center=False, fix_signs='scores', S1_args={}, S2_args={}):
        """
        Initialize the ICS object with specified parameters.

        Parameters:
            S1 (function returning a scatter object): (default: cov) Function to compute the first scatter matrix.
            S2 (function returning a scatter object): (default: covW) Function to compute the second scatter matrix.
            algorithm (str): (default: 'whiten') The algorithm used for transformation ('standard', 'whiten', 'QR').
            center (bool): (default: False): a logical indicating whether the invariant coordinates should be centered with
            respect to the first locattion or not. Centering is only applicable if the first scatter object contains a
            location component, otherwise this is set to False. Note that this only affects the scores of the invariant
            components (output scores), but not the generalized kurtosis values (output kurtosis).
            fix_signs(str): (default: 'scores') How to fix the signs of the invariant coordinates. Possible values are
            "scores" to fix the signs based on (generalized) skewness values of the coordinates, or "w" to fix the signs
            based on the coefficient matrix of the linear transformation.
            S1_args (dict): Additional arguments for S1.
            S2_args (dict): Additional arguments for S2.

        Examples:
            >>> ICS = ICS(S1 = cov,
            >>>           S2 = covW,
            >>>           algorithm = 'standard',
            >>>           center = False,
            >>>           fix_signs = 'scores',
            >>>           S1_args = {},
            >>>           S2_args = {})
            >>>
        """
        assert callable(S1), "S1 must be a function returning a Scatter object."
        assert callable(S2), "S2 must be a function returning a Scatter object."
        assert isinstance(center, bool), "center must be boolean."

        # Check for valid algorithm string
        valid_algo = ['whiten', 'standard', 'QR']
        assert algorithm in valid_algo, f"algorithm must be one of {valid_algo}"

        # Check for valid fix_signs string
        valid_fix_signs = ['scores', 'W']
        assert fix_signs in valid_fix_signs, f"fix_signs must be one of {valid_fix_signs}"

        # Check if the QR algorithm is applicable
        if algorithm == "QR":
            if not (S1 == cov and (S2 == covW or S2 == covAxis or S2 == cov4)):
                warnings.warn("QR algorithm is not applicable; proceeding with the standard algorithm")
                algorithm = "standard"

        self.S1 = S1
        self.S2 = S2
        self.algorithm = algorithm
        self.center = center
        self.fix_signs = fix_signs
        self.S1_args = S1_args
        self.S2_args = S2_args
        self.W = None
        self.scores = None
        self.kurtosis = None
        self.skewness = None
        self.feature_names_in_ = None
        self.S1_X_ = None

    def fit(self, X):
        """
        Fit the ICS model to the data.

        This function relies on several helper methods to perform the ICS transformation:
        _validate_input, _compute_first_scatter, _compute_second_scatter,
        _transform_second_scatter, _compute_transformation, _compute_transformation_qr,
        _center_data, _fix_component_signs.

        Parameters:
            X (array-like): Data to fit the ICS model, where rows are samples and columns are features.

        Returns:
            self:The fitted ICS object.
        """
        if isinstance(X, pd.DataFrame):
            column_names = X.columns
            X = X.values
        else:
            column_names = [f'Feature_{i}' for i in range(X.shape[1])]
        self.feature_names_in_ = column_names

        X = np.asarray(X)
        self._validate_input(X)
        S1_X, S1_X_inv_sqrt = self._compute_first_scatter(X)

        if self.algorithm == "whiten":
            # Whiten the data using the inverse square root of S1
            Y = np.dot(X, S1_X_inv_sqrt)
            S2_Y = self._compute_second_scatter(Y)
            W, gen_kurtosis = self._compute_transformation(S1_X_inv_sqrt, S2_Y)
        elif self.algorithm == "QR":
            # Use the QR decomposition for transformation
            W, gen_kurtosis = self._compute_transformation_qr(X,  S1_X)
        else:
            # Use the standard algorithm
            S2_X = self._compute_second_scatter(X)
            S2_Y = self._transform_second_scatter(S1_X_inv_sqrt, S2_X)
            W, gen_kurtosis = self._compute_transformation(S1_X_inv_sqrt, S2_Y)

        W_final, gen_skewness = self._fix_component_signs(X, W)

        if self.center:
            self.S1_X_ = S1_X

        self.W = W_final
        self.kurtosis = gen_kurtosis
        self.skewness = gen_skewness

        return self

    def transform(self, X):
        """
        Transform the data using the fitted ICS model.

        Parameters:
            X (array-like): Data to transform.

        Returns:
            np.ndarray: Transformed matrix in which each column contains the scores of the corresponding invariant
            coordinate.
        """
        if self.W is None:
            raise ValueError("The ICS model must be fitted before transforming data.")

        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X)

        assert self.W.shape[0] == X.shape[1], f"The fitted model expects {self.W.shape[0]} features in X."

        if self.center:
            # Center the data if required
            X = self._center_data(X, self.S1_X_)

        # Compute the final transformed data
        Z_final = X @ self.W.T
        self.scores = Z_final

        return Z_final

    def fit_transform(self, X):
        """
        Fit the ICS model and transform the data using the fitted ICS model.

        Parameters:
            X (array-like): Data to fit and transform.

        Returns:
            np.ndarray: Transformed matrix in which each column contains the scores of the corresponding invariant
            coordinate.
        """
        self.fit(X)
        return self.transform(X)

    def plot(self, **kwargs):
        """Plot the transformed data using the fitted ICS model."""
        if self.scores is None:
            raise ValueError("No transformed data available. Fit the model first.")
        plot_scores(self.scores, **kwargs)


    def plot_kurtosis(self, **kwargs):
        """Plot the generated kurtosis."""
        if self.kurtosis is None:
            raise ValueError("No generated kurtosis available. Fit the model first.")
        _plot_kurtosis(self.kurtosis, **kwargs)


    def describe(self):
        """
        Print a summary of the ICS model.

        This includes the algorithm used, whether data was centered, how signs were fixed;
        and displays the generalized kurtosis, transformation matrix, transformed data, and
        the skewness of the data.
        """
        feature_names = self.feature_names_in_

        print("\nICS based on two scatter matrices")
        print(f"S1: {self.S1.__name__}")
        print(f"S1_args: {self.S1_args}")
        print(f"S2: {self.S2.__name__}")
        print(f"S2_args: {self.S2_args}")

        print("\nInformation on the algorithm:")
        print(f"algorithm: {self.algorithm}")
        print(f"center: {self.center}")
        print(f"fix_signs: {self.fix_signs}")

        # Print the kurtosis values
        print("\nThe generalized kurtosis measures of the components are:")
        if self.kurtosis is not None:
            for idx, val in enumerate(self.kurtosis):
                print(f"IC_{idx + 1}: {val:.4f}")
        else:
            print("None")

        # Print the coefficient matrix
        print("\nThe coefficient matrix of the linear transformation is:")
        if self.W is not None:
            header = "     " + " ".join(f"{name:>12}" for name in feature_names)
            print(header)
            for idx, row in enumerate(self.W, start=1):
                row_str = " ".join(f"{val:>12.5f}" for val in row)
                print(f"IC_{idx:<3} {row_str}")
        else:
            print("None")

    def _validate_input(self, X):
        """
        Validate the input data matrix.

        This step ensures there are no missing values, the data is at least bi-variate,
        the algorithm name is valid, and the fix-signs string is correct. If the 'QR' algorithm
        is chosen, it also verifies the applicability of the specified scatter matrices.

        Parameters:
            X (np.ndarray): Data to validate.

        Raises:
            ValueError: If the input data doesn't meet any of the requirements.

        Algorithm:
            standard, whiten, QR
        """
        # Check for missing values
        p = X.shape[1]
        if pd.isnull(X).any():
            raise ValueError("Missing values are not allowed in X")
        if p <= 1:
            raise ValueError("X must be at least bi-variate")

    def _compute_first_scatter(self, X):
        """
        Apply the first scatter matrix to the data and compute the inverse square root.

        This step is the first step of computing ICS and is common across all 3 algorithms.

        Parameters:
            X (np.ndarray): The data matrix

        Returns:
            tuple: The first scatter matrix applied to the data, and its inverse square root

        Algorithms:
            standard, whiten, QR
        """
        S1_X = self.S1(X, **self.S1_args)
        if not isinstance(S1_X, Scatter):
            raise ValueError("S1 must return a Scatter object")

        S1_X_inv_sqrt = sqrt_symmetric_matrix(S1_X.scatter, inverse=True)
        return S1_X, S1_X_inv_sqrt

    def _compute_second_scatter(self, X):
        """
        Apply the second scatter matrix either directly on the data (standard algorithm) or on the whitened data
        (whiten algorithm).

        Parameters:
            X (np.ndarray): The data matrix or whitened data.

        Returns:
            Scatter: The second scatter matrix applied to the passed data.

        Algorithm:
            standard, whiten
        """
        S2_X = self.S2(X, **self.S2_args)
        if not isinstance(S2_X, Scatter):
            raise ValueError("S2 must return an Scatter object")
        return S2_X

    def _transform_second_scatter(self, S1_X_inv_sqrt, S2_X):
        """
        Transform the second scatter matrix using the inverse square root of the first scatter matrix

        Parameters:
            S1_X_inv_sqrt (np.ndarray): The inverse square root of the first scatter matrix
            S2_X (Scatter): The second scatter matrix

        Returns:
            Scatter: The transformed scatter matrix.

        Algorithm:
            standard
        """
        S2_Y = Scatter(
            location=None,
            scatter=multi_dot([S1_X_inv_sqrt, S2_X.scatter, S1_X_inv_sqrt]),
            label=S2_X.label
        )
        return S2_Y

    def _compute_transformation(self, S1_X_inv_sqrt, S2_Y):
        """
        Compute the transformation matrix and generalized kurtosis.

        This method performs the eigen decomposition of the transformed second scatter matrix (S2_Y),
        sorts the eigenvalues and eigenvectors, and computes the transformation matrix W by multiplying
        the transpose of the eigenvector matrix with the inverse square root of the first scatter matrix (S1_X_inv_sqrt).

        Parameters:
            S1_X_inv_sqrt (np.ndarray): The inverse square root of the first scatter matrix.
            S2_Y (Scatter): The second scatter matrix transformed (standard algorithm) or applied to the whitened data.

        Returns:
            tuple: The transformation matrix and generalized kurtosis.

        Algorithms:
            standard, whiten
        """
        S2_Y_eigenval, S2_Y_eigenvect = np.linalg.eig(S2_Y.scatter)
        S2_Y_eigenval, S2_Y_eigenvect = sort_eigenvalues_eigenvectors(S2_Y_eigenval, S2_Y_eigenvect)
        gen_kurtosis = S2_Y_eigenval
        W = np.dot(S2_Y_eigenvect.T, S1_X_inv_sqrt)
        return W, gen_kurtosis

    def _compute_transformation_qr(self, X, S1_X):
        """
        Compute the transformation matrix and generalized kurtosis using QR decomposition.

        This method follows these steps:
        1. Center the data matrix X.
        2. Reorder rows by decreasing infinity norm for numerical stability.
        3. Perform QR decomposition with column pivoting on the reordered matrix.
        4. Compute the leverage scores for further calculations.
        5. Apply the second scatter matrix function and compute S2_Y.
        6. Perform eigenvalue decomposition on S2_Y.
        7. Compute the transformation matrix W using the solution of a linear system involving R and eigenvectors.
        8. Reorder W according to the original pivoting order.

        Parameters:
            X (np.ndarray): The data matrix.
            S1_X (Scatter): The first scatter matrix.

        Algorithm:
            QR

        Returns:
            tuple: The transformation matrix and generalized kurtosis.
        """
        n, p = X.shape

        # Center the data
        T1_X = S1_X.location
        if T1_X is None:
            X_centered = X
        else:
            X_centered = X - T1_X

        # Reorder rows by decreasing infinity norm for numerical stability
        norm_inf = np.max(np.abs(X_centered), axis=1)
        order_rows = np.argsort(norm_inf)[::-1]
        X_reordered = X_centered[order_rows,]

        # Perform QR decomposition with column pivoting on the reordered and scaled matrix
        Q, R, P = qr(X_reordered / np.sqrt(n - 1), mode='economic', pivoting=True)

        # Compute leverage scores
        d = (n - 1) * np.sum(Q ** 2, axis=1)

        # Determine alpha and cf values based on the second scatter matrix function
        if self.S2 == cov4:
            alpha = 1
            cf = 1 / (p + 2)
        elif self.S2 == covAxis:
            alpha = -1
            cf = p
        elif self.S2 == covW:
            alpha = self.S2_args.get('alpha', 1)
            cf = self.S2_args.get('cf', 1)

        # Apply the weighting function to leverage scores and compute S2_Y
        d = d[:, np.newaxis]
        S2_Y = cf * (n - 1) / n * (Q * (d ** alpha)).T @ Q

        # Perform eigenvalue decomposition on S2_Y
        eigenvalues, eigenvectors = np.linalg.eig(S2_Y)
        _check_gen_kurtosis(eigenvalues)
        eigenvalues, eigenvectors = sort_eigenvalues_eigenvectors(eigenvalues, eigenvectors)

        # Compute the transformation matrix W and reorder it according to the original pivoting order
        W = np.linalg.solve(R, eigenvectors).T
        W = W[:, np.argsort(P)]

        return W, eigenvalues

    def _center_data(self, X, S1_X):
        """
        Center the data matrix using the location component of the first scatter matrix.

        Parameters:
            X (np.ndarray): The data matrix
            S1_X (Scatter): The first scatter matrix

        Returns:
            np.ndarray: The centered data matrix

        Algorithm:
            standard, whiten, QR
        """
        T1_X = S1_X.location
        if T1_X is None:
            warnings.warn("Location component in S1 is required for centering the data. Proceeding without centering")
            self.center = False
        else:
            X = X - T1_X
        return X

    def _fix_component_signs(self, X, W):
        """
        Fix the signs of the components based on the specified method.

        Parameters:
            X (np.ndarray): The data matrix.
            W (np.ndarray): The transformation matrix.

        Returns:
            tuple: The final transformation matrix and skewness values.

        Algorithm:
            standard, whiten, QR
        """
        if self.fix_signs == "scores":
            Z = np.dot(X, W.T)
            gen_skewness = np.mean(Z, axis=0) - np.median(Z, axis=0)
            skewness_signs = np.where(gen_skewness >= 0, 1, -1)
            gen_skewness = skewness_signs * gen_skewness
            W_final = W * skewness_signs[:, np.newaxis]
        else:
            # Calculate row signs based on the maximum absolute value
            row_signs = np.apply_along_axis(_sign_max, 1, W)
            row_norms = np.sqrt(np.sum(W ** 2, axis=1))
            W_final = (W.T / (row_signs * row_norms)).T
            gen_skewness = None

        return W_final, gen_skewness
