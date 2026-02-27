"""
Module for plotting transformed data using Plotly.

This module provides functions to plot the results of ICS.

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_array


def plot_ics(scores, **kwargs):
    """
    Plots a scatterplot matrix of the component scores of an invariant coordinate system obtained via an ICS
    transformation.
    It plots the full scatterplot matrix of the components only if there are less than seven components. Otherwise, the
    three first and three last components will be plotted. This is because the components with extreme kurtosis are the
    most interesting ones.

    Parameters:
        scores (ndarray): results from an ICS transformation.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from icspylab import ICS
        >>> iris = load_iris()
        >>> X = iris.data
        >>> ics = ICS()
        >>> X_new = ics.fit_transform(X)
        >>> plot_ics(X_new)
    """

    X = check_array(
        scores,
        force_writeable=True,
        ensure_all_finite=True,
        accept_sparse=("csr", "csc"),
        ensure_2d=True,
        ensure_min_features=2,
        copy=False,
    )

    scores_df = pd.DataFrame(X, columns=[f"IC_{i+1}" for i in range(X.shape[1])])

    p = scores_df.shape[1]

    # Determine which components to plot (3 fist and 3 last components)
    if p <= 6:
        sns.pairplot(scores_df, **kwargs)
    else:
        cols = list(range(3)) + list(range(p-3, p))
        sns.pairplot(scores_df.iloc[:, cols], **kwargs)

    plt.show()


def _plot_kurtosis(kurtosis, **kwargs):
    """Plot the generated kurtosis."""

    kurtosis = np.asarray(kurtosis)
    x = [f"IC_{i+1}" for i in np.arange(len(kurtosis))]

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=x, y=kurtosis, s=100, color='dodgerblue', **kwargs)
    plt.grid(axis='x')
    plt.ylabel('Generalized kurtosis')
    plt.title('Scatter plot of generalized kurtosis')
    plt.show()
