"""
Module for plotting transformed data using Plotly.

This module provides functions to plot the results of ICS.

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_scores(scores, **kwargs):
    """
    Plots a scatterplot matrix of the component scores of an invariant coordinate system obtained via an ICS
    transformation.
    It plots the full scatterplot matrix of the components only if there are less than seven components. Otherwise, the
    three first and three last components will be plotted. This is because the components with extreme kurtosis are the
    most interesting ones.

    Parameters:
        scores (np.ndarray): results from an ICS transformation.
    """

    if isinstance(scores, (np.ndarray, pd.DataFrame, list)):
        scores_df = pd.DataFrame(scores, columns=[f"IC_{i+1}" for i in range(scores.shape[1])])
    else:
        raise TypeError("`scores` must be array-like: numpy.ndarray, or pandas.DataFrame.")

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
