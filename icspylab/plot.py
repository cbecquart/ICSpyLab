import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_array


def plot_ics(scores, components="first_last", y=None, col_names=None, **kwargs):
    """
    Plots a scatterplot matrix of the component scores of an invariant
    coordinate system obtained via an ICS transformation.

    It plots the full scatterplot matrix of the components only if there
    are less than seven components. Otherwise, the three first and three
    last components will be plotted by default. This is because the components with
    extreme kurtosis are the most interesting ones.

    Parameters:
        scores (ndarray): Results from an ICS transformation.
        components ({"first_last", "first"}, default: "first_last"): If p>6, plot either the 3 first and 3 last components (default) or the first 6 components ("first").
        y (array-like, optional): Labels used to color the points.
        col_names (list, optional): Names of columns to plot.
        **kwargs: Additional keyword arguments passed to sns.pairplot.

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

    valid_components = {"first_last", "first"}

    if components not in valid_components:
        raise ValueError(
            f"Invalid value for `components`: {components}. "
            f"Expected one of {valid_components}."
        )

    if col_names is None:
        col_names_ = [f"IC_{i+1}" for i in range(X.shape[1])]
    else:
        if not isinstance(col_names, list):
            raise TypeError(
                f"'col_names' must be a list, got {type(col_names).__name__} instead."
            )
        if len(col_names) != X.shape[1]:
            raise ValueError(
                f"'col_names' must contain exactly {X.shape[1]} elements."
            )
        col_names_ = col_names.copy()

    scores_df = pd.DataFrame(X, columns=col_names_)

    # Add labels for coloring
    hue = None
    palette = None
    if y is not None:
        y = np.asarray(y)

        if len(y) != X.shape[0]:
            raise ValueError(
                f"'y' must contain exactly {X.shape[0]} elements."
            )

        scores_df["label"] = y
        hue = "label"
        palette = "deep"

    p = scores_df.shape[1] - (1 if hue else 0)

    # Determine which components to plot
    if p <= 6:
        sns.pairplot(
            scores_df,
            hue=hue,
            palette=palette,
            **kwargs
        )
    else:
        if components == "first_last":
            cols = (
                [scores_df.columns[i] for i in range(3)] +
                [scores_df.columns[i] for i in range(p - 3, p)]
            )
        else:
            cols = [scores_df.columns[i] for i in range(6)]

        if hue:
            cols.append("label")

        sns.pairplot(
            scores_df[cols],
            hue=hue,
            palette=palette,
            **kwargs
        )

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
