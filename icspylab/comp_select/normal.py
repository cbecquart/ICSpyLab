import numpy as np
from scipy.stats import shapiro, normaltest, jarque_bera, kurtosistest, skewtest
import numbers
from .base import ComponentSelect, _validate_nb_select


def _normaltest(x):
    """D’Agostino and Pearson’s test."""
    stat, p = normaltest(x)
    return {"statistic": stat, "p_value": p}

def _agostino_test(x):
    """D'Agostino test of skewness in normally distributed data."""
    stat, p = skewtest(x)
    return {"statistic": stat, "p_value": p}

def _anscombe_test(x):
    """Anscombe-Glynn test of kurtosis in normally distributed data."""
    stat, p = kurtosistest(x)
    return {"statistic": stat, "p_value": p}

def _jarque_test(x):
    """Jarque-Bera normality test."""
    stat, p = jarque_bera(x)
    return {"statistic": stat, "p_value": p}

def _shapiro_test(x):
    """Shapiro-Wilk normality test."""
    stat, p = shapiro(x)
    return {"statistic": stat, "p_value": p}

def normal_crit(X, W, level=0.05, test="agostino", max_select=None, **kwargs):
    """
    Identifies invariant coordinates that deviate from normality using univariate normality tests. Only the first and
    last components are investigated.

    SciPy implementations are used. The available tests are:
    `normal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html>`_,
    `agostino <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html>`_,
    `jarque <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html>`_,
    `anscombe <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosistest.html>`_, and
    `shapiro <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html>`_.

    Parameters:
        X (ndarray): Data to fit the ICS model, where rows are samples and columns are features.
        W (ndarray): Transformation matrix in which each row contains the coefficients of the linear transformation to the corresponding invariant coordinate.
        level (float, default=0.05) The initial level used to make a decision based on the test p-values.
        test ({'normal', 'agostino', 'jarque', 'anscombe', 'shapiro'}, default='agostino') Name of the normality test to be used. Refer to the SciPy documentation for more information about the tests.
        max_select (int or None, default=None): Maximum number of components to select.

    Returns:
        dict: Summary of the component selection step

    References:
        - Archimbaud, A., Alfons, A., Nordhausen, K., & Ruiz-Gazen, A. (2023). ICSClust: Tandem clustering with invariant coordinate selection.
        - Alfons, A., Archimbaud, A., Nordhausen, K., & Ruiz-Gazen, A. (2024). Tandem clustering with invariant coordinate selection. Econometrics and Statistics. doi:10.1016/j.ecosta.2024.03.002.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from icspylab import ICS, normal_crit
        >>> iris = load_iris()
        >>> X = iris.data
        >>> ics = ICS(S1="cov", S2="cov4")
        >>> selection_res = normal_crit(X=X, W=ics.components_)
        >>> print(selection_res.info)
        {'crit': 'normal', 'level': 0.05, 'max_select': 3, 'test': 'agostino', 'pvalues': array([0.07492811, 0.19460223, 0.9311222 , 0.00942277]), 'adjusted_levels': [0.05, 0.025], 'component_names': ['IC_4']}

    """

    # test validation
    tests = {
        "normal": _normaltest,
        "agostino": _agostino_test,
        "jarque": _jarque_test,
        "anscombe": _anscombe_test,
        "shapiro": _shapiro_test
    }

    if test not in tests:
        valid = ", ".join(tests.keys())
        raise ValueError(f"test must be one of: {valid}")

    test_fun = tests[test]

    # level validation
    if not isinstance(level, numbers.Real):
        raise TypeError("level must be a real number.")

    if not (0.0 < float(level) < 1.0):
        raise ValueError("level must be between 0 and 1.")

    # max_select validation
    all_comp_names = [f"IC_{i+1}" for i in range(X.shape[1])]
    p = X.shape[1]
    max_select = _validate_nb_select(max_select, p)

    # Transform X to apply the normality tests
    Z = X @ W.T

    # Apply marginal normality tests to all components and select on p-values
    test_pvals = np.array([test_fun(Z[:, j])["p_value"] for j in range(Z.shape[1])])
    comp_signif = test_pvals <= level

    # If none of them are significative
    if comp_signif.sum() == 0:
        selected_component_names = []
        adjusted_levels = [level]

    # Else: we select the components on the extreme left and right while they are not gaussian
    else:
        temp = 1
        selected_component_names = []
        adjusted_levels = [level]

        pvals = test_pvals.copy()
        comps = all_comp_names.copy()

        while temp <= max_select:

            left_pval = pvals[0]
            right_pval = pvals[-1]

            if (left_pval < level) and (left_pval < right_pval):
                selected_component_names.append(comps[0])
                comps = comps[1:]
                pvals = pvals[1:]

            elif right_pval < level:
                selected_component_names.append(comps[-1])
                comps = comps[:-1]
                pvals = pvals[:-1]

            else:
                break

            temp += 1

            # Adjust the alpha level by dividing by the old weight and multiplying by the new one.
            pvals = pvals * temp / (temp - 1)
            adjusted_levels.append(level / temp)

    # Keep only the selected components
    name_to_idx = {name: i for i, name in enumerate(all_comp_names)}
    idx = [name_to_idx[name] for name in selected_component_names]
    components = W[idx, :]

    # ComponentSelect class
    n_components = len(selected_component_names)

    info = {
        "crit": "normal",
        "level": level,
        "max_select": max_select,
        "test": test,
        "pvalues": test_pvals.copy(),
        "adjusted_levels": adjusted_levels,
        "component_names": selected_component_names
    }

    return ComponentSelect(label="normal", components=components, n_components=n_components, component_names=selected_component_names, info=info)
