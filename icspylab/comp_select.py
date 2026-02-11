import numpy as np
from scipy.stats import shapiro, normaltest, jarque_bera, kurtosistest, skewtest
from sklearn.utils.validation import check_array
import numbers



class ComponentSelect:
    """
    A class to represent a component selection method and its related data.

    Attributes:
        label (str): Label of the component selection method.
        components (ndarray): Invariant components selected by the method.
        n_components (int): Number of invariant components selected by the method.
        component_names (ndarray): Names of invariant components selected by the method.
        info (dict): Additional information specific to the method.
    """

    def __init__(self, label, components, n_components, component_names, info):
        """
        Initialize the ComponentSelect object with specified parameters.

        Parameters:
            label (str): Label of the component selection method.
            components (ndarray): Invariant components selected by the method.
            n_components (int): Number of invariant components selected by the method.
            component_names (list): Names of invariant components selected by the method.
            info (dict): Additional information specific to the method.
        """
        self.label = label
        self.components = components
        self.n_components = n_components
        self.component_names = component_names
        self.info = info


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


def med_crit(kurtosis, W, nb_select=None, **kwargs):
    """
    Identifies as interesting the invariant coordinates whose generalized eigenvalues (kurtosis) are the furthermost
    away from the median of all generalized eigenvalues (kurtosis).

    Parameters:
        kurtosis (ndarray): Array of kurtosis values.
        W (ndarray): Transformation matrix in which each row contains the coefficients of the linear transformation to the corresponding invariant coordinate.
        nb_select (int or None, default=None): Exact number of components to select. If None (default), number of components to select is the number of variables minus one.

    Returns:
        dict: Summary of the component selection step
    """

    # gen_kurtosis validation
    gen_kurtosis = check_array(
        kurtosis,
        ensure_2d=False,
        dtype=float,
        ensure_all_finite=True,
    )

    if gen_kurtosis.ndim != 1:
        raise ValueError("gen_kurtosis must be 1D.")

    # nb_select validation
    p = len(gen_kurtosis)
    nb_select = _validate_nb_select(nb_select, p)

    all_comp_names = [f"IC_{i + 1}" for i in range(p)]

    # Components associated with the furthest eigenvalues from the median
    med_gen_kurtosis = np.median(gen_kurtosis)
    gen_kurtosis_diff = np.abs(gen_kurtosis - med_gen_kurtosis)

    idx_sel = np.argsort(gen_kurtosis_diff)[::-1][: nb_select]
    selected_component_names = [all_comp_names[i] for i in idx_sel]

    # Keep only the selected components
    name_to_idx = {name: i for i, name in enumerate(all_comp_names)}
    idx = [name_to_idx[name] for name in selected_component_names]
    components = W[idx, :]

    # ComponentSelect class
    n_components = len(selected_component_names)

    info = {
        "crit": "med",
        "nb_select": nb_select,
        "gen_kurtosis": gen_kurtosis,
        "med_gen_kurtosis": med_gen_kurtosis,
        "gen_kurtosis_diff_med": gen_kurtosis_diff,
        "component_names": selected_component_names
    }

    return ComponentSelect(label="med", components=components, n_components=n_components, component_names=selected_component_names, info=info)


def _validate_nb_select(nb_select, p):
    if nb_select is None:
        nb_select = p - 1
    else:
        if not isinstance(nb_select, (int, np.integer)):
            raise TypeError("nb_select must be an integer or None.")

        if nb_select <= 0:
            raise ValueError("nb_select must be strictly positive.")

        if nb_select >= p:
            raise ValueError("nb_select must be smaller than the number of kurtosis values.")
    return nb_select
