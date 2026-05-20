import numpy as np
from scipy.optimize import brent
from scipy.stats import norm, multivariate_normal
import numbers
from .base import ComponentSelect, _validate_nb_select


def _ftu_exact_statistics(X):
    """Reference: Siffer, A., Fouque, P.-A., Termier, A. and Largouët, C. (2018), Are your data gathered?, In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2210–2218. <doi:10.1145/3219819.3219994⟩>."""

    def _ftu_exact_ratio(X):
        a, b, c = X.min(), X.mean(), X.max()
        s, fmin, _, _ = brent(
            lambda s: np.var(np.abs(X - s)),
            brack=(a, b, c),
            full_output=True,
        )

        return float(s), float(fmin) / X.var()

    s, ratio = _ftu_exact_ratio(X)

    return s, ratio * 4


def _ftu_approx_statistics(X):
    """Reference: Siffer, A., Fouque, P.-A., Termier, A. and Largouët, C. (2018), Are your data gathered?, In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2210–2218. <doi:10.1145/3219819.3219994⟩>."""

    def _ftu_approx_ratio(X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        a, b, c = X.min(), X.mean(), X.max()
        s, _, _, _ = brent(
            lambda s: np.var((X - s) ** 2),
            brack=(a, b, c),
            full_output=True,
        )
        return float(s), float(np.var(np.abs(X - s)) / X.var())

    s, ratio = _ftu_approx_ratio(X)

    return s, ratio * 4


def _p_val_hat_T(n, t):
    """Compute the p-value of the DFTU using the test statistic T=min(Phi1, Phi2)"""

    # Marginal distributions of Phi1 and Phi2
    a = 0.273
    d = 1
    sigma = a * (d + 1) ** 2 / np.sqrt(n)
    distrib_Phi = norm(loc=1, scale=sigma)
    p_hat_Phi = distrib_Phi.cdf(t)

    # Join distribution of Phi1 and Phi2
    rho = -0.37
    mean = [1, 1]
    cov = [
        [sigma ** 2, rho * sigma ** 2],
        [rho * sigma ** 2, sigma ** 2]
    ]
    distrib_Phi1_Phi2 = multivariate_normal(mean=mean, cov=cov)
    p_hat_Phi1_Phi2 = distrib_Phi1_Phi2.cdf([t, t])

    p_val_hat = 2 * p_hat_Phi - p_hat_Phi1_Phi2

    return p_val_hat


def dftu(x):
    """
    Apply the Double Folding Test of Unimodality (DFTU), a two-step extension of Siffer's Folding Test of Unimodality (FTU). The null hypothesis states that the underlying distribution is unimodal. Small p-values provide evidence against unimodality.

    Parameters:
        x (ndarray): Data

    Returns:
        stat (float): Test statistic
        p_val (float): Associated p-value

    Details:
        The test is based on the statistic:

    .. math::

        T = \\min(\\Phi_1, \\Phi_2)

    :math:`\Phi_1` and :math:`\Phi_2` are obtained from two successive folding steps.

        Hypotheses:

        .. math::

            H_0: T \\geq 1 \\quad \\text{(the distribution is unimodal)}

            H_1: T < 1 \\quad \\text{(the distribution is not unimodal)}

        Small values of :math:`T` provide evidence against unimodality.

    Reference:
        - Becquart, C., Archimbaud, A., Ruiz, A.M., Smida, Z., A Note on the Folding Test of Unimodality: limitations and an improved alternative.
        - Siffer, A., Fouque, P.-A., Termier, A. and Largouët, C. (2018), Are your data gathered?, In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2210–2218. <doi:10.1145/3219819.3219994>.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from icspylab import dftu
        >>> iris = load_iris()
        >>> X = iris.data
        >>> stat, p = dftu(X[:,0])
        >>> print(round(stat, 2), round(p, 2))
        1.15 1.0
    """

    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    n = x.shape[0]

    # Phi1
    _, Phi1 = _ftu_exact_statistics(x)
    # Phi2
    s_approx, _ = _ftu_approx_statistics(x)
    x_new = np.abs(x - s_approx)
    _, Phi2 = _ftu_exact_statistics(x_new)

    stat = np.min([Phi1, Phi2])
    p_val = _p_val_hat_T(n, stat)

    return stat, p_val


def unimodal_crit(X, W, level=0.05, max_select=None, **kwargs):
    """
    Identifies invariant coordinates that are multimodal using the univariate Fouble Folding Test of Unimodality (DFTU).
    Only the first and last components are investigated.

    Parameters:
        X (ndarray): Data to fit the ICS model, where rows are samples and columns are features.
        W (ndarray): Transformation matrix in which each row contains the coefficients of the linear transformation to the corresponding invariant coordinate.
        level (float, default=0.05) The initial level used to make a decision based on the test p-values.
        max_select (int or None, default=None): Maximum number of components to select.

    Returns:
        dict: Summary of the component selection step

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from icspylab import ICS, unimodal_crit
        >>> iris = load_iris()
        >>> X = iris.data
        >>> ics = ICS(S1="cov", S2="cov4")
        >>> selection_res = unimodal_crit(X=X, W=ics.components_)
        >>> print(selection_res.info)
        {'crit': 'unimodal', 'level': 0.05, 'max_select': 3, 'pvalues': array([9.99998344e-01, 9.97549948e-01, 9.99996927e-01, 2.85058691e-12]), 'adjusted_levels': [0.05, 0.025], 'component_names': ['IC_4']}

    """

    def test_fun(x):
        stat, p = dftu(x)
        return {"statistic": stat, "p_value": p}


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
        "crit": "unimodal",
        "level": level,
        "max_select": max_select,
        "pvalues": test_pvals.copy(),
        "adjusted_levels": adjusted_levels,
        "component_names": selected_component_names
    }

    return ComponentSelect(label="unimodal", components=components, n_components=n_components, component_names=selected_component_names, info=info)
