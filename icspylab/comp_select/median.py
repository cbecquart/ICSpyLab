import numpy as np
from sklearn.utils.validation import check_array
from .base import ComponentSelect, _validate_nb_select


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

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from icspylab import ICS, med_crit
        >>> iris = load_iris()
        >>> X = iris.data
        >>> ics = ICS(S1="cov", S2="cov4")
        >>> X_new = ics.fit_transform(X)
        >>> selection_res = med_crit(kurtosis=ics.kurtosis_, W=ics.components_)
        >>> print(selection_res.info)
        {'crit': 'med', 'nb_select': 3, 'gen_kurtosis': array([1.20739878, 1.0269412 , 0.9292235 , 0.74046722]), 'med_gen_kurtosis': np.float64(0.9780823483964416), 'gen_kurtosis_diff_med': array([0.22931644, 0.04885885, 0.04885885, 0.23761513]), 'component_names': ['IC_4', 'IC_1', 'IC_2']}

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
