import numpy as np


class ComponentSelect:
    """
    A class to represent a component selection method and its related data.

    Attributes:
        label (str): Label of the component selection method.
        components (ndarray): Invariant components selected by the method.
        n_components (int): Number of invariant components selected by the method.
        component_names (ndarray): Names of invariant components selected by the method.
        info (dict or None): Additional information specific to the method.
    """

    def __init__(self, label, components, n_components, component_names, info):
        """
        Initialize the ComponentSelect object with specified parameters.

        Parameters:
            label (str): Label of the component selection method.
            components (ndarray): Invariant components selected by the method.
            n_components (int): Number of invariant components selected by the method.
            component_names (list): Names of invariant components selected by the method.
            info (dict or None): Additional information specific to the method.
        """
        self.label = label
        self.components = components
        self.n_components = n_components
        self.component_names = component_names
        self.info = info


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
