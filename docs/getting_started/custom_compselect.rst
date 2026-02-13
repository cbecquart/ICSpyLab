Custom component selection
==========================


This section illustrates how to use a custom method to select the invariant components.

Let's start by some data exploration, we apply ICS with the scatter pair COV-COV4 on the Iris dataset.
By default, method_select=None and all invariant components are kept.

.. code-block:: python

    import pandas as pd
    from icspylab import ICS, cov, cov4
    from sklearn.datasets import load_iris

    # Load dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Instantiate ICS object
    ics = ICS(S1=cov, S2=cov4, algorithm='standard', S2_args={'alpha': 1, 'cf': 2})

    # Fit and transform the ICS model
    X_ics = ics.fit_transform(X)


Looking at the invariant coordinates on the plot above, you decide that you want to keep only the last component.
If you just need a one shot utilisation you can simply apply the selection method on the output X_ics.

.. code-block:: python

    X_ics_reduced = X_ics[:, -1]
    X_ics_reduced.shape


However, if you want to integrate the component selection step into a pipeline, you might want to implement this
step during the ICS fit method. To do so, recall that the method_select parameter of an :class:`ICS` instance is
(if not None) a callable returning a :class:`ComponentSelect` object. Each :class:`ComponentSelect` have the
following attributes: label, components, n_components, component_names, info.




