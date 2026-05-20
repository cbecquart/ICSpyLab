.. _custom_scatter:

Custom Scatter
==============

This section provides an example showing how to define and use a custom scatter
operator with ICS, using the :class:`~icspylab.scatter.Scatter` class.

Suppose you want to apply ICS with the Minimum Covariance Determinant (MCD)
scatter, and you did not realize that it is already implemented in ``ICSpyLab``.

In this example, we use the implementation provided by scikit-learn. The code
below illustrates how to compute the MCD scatter on the Iris dataset.


.. code-block:: python

    from sklearn.datasets import load_iris
    from sklearn.covariance import MinCovDet

    iris = load_iris()
    X = iris.data

    mcd_fit = MinCovDet().fit(X)
    mcd_loc = mcd_fit.location_
    mcd_cov = mcd_fit.covariance_

    print("location: ", mcd_loc)
    print("scatter: ", mcd_cov)


.. code-block:: text

    location:  [5.76788321 3.0540146  3.62992701 1.15693431]
    scatter:  [[ 0.62359585 -0.06870339  1.2013055   0.50690727]
               [-0.06870339  0.19486945 -0.39176586 -0.14266417]
               [ 1.2013055  -0.39176586  3.11591145  1.33218733]
               [ 0.50690727 -0.14266417  1.33218733  0.60010943]]


We can now use this scatter to apply ICS. Recall that the scatter parameters of an
:class:`~icspylab.ics.ICS` instance (``S1`` and ``S2``) must be callables returning a
:class:`~icspylab.scatter.Scatter` object. Each ``Scatter`` object provides three attributes:
:attr:`~icspylab.scatter.Scatter.location`,
:attr:`~icspylab.scatter.Scatter.scatter`, and
:attr:`~icspylab.scatter.Scatter.label`.

Below, we define a function computing the MCD scatter, which we will use as
``S1``:

.. code-block:: python

    from icspylab import Scatter

    def mcd_scatter(X, **kwargs):
        mcd_fit = MinCovDet(**kwargs).fit(X)
        mcd_loc = mcd_fit.location_
        mcd_cov = mcd_fit.covariance_

        return Scatter(location=mcd_loc, scatter=mcd_cov, label="MCD")

    mcd_scatter(X)


.. code-block:: text

    <icspylab.scatter.Scatter at 0x7c2a89188d10>


.. code-block:: python

    from icspylab import ICS

    ics = ICS(S1=mcd_scatter, S2="cov")

    # Fit and transform the ICS model
    X_new = ics.fit_transform(X)


This tutorial demonstrates how ``ICSpyLab`` can be extended through user-defined
scatter operators, enabling advanced customization of the ICS workflow.
