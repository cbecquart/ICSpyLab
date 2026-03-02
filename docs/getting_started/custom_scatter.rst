.. _custom_scatter:

Custom Scatter
==============

This section provides an example showing how to define and use a custom scatter
operator with ICS, using the :class:`icspylab.scatter.Scatter` class.

Suppose you want to apply ICS with the Minimum Covariance Determinant (MCD)
scatter, and you did not realize that it is already implemented in ``ICSpyLab``.

In this example, we use the implementation provided by scikit-learn. The code
below illustrates how to compute the MCD scatter on the Iris dataset.


.. code-block:: python

    import pandas as pd
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

    location:  [5.74328358 3.05074627 3.57462687 1.12985075]
    scatter:  [[ 0.5646937  -0.06801738  1.08803854  0.45863333]
     [-0.06801738  0.18623079 -0.3834139  -0.14166407]
     [ 1.08803854 -0.3834139   2.84995322  1.21262308]
     [ 0.45863333 -0.14166407  1.21262308  0.54209401]]


We can now use this scatter to apply ICS. Recall that the scatter parameters of an
:class:`icspylab.ics.ICS` instance (``S1`` and ``S2``) must be callables returning a
:class:`icspylab.scatter.Scatter` object. Each :class:`Scatter` object provides three attributes:
:attr:`icspylab.scatter.Scatter.location`,
:attr:`icspylab.scatter.Scatter.scatter`, and
:attr:`icspylab.scatter.Scatter.label`.

Below, we define a function computing the MCD scatter, which we will use as
``S1``:

.. code-block:: python

    def mcd_scatter(X, **kwargs):
        mcd_fit = MinCovDet(**kwargs).fit(X)
        mcd_loc = mcd_fit.location_
        mcd_cov = mcd_fit.covariance_

        return Scatter(location=mcd_loc, scatter=mcd_cov, label="MCD")

    mcd_scatter(X)


.. code-block:: text

    <icspylab.scatter.Scatter at 0x7c2a89188d10>


.. code-block:: python

    ics = ICS(S1=mcd_scatter, S2=cov)

    # Fit and transform the ICS model
    ics.fit_transform(X)


This tutorial demonstrates how ``ICSpyLab`` can be extended through user-defined
scatter operators, enabling advanced customization of the ICS workflow.
