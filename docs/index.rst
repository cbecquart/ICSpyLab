.. Invariant Coordinate Selection documentation master file, created by
   sphinx-quickstart on Wed Jun 14 15:26:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ICSpyLab documentation!
==========================================================

Overview
--------
Invariant Coordinate Selection (ICS) is a data transformation method. It transforms the data, via the simultaneous
diagonalization of two scatter matrices, into an invariant coordinate system or independent components,
depending on the underlying assumptions.
It is particularly useful for dimension reduction. Unlike PCA, ICS is not based on variance maximization but on the
maximization/minimization of a generalized kurtosis, and it is invariant not only to orthogonal data transformations but
to any affine transformation.

This package brings the main functionalities of the `ICS R package <https://cran.r-project.org/web/packages/ICS/index.html>`_
to Python, offering tools for identifying and selecting invariant coordinates in multivariate data.
It includes various covariance estimators, transformation settings,
and plotting utilities. Our extensive testing ensures results consistent with the R package, making it easy for users to
transition from R to Python or start fresh with ICS.

.. _GitHub Repository: https://github.com/cbecquart/ICSpyLab

You can find the source code for this project on our `GitHub Repository`_.

Key Features
------------
- Implementation of the ICS model
- Large catalogue of scatter matrices
- Methods for component selection
- Plotting capabilities for transformed data

Getting Started
---------------
For installation instructions and a quick start guide, see the :doc:`getting_started/index` section.
Some details about ICS and guidelines to use the package are also available.

Examples
--------
For more detailed examples, refer to the :doc:`examples/index` section.

API Documentation
-----------------
The ICSpyLab package provides several modules to work with invariant coordinate selection.
Each module is documented in the :doc:`api/index` section.

Testing
-------
For details on testing, see the :doc:`test` page.

Contributors
------------
For a list of contributors and their contact information, see the :doc:`contributors` page.

Contribute
----------

We welcome contributions of all kinds, including bug fixes, new features, and documentation improvements.

To get started, please check out our `CONTRIBUTING <https://github.com/cbecquart/ICSpyLab/tree/main?tab=contributing-ov-file>`_ guide.

Citation
--------

If you use this package in academic work, please check out our :doc:`citation` page.

References
----------
For references, see the :doc:`references` page.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting_started/index
   examples/index
   api/index
   test
   contributors
   citation
   references

Indices
-------

* :ref:`genindex`