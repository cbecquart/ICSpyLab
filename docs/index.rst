.. Invariant Coordinate Selection documentation master file, created by
   sphinx-quickstart on Wed Jun 14 15:26:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ICSpyLab documentation!
==========================================================

Overview
========
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
- Detailed summary of the results
- Plotting capabilities for transformed data (coming soon)
- Scatter Matrices

Installation
------------
For installation instructions, see the :doc:`installation` page.

Usage
-----
For examples and usage, see the :doc:`usage` page.

API Documentation
-----------------
The ICSpyLab package provides several modules to work with invariant coordinate selection. For detailed information on each module, including available functions and their usage, see the following sections:

- :doc:`ics`: Core functions and classes for performing invariant coordinate selection.
- :doc:`plot`: Functions for visualizing the results of ICS analysis.
- :doc:`scatter`: Tools for generating scatter plots and other visualizations.
- :doc:`utils`: Utility functions to support the main functionalities.

Each section includes detailed descriptions, parameter information, return types, and usage examples to help you effectively use the package.

Testing
-------
For details on testing, see the :doc:`test` page.

Contributors
------------
For a list of contributors and their contact information, see the :doc:`contributors` page.

References
----------
For references, see the :doc:`references` page.

.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   usage
   ics
   plot
   scatter
   utils
   test
   contributors
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`