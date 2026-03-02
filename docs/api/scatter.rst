Scatter
=======

Module containing scatter matrix calculations and the Scatter class.

This module provides various functions to compute scatter matrices, which are essential for the
ICS algorithm. The implemented scatter matrices include the
covariance matrix, weighted covariance matrix, and pairwise one-step M-estimate of scatter. These
scatter matrices are encapsulated in the Scatter class, which includes information about
the location (mean) and a label describing the type of scatter matrix. If you want to use ICS with other scatter
matrices than the ones provided in this module, you would need to create a Scatter object.
See the :ref:`custom_scatter` for a complete example.

Most of the implemented scatters come from the `R package ICS <https://cran.r-project.org/web/packages/ICS/index.html>`_.

.. automodule:: icspylab.scatter
   :members:
   :undoc-members:
   :show-inheritance:

