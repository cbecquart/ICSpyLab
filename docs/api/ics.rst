ICS
===

Module containing the main Invariant Coordinate Selection (ICS) Class and associated methods.

The ICS class provides methods to fit the ICS model from the data, transform data using the model,
and provide a detailed summary of the results. This module relies on scatter matrices
(defined in the Scatter page). The ICS class supports three different
algorithms for applying ICS to data: ('standard', 'whiten', and 'QR'), which can be specified
as parameters during instantiation. Additional options such as the choice of scatter matrices,
centering the data, and fixing the signs can also be defined.

This implementation is based on the function ICS-S3 from the R package `ICS <https://cran.r-project.org/web/packages/ICS/index.html>`_.
For more details about the algorithms 'standard', 'whiten' and 'QR', as well as the 'fix_signs' argument, see the R package
`documentation <https://cran.r-project.org/web/packages/ICS/ICS.pdf>`_ (function ICS-S3).

.. automodule:: icspylab.ics
   :members:
   :undoc-members:
   :show-inheritance:

