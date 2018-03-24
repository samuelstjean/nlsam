.. NLSAM documentation master file, created by
   sphinx-quickstart on Sat Mar 24 10:44:02 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NLSAM's documentation!
=================================

This is the documentation detailing the internal of the non local spatial and angular matching (NLSAM) denoising algorithm for diffusion MRI, which is available at https://github.com/samuelstjean/nlsam.

You can find the original paper and full details of the algorithm as presented in
::
    St-Jean, S., Coupé, P., & Descoteaux, M. (2016)
    "Non Local Spatial and Angular Matching :
    Enabling higher spatial resolution diffusion MRI datasets through adaptive denoising."
    Medical Image Analysis, 32(2016), 115–130.

Which you can grab a copy from the publisher website_ or from arxiv_.

You an find below the documentation for each modules and a few instructions on topics such as noise estimation and detailed installation instructions.

.. toctree::
   :maxdepth: 3
   :caption: Modules

   modules

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Instructions

   wiki/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _website: https://www.sciencedirect.com/science/article/pii/S1361841516000335
.. _arxiv: https://arxiv.org/pdf/1606.07239.pdf