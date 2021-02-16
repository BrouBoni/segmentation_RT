.. segmentation_RT documentation master file, created by
   sphinx-quickstart on Tue Feb 16 11:46:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to segmentation_RT's documentation!
===========================================

Automatic contouring with the help of deep learning is a nontrivial matter and a research area of its own.

The purpose of segmentation_RT is to provide a quick method to use deep learning for contouring. Three modules are
provided to do so:

   - :ref:`rs2mask <rs2mask>`: create a dataset from dicom data
   - dl: deep learning module for training and testing
   - :ref:`mask2rs <mask2rs>`: create RT Structure Set from mask.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   rs2mask/modules
   mask2rs/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
