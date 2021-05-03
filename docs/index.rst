.. segmentation_RT documentation master file, created by
   sphinx-quickstart on Tue Feb 16 11:46:48 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

segmentation_RT
===========================================
|Documentation Status| |Build Status| |Coverage Status| |Codacy Badge|

.. |Documentation Status| image:: https://readthedocs.org/projects/segmentation-rt/badge/?version=latest
  :target: https://segmentation-rt.readthedocs.io/en/latest/?badge=latest

.. |Build Status| image:: https://travis-ci.com/guilgautier/DPPy.svg?branch=master
  :target: https://travis-ci.com/BrouBoni/segmentation_RT

.. |Coverage Status| image:: https://coveralls.io/repos/github/BrouBoni/segmentation_RT/badge.svg?branch=main
  :target: https://coveralls.io/github/BrouBoni/segmentation_RT?branch=main

.. |Codacy Badge| image:: https://app.codacy.com/project/badge/Grade/443a2c7e654a4b819711f07ba5ef9ab2
  :target: https://www.codacy.com/gh/BrouBoni/segmentation_RT/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=BrouBoni/segmentation_RT&amp;utm_campaign=Badge_Grade

Automatic contouring with the help of deep learning is a nontrivial matter and a research area of its own.

The purpose of segmentation_RT is to provide a quick method to use deep learning for contouring. Three modules are
provided to do so:

   - :doc:`rs2mask <source/segmentation_rt.rs2mask>`: create a deep learning friendly dataset from dicom data.
   - :doc:`dl <source/segmentation_rt.dl>`: deep learning module for training and testing.
   - :doc:`mask2rs <source/segmentation_rt.mask2rs>`: create RT Structure Set from a previously generated masks.

This is an early release which only allows the use of CT for the moment. MRI supports will come soon.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   source/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Modules
   :hidden:

   source/segmentation_rt.rs2mask
   source/segmentation_rt.dl
   source/segmentation_rt.mask2rs

.. toctree::
   :maxdepth: 2
   :caption: Util
   :hidden:

   source/segmentation_rt.util