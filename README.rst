segmentation_RT: Auto-segmentation Deep Learning Toolbox for radiotherapy
=========================================================================

Introduction
------------

Automatic contouring with the help of deep learning is a nontrivial matter and a research area of its own.

The purpose of segmentation_RT is to provide a quick method to use deep learning for contouring. Three modules are
provided to do so:

   - rs2mask: create a dataset from dicom data.
   - dl: deep learning module for training and testing.
   - mask2rs: create RT Structure Set from mask.

Installation
------------

segmentation_RT works with `Python 3.8 <http://docs.python.org/3/>`__.

Dependencies
~~~~~~~~~~~~

This project is based on `PyTorch 1.7.1 <https://pytorch.org>`__ and uses `TorchIO 0.18.29 <hhttps://torchio.readthedocs.io>`__.

How to use it
-------------

The main segmentation_RT documentation is available online at `http://segmentation_RT.readthedocs.io <http://segmentation_RT.readthedocs.io>`_.
There are also an main example available at https://github.com/BrouBoni/segmentation_RT/blob/main/main.py.
For more details, check below.
