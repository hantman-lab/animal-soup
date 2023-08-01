.. animal-soup documentation master file, created by
   sphinx-quickstart on Tue Aug  1 11:27:02 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to animal-soup's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   User Guide <user_guide>
   API<api/index>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Summary
=======

``animal-soup`` is an automated behavioral classification software that allows members of the
Hantman Lab to quickly and efficiently generate ethograms for their reach-to-grab behavioral data.
It is a collection of "pandas extensions", which are functions that operate on pandas DataFrames and Series.
``animal-soup`` essentially creates a user-friendly "psuedo-database" of your behavioral data, interfacing with
with `fastplotlib <https://github.com/fastplotlib/fastplotlib>`_ for fast visualization and editing of
predicted behavioral labels. Behavioral classification training and inference has been adopted from
`DeepEthogram <https://github.com/jbohnslav/deepethogram>`_

Installation
============

For installation, please see the instructions in the README on GitHub:

https://github.com/hantman-lab/animal-soup

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`