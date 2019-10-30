Quickstart Guide
================

Jupyter notebooks are intended to be the primary interface to PyPOP.  This guide uses several
example notebooks to demonstrate the core functionality of PyPOP for calculation of the POP Metrics
as well as advanced analysis of trace files.

Requirements
------------

This guide assumes that you have successfully :doc:`installed PyPOP<install>` and are able
to open `Jupyter (IPython) notebooks`_.

.. _Jupyter (IPython) notebooks: https://jupyter.org/install

In addition, you will need a copy of the examples. These are located in the examples directory,
which can be found using the `pypop.examples` module:

.. command-output:: python -m pypop.examples

Copy these to directory where you have read permissions, e.g.

.. code-block:: bash

  $ cp -vr $(python -m pypop.examples) $HOME/pypop_examples


Analysis using IPython Notebooks
--------------------------------

Example notebooks are provided for the different analysis types

Running the MPI Metrics Example Notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To open the notebook you can either start a `Jupyter session`_, and navigate to the
examples directory you created, or open the notebook directly from the console:

.. _Jupyter session: https://jupyter.readthedocs.io/en/latest/running.html

.. code-block:: bash
  
  $ cd $HOME/pypop_examples
  $ jupyter notebook MPI\ Metrics\ Example.ipynb

This demonstrates basic, in-notebook use of the PyPOP tool for generating the POP MPI metrics, and
shows how the underlying data can be accessed as Pandas Dataframes.  The example notebook is
self-documenting, and code cells can be run directly in the browser using :kbd:`shift` +
:kbd:`enter`.

Running the Hybrid Metrics Example Notebook
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There is also a hybrid metrics example notebook, which

.. code-block:: bash

  $ cd $HOME/pypop_examples
  $ jupyter notebook Hybrid\ Metrics\ Example.ipynb

This is a slightly more detailed example, showing how the code can be used to perform a more in
depth analysis that is supported by the command line version of the tool.

Command Line Scripts
--------------------

To quickly generate strong scaling metrics for MPI and MPI/OpenMP hybrid codes on the command line,
some convenience scripts are provided.

MPI Strong Scaling Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^

The `pypop-mpi-metrics` script takes a list of trace files and calculates the POP MPI strong
scalings metrics.  By default output is provided in the form of a csv file containing the metrics,
along with images with the metric table and a speedup scaling plot.

.. command-output:: pypop-mpi-metrics -h


MPI/OpenMP Hybrid Strong Scaling Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `pypop-hybrid-metrics` script takes a list of trace files and calculates prototype hybrid
MPI+OpenMP strong scalings metrics.  By default output is provided in the form of a csv file
containing the metrics, along with images with the metric table and a speedup scaling plot.

.. command-output:: pypop-hybrid-metrics -h


