Installing PyPOP
================

Prerequisites
-------------

The major prerequisite for PyPOP is that the Paramedir_ and Dimemas_ tools are installed and
available on the path.

These can be obtained from the `BSC tools project`_ along with instructions for their installation.

.. _Paramedir: https://tools.bsc.es/paraver#batchprocessing
.. _Dimemas: https://tools.bsc.es/dimemas
.. _BSC tools project: https://tools.bsc.es

These must be available on the system PATH (Linux ``$PATH`` or Windows ``%PATH%`` variables) so
that they can be found by PyPOP.

Installation
------------

PyPOP is designed to be installed using pip_ the Python package manager.

.. _pip: https://pypi.org/project/pip/

This will install both pypop and any required dependencies which are not currently installed.

To install from the github repository do one of the following:

Directly install using pip
^^^^^^^^^^^^^^^^^^^^^^^^^^

Instruct pip to install by downloading directly:

.. code-block:: bash

  $ pip install [--user] git+https://github.com/numericalalgorithmsgroup/pypop

The optional ``--user`` directive instructs pip to install to the users home directory instead of
the system site package directory.


Clone then install
^^^^^^^^^^^^^^^^^^

Alternatively, the repository can first be cloned, then the package installed:

.. code-block:: bash

  $ git clone https://github.com/numericalalgorithmsgroup/pypop
  $ cd pypop
  $ pip install [--user] .
