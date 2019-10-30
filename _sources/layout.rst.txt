PyPOP Package Layout
====================

The PyPOP package is laid out with two API levels.  The first provides a high-level
inteface to load summary statistics from traces and calculate various POP metrics from
them.  This is backed by a low level interface which provides more direct access to the
capabilites of the various tools, allowing scripting of custom analyses.

High Level Interface
--------------------

The key components of the PyPOP package are the :py:mod:`~pypop.metrics` and
:py:mod:`~pypop.traceset` modules.  These contain the classes handling the loading and analysis of
traces, and calculation of metrics.

.. automodule:: pypop.traceset

.. automodule:: pypop.metrics


Low Level Interface
-------------------

.. automodule:: pypop.extrae

.. automodule:: pypop.dimemas

.. automodule:: pypop.prv

.. automodule:: pypop.utils


