#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

import os
import pickle

from warnings import warn

import pandas
import numpy

try:
    import qgrid
except ImportError:
    pass

from .tracestats import TraceStats

_k_grid_options = {
    # SlickGrid options
    "fullWidthRows": True,
    "syncColumnCellResize": True,
    "forceFitColumns": False,
    "defaultColumnWidth": 100,
    "rowHeight": 28,
    "enableColumnReorder": False,
    "enableTextSelectionOnCells": True,
    "editable": True,
    "autoEdit": True,
    "explicitInitialization": True,
    # Qgrid options
    "maxVisibleRows": 15,
    "minVisibleRows": 8,
    "sortable": False,
    "filterable": False,
    "highlightSelectedCell": False,
    "highlightSelectedRow": True,
}

_k_idx_cols = ["process", "thread"]

_k_threads_stats = [
    "Serial Useful Computation",
    "Total Runtime",
    "Useful Instructions",
    "Useful Cycles",
    "Parallel Region Time",
    "Parallel Useful",
]

_k_supported_stats = {"threads": _k_threads_stats}


class ManualEntryStats(TraceStats):
    """Manually entered summary statistics

    This class supports loading manually recorded statistics from a CSV file or
    interactive entry in an IPython notebook via the qgrid package.  The data storage is
    backed only by the CSV format and does not use pickle caching.  Instead,
    metadata is written as key=value pairs into the csv header.

    Attributes
    ----------
    metadata
    stats
    """

    # Override init because this class is special in terms of the way we can choose to
    # input the data, in particular we do not support caching because the data is already
    # provided in the csv or can be set by the user manually
    def __init__(
        self, stats_file, processes=1, threads=1, analysis_type="Threads", **kwargs
    ):
        """
        
        """

        self._stats_file = stats_file
        # Begin by looking up appropriate stats for analysis type
        try:
            stats_to_capture = _k_supported_stats[analysis_type.lower()]
        except IndexError:
            IndexError("Unknown analysis_type {}".format(analysis_type))
        except AttributeError:
            TypeError("analysis_type must be a string")

        # First see if there is a valid csv to load
        try:
            self._stats = pandas.read_csv(
                stats_file, index_col=_k_idx_cols, dtype=numpy.float64
            )
        except FileNotFoundError:
            self._stats = pandas.DataFrame(
                columns=stats_to_capture,
                index=pandas.MultiIndex.from_product(
                    (range(1, processes + 1), range(1, threads + 1)), names=_k_idx_cols,
                ),
                dtype=numpy.float64,
            )

    def _sheetupdate(self, event, sheet):
        """Callback handler to save updated sheet
        """
        # Make sure self._stats is up to date, and save to file
        self._stats = self._sheet.get_changed_df()
        self._stats.to_csv(self._stats_file)

    def table(self):
        try:
            self._sheet = qgrid.show_grid(
                self._stats, show_toolbar=False, grid_options=_k_grid_options
            )
        except NameError:
            raise RuntimeError("Interactive stats editing requires the qgrid package")

        self._sheet.on(["cell_edited"], self._sheetupdate)

        return self._sheet
