#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""\
Trace Data Summary Statistics Calculator Classes
------------------------------------------------

PyPop provides classes for importing summary statistics for different profiling tools.

Currently the following tools are supported:

    * Extrae

"""

from warnings import warn

try:
    from tqdm.auto import tqdm
except ImportError:
    from .utils import return_first_arg as tqdm

from .trace import Trace


class TraceSet:
    """A set of tracefiles for collective analysis

    Collect statistics from provided trace files, currently only Extrae .prv files are
    supported.

    This are the necessary statistics for calculating the POP metrics (including hybrid
    metrics) for the provided application traces. Data caching is used to improve
    performance of subsequent analysis runs, with md5 checksumming used to detect
    tracefile changes. (This is particularly useful for large trace files where
    calculation of statistics can take a long time).

    Parameters
    ----------
    path_list: str or iterable of str
       String or iterable of strings providing path(s) to the tracefiles of interest.

    force_recalculation: bool
        Force recalculation of statistics even if a `.tracesummary` file is present.
        If the original tracefile is not present this will cause an error to be
        thrown.

    chop_to_roi: bool
        If true, cut trace down to the section bracketed by the first
        Extrae_startup and last Extrae_shutdown event. Default false.

    no_progress: bool
        If true, disable the use of tqdm progress bar

    outpath: str or None
        Optional output directory for chopped and idealised traces. (If not specified
        will be created in a temporary folder and deleted.)

    eager_load: bool (default True)
        If `True` full analysis of traces immediately, if `False` wait until statistics
        are requested for a given trace
    """

    def __init__(
        self,
        path_list=None,
        force_recalculation=False,
        chop_to_roi=False,
        no_progress=False,
        outpath=None,
        eager_load=True,
        ignore_cache=None,
        tag=None,
        chop_fail_is_error=False,
    ):
        # Setup data structures
        self.traces = set()

        if ignore_cache is not None:
            warn(
                "ignore_cache is deprecated. Please use force_recalculation.",
                FutureWarning,
            )
            force_recalculation = ignore_cache

        # Add traces
        self.add_traces(
            path_list,
            force_recalculation,
            chop_to_roi,
            no_progress,
            outpath,
            eager_load,
            tag,
            chop_fail_is_error,
        )

    def add_traces(
        self,
        path_list=None,
        force_recalculation=False,
        chop_to_roi=False,
        no_progress=False,
        outpath=None,
        eager_load=True,
        tag=None,
        chop_fail_is_error=False,
    ):
        """Collect statistics from provided trace files, currently only Extrae .prv files
        are supported.

        This are the necessary statistics for calculating the POP metrics (including
        hybrid metrics) for the provided application traces. Data caching is used to
        improve performance of subsequent analysis runs, with md5 checksumming used to
        detect tracefile changes. (This is particularly useful for large trace files
        where calculation of statistics can take a long time).

        Parameters
        ----------
        path_list: str or iterable of str
            String or iterable of strings providing path(s) to the tracefiles of
            interest.

        force_recalculation: bool
            Force recalculation of statistics even if a `.tracesummary` file is present.
            If the original tracefile is not present this will cause an error to be
            thrown.

        chop_to_roi: bool
            If true, cut trace down to the section bracketed by the first pair of
            Extrae_startup and Extrae_shutdown commands. Default false.

        no_progress: bool
            If true, disable the use of tqdm progress bar

        outpath: str or None
            Optional output directory for chopped and idealised traces. (If not specified
            will be created in a temporary folder and deleted.)

        eager_load: bool (default True)
            If `True` full analysis of traces immediately, if `False` wait until
            statistics are requested for a given trace.
        """
        if isinstance(path_list, str):
            path_list = [path_list]
        # Try to get a list length for tqdm
        try:
            npath = len(path_list)
        except TypeError:
            npath = None

        if path_list:
            for path in tqdm(path_list, total=npath, disable=no_progress, leave=False):
                self.traces.add(
                    Trace.load(
                        path,
                        force_recalculation=force_recalculation,
                        chop_to_roi=chop_to_roi,
                        outpath=outpath,
                        eager_load=eager_load,
                        tag=tag,
                        chop_fail_is_error=chop_fail_is_error,
                    )
                )

    def by_key(self, key):
        """Return a dictionary of traces keyed by a user supplied key derivation function

        Parameters
        ----------
        key: func
            A function which takes a Trace object and returns a valid dictionary key

        Returns
        -------
        traces_by_key: dict
            A dictionary of traces organised by the requested key
        """

        return {key(v): v for v in self.traces}

    def by_commsize(self):
        """Return a dictionary of traces keyed by commsize

        This is a helper function equivalent to

        by_key(lambda x: x.metadata.num_processes)

        Returns
        -------
        traces_by_key: dict
            A dictionary of traces organised by the requested key
        """

        return self.by_key(lambda x: x.metadata.num_processes)

    def by_threads_per_process(self):
        """Return a dictionary of traces keyed by threads per process

        This is a helper function equivalent to

        by_key(lambda x: x.metadata.threads_per_process[0])

        Returns
        -------
        traces_by_key: dict
            A dictionary of traces organised by the requested key
        """

        return self.by_key(lambda x: x.metadata.threads_per_process[0])

    def by_hybrid_layout(self):
        """Return a dictionary of traces keyed by hybrid layout

        This is a helper function equivalent to

        by_key(
            lambda x: "{}x{}".format(
                x.metadata.num_processes, x.metadata.threads_per_process[0]
            )
        )

        Returns
        -------
        traces_by_key: dict
            A dictionary of traces organised by the requested key
        """

        return self.by_key(
            lambda x: "{}x{}".format(
                x.metadata.num_processes, x.metadata.threads_per_process[0]
            )
        )
