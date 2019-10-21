#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Routines for calculating and managing trace data
"""

import os
import pickle

from os.path import dirname
from warnings import warn

from pkg_resources import resource_filename

import pandas

try:
    from tqdm.auto import tqdm
except ImportError:
    from .utils import return_first_arg as tqdm

from .dimemas import dimemas_idealise
from .extrae import paramedir_analyze, chop_prv_to_roi, remove_trace
from .prv import get_prv_header_info
from .utils import chunked_md5sum


class RunData:
    """
    Attributes
    ----------
    traceinfo: `TraceInfo`
        Trace metadata information

    stats: `pd.DataFrame`
        Trace statistics
    """

    def __init__(self, traceinfo, stats):
        self.traceinfo = traceinfo
        self.stats = stats

    def __hash__(self):
        return hash(self.traceinfo)

    def __iter__(self):
        self._n = 0
        return self

    def _repr_html_(self):
        return self.stats._repr_html_()


base_configs = {
    k: resource_filename(__name__, v)
    for k, v in {
        "Serial Useful Computation": "cfgs/serial_useful_computation.cfg",
        "Total Runtime": "cfgs/total_runtime.cfg",
        "Useful Instructions": "cfgs/useful_instructions.cfg",
        "Useful Cycles": "cfgs/useful_cycles.cfg",
    }.items()
}

omp_configs = {
    k: resource_filename(__name__, v)
    for k, v in {
        "OpenMP Total Runtime": "cfgs/omp_total_runtime.cfg",
        "OpenMP Useful Computation": "cfgs/omp_useful_computation.cfg",
    }.items()
}

ideal_configs = {
    k: resource_filename(__name__, v)
    for k, v in {
        "Ideal Useful Computation": "cfgs/total_useful_computation.cfg",
        "Ideal Runtime": "cfgs/total_runtime.cfg",
    }.items()
}


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

    cache_stats: bool
        Cache the calculated statistics as a pickled data file in the trace
        directory, along with the checksum of the source trace file. (Default True)

    ignore_cache: bool
        By default, if a cache file is present for a given trace it will be used.
        This behaviour can be overridden by setting ignore_cache=True.

    chop_to_roi: bool
        If true, cut trace down to the section bracketed by the first pair of
        Extrae_startup and Extrae_shutdown commands. Default false.

    no_progress: bool
        If true, disable the use of tqdm progress bar

    """

    def __init__(
        self,
        path_list=None,
        cache_stats=True,
        ignore_cache=False,
        chop_to_roi=False,
        no_progress=False,
    ):
        # Setup data structures
        self.traces = set()

        # Add traces
        self.add_traces(path_list, cache_stats, ignore_cache, chop_to_roi, no_progress)

    def add_traces(
        self,
        path_list=None,
        cache_stats=True,
        ignore_cache=False,
        chop_to_roi=False,
        no_progress=False,
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

        cache_stats: bool
            Cache the calculated statistics as a pickled data file in the trace
            directory, along with the checksum of the source trace file. (Default True)

        ignore_cache: bool
            By default, if a cache file is present for a given trace it will be used.
            This behaviour can be overridden by setting ignore_cache=True.

        chop_to_roi: bool
            If true, cut trace down to the section bracketed by the first pair of
            Extrae_startup and Extrae_shutdown commands. Default false.

        no_progress: bool
            If true, disable the use of tqdm progress bar

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
                    self._collect_statistics(
                        path, cache_stats, ignore_cache, chop_to_roi
                    )
                )

    def by_key(self, key):
        """Return a dictionary of traces with given key

        Parameters
        ----------
        key: func
            A function which takes a RunData object and returns a valid dictionary key

        Returns
        -------
        traces_by_key: dict
            A dictionary of traces organised by the requested key
        """

        return {key(v): v for v in self.traces}

    def by_commsize(self):
        """Return a dictionary of traces keyed by commsize

        This is a helper function equivalent to

        by_key(lambda x: x.traceinfo.application_layout.commsize)

        Returns
        -------
        traces_by_key: dict
            A dictionary of traces organised by the requested key
        """

        return self.by_key(lambda x: x.traceinfo.application_layout.commsize)

    def _collect_statistics(self, trace, cache_stats, ignore_cache, chop_to_roi):

        # Calculate a checksum for cache purposes
        csum = chunked_md5sum(trace)
        trace_dir = dirname(trace)
        pkl_path = os.path.join(trace_dir, "stats_{}.pkl".format(csum))

        if not ignore_cache:
            # If we can, load the cached version
            try:
                with open(pkl_path, "rb") as fh:
                    traceinfo, stats = pickle.load(fh)
                    # Quick sanity check
                    if (
                        traceinfo.application_layout.commsize
                        != get_prv_header_info(trace).application_layout.commsize
                    ):
                        raise ValueError("Mismatched cache -- hash collision?")

                    # Assign data from pickle and do next file
                    return RunData(traceinfo, stats)

            # Otherwise continue and analyze the file
            except FileNotFoundError:
                pass

        # If ignoring cache, or cache not present, do the analysis
        if cache_stats:
            cache_stats = pkl_path
        return self._analyze_tracefile(trace, cache_stats, chop_to_roi)

    def _analyze_tracefile(self, trace, cache_path, chop_to_roi):

        traceinfo = get_prv_header_info(trace)

        if chop_to_roi:
            cut_trace = chop_prv_to_roi(trace)
        else:
            cut_trace = trace

        stats = [
            paramedir_analyze(
                cut_trace, cfg, index_by_thread=True, statistic_names=[name]
            )
            for name, cfg in base_configs.items()
        ]

        hybrid = False
        try:
            omp_stats = [
                paramedir_analyze(
                    cut_trace, cfg, index_by_thread=True, statistic_names=[name]
                )
                for name, cfg in omp_configs.items()
            ]
            stats += omp_stats
            hybrid = True
        except RuntimeError:
            pass

        # Remember to clean up after ourselves
        if chop_to_roi:
            remove_trace(cut_trace)

        try:
            ideal_trace = dimemas_idealise(trace)
            if chop_to_roi:
                cut_ideal_trace = chop_prv_to_roi(ideal_trace)
            else:
                cut_ideal_trace = ideal_trace
            stats.extend(
                [
                    paramedir_analyze(
                        cut_ideal_trace,
                        cfg,
                        index_by_thread=True,
                        statistic_names=[name],
                    )
                    for name, cfg in ideal_configs.items()
                ]
            )
            # Keeping things tidy
            if chop_to_roi:
                remove_trace(cut_ideal_trace)
            remove_trace(ideal_trace)
        except RuntimeError as err:
            warn(
                "Failed to run Dimemas:\n{}"
                "Continuing with reduced MPI detail.".format(err)
            )

        stats = pandas.concat(stats).T
        stats["IPC"] = stats["Useful Instructions"] / stats["Useful Cycles"]

        stats["Total Useful Computation"] = stats["Serial Useful Computation"]
        stats["Total Non-MPI Runtime"] = stats["Serial Useful Computation"]

        if hybrid:
            stats["Total Useful Computation"] += stats["OpenMP Useful Computation"]
            stats["Total Non-MPI Runtime"] += stats["OpenMP Total Runtime"]

        stats["Frequency"] = stats["Useful Cycles"] / stats["Total Useful Computation"]

        if cache_path:
            with open(cache_path, "wb") as fh:
                pickle.dump((traceinfo, stats), fh)

        return RunData(traceinfo, stats)
