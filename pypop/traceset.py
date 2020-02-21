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

import os
import pickle
import shutil

from os.path import dirname, splitext, basename
from warnings import warn

from pkg_resources import resource_filename

import pandas
import numpy

try:
    from tqdm.auto import tqdm
except ImportError:
    from .utils import return_first_arg as tqdm

from .dimemas import dimemas_idealise
from .extrae import paramedir_analyze_any_of, chop_prv_to_roi, remove_trace
from .prv import get_prv_header_info
from .utils import chunked_md5sum
from . import config


class RunData:
    """
    Attributes
    ----------
    metadata: :class:`~pypop.prv.TraceMetadata`
        Trace metadata information

    stats: `pd.DataFrame`
        Trace statistics

    tracefile: str
        Tracefile path

    chopped: bool
        Was tracefile chopped to ROI before analysis?
    """

    def __init__(self, metadata, stats, tracefile=None, chopped=False):
        self.metadata = metadata
        self.stats = stats
        self.tracefile = tracefile
        self.chopped = chopped

    def __hash__(self):
        return hash((self.metadata, self.chopped))

    def __iter__(self):
        self._n = 0
        return self

    def _repr_html_(self):
        return self.stats._repr_html_()


base_configs = {
    k: tuple(resource_filename(__name__, w) for w in v)
    for k, v in {
        "Serial Useful Computation": (
            "cfgs/serial_useful_computation.cfg",
            "cfgs/serial_useful_computation_omp_loop.cfg",
            "cfgs/serial_useful_computation_omp_task.cfg",
            "cfgs/serial_useful_computation_no_omp.cfg",
        ),
        "Total Runtime": (
            "cfgs/total_runtime_excl_disabled.cfg",
            "cfgs/total_runtime.cfg",),
        "Useful Instructions": ("cfgs/useful_instructions.cfg",),
        "Useful Cycles": ("cfgs/useful_cycles.cfg",),
    }.items()
}

omp_configs = {
    k: tuple(resource_filename(__name__, w) for w in v)
    for k, v in {
        "OpenMP Total Runtime": ("cfgs/omp_total_runtime.cfg",),
        "OpenMP Useful Computation": (
            "cfgs/omp_useful_computation.cfg",
            "cfgs/omp_useful_computation_loop.cfg",
            "cfgs/omp_useful_computation_task.cfg",
        ),
    }.items()
}

ideal_configs = {
    k: tuple(resource_filename(__name__, w) for w in v)
    for k, v in {
        "Ideal Useful Computation": ("cfgs/total_useful_computation.cfg",),
        "Ideal Runtime": ("cfgs/total_runtime.cfg",),
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
        
    outpath: str or None
        Optional output directory for chopped trace. (If not specified will be
        created in a temporary folder and deleted.)

    """

    def __init__(
        self,
        path_list=None,
        cache_stats=True,
        ignore_cache=False,
        chop_to_roi=False,
        no_progress=False,
        outpath=None,
    ):
        # Setup data structures
        self.traces = set()

        # Add traces
        self.add_traces(path_list, cache_stats, ignore_cache, chop_to_roi, no_progress, outpath)

    def add_traces(
        self,
        path_list=None,
        cache_stats=True,
        ignore_cache=False,
        chop_to_roi=False,
        no_progress=False,
        outpath=None,
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
            
        outpath: str or None
            Optional output directory for chopped trace. (If not specified will be
            created in a temporary folder and deleted.)

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
                        path, cache_stats, ignore_cache, chop_to_roi, outpath,
                    )
                )

    def by_key(self, key):
        """Return a dictionary of traces keyed by a user supplied key derivation function

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

        by_key(lambda x: x.metadata.application_layout.commsize)

        Returns
        -------
        traces_by_key: dict
            A dictionary of traces organised by the requested key
        """

        return self.by_key(lambda x: x.metadata.application_layout.commsize)

    def by_threads_per_process(self):
        """Return a dictionary of traces keyed by threads per process

        This is a helper function equivalent to

        by_key(lambda x: x.metadata.application_layout.rank_threads[0][0])

        Returns
        -------
        traces_by_key: dict
            A dictionary of traces organised by the requested key
        """

        return self.by_key(lambda x: x.metadata.application_layout.rank_threads[0][0])

    def _collect_statistics(self, trace, cache_stats, ignore_cache, chop_to_roi, outpath):

        # Calculate a checksum for cache purposes
        csum = chunked_md5sum(trace)
        trace_dir = dirname(trace)
        pkl_path = os.path.join(trace_dir, "stats_{}.pkl".format(csum))

        if not ignore_cache:
            # If we can, load the cached version
            try:
                with open(pkl_path, "rb") as fh:
                    metadata, stats = pickle.load(fh)
                    # Quick sanity check
                    if (
                        metadata.application_layout.commsize
                        != get_prv_header_info(trace).application_layout.commsize
                    ):
                        raise ValueError("Mismatched cache -- hash collision?")

                    # Assign data from pickle and do next file
                    return RunData(metadata, stats, trace)

            # Otherwise continue and analyze the file
            except FileNotFoundError:
                pass

        # If ignoring cache, or cache not present, do the analysis
        if cache_stats:
            cache_stats = pkl_path
        return self._analyze_tracefile(trace, cache_stats, chop_to_roi, outpath)

    def _analyze_tracefile(self, trace, cache_path, chop_to_roi, outpath):

        metadata = get_prv_header_info(trace)
        
        if outpath:
            try:
                os.makedirs(outpath, exist_ok=True)
            except OSError as err:
                print("FATAL: {}".format(err))
    
        if chop_to_roi:
            if outpath:
                tgtname = ".chop".join(splitext(basename(trace)))
                outfile = os.path.join(outpath, tgtname)
            else:
                outfile=None
            cut_trace = chop_prv_to_roi(trace, outfile)
        else:
            cut_trace = trace

        stats = [
            paramedir_analyze_any_of(
                cut_trace, cfg, index_by_thread=True, statistic_names=[name]
            )
            for name, cfg in base_configs.items()
        ]

        try:
            omp_stats = [
                paramedir_analyze_any_of(
                    cut_trace, cfg, index_by_thread=True, statistic_names=[name]
                )
                for name, cfg in omp_configs.items()
            ]
            stats += omp_stats
        except RuntimeError:
            skel = next(iter(stats))
            zero_df = pandas.DataFrame(index=skel.T.index)
            for name in omp_configs:
                zero_df[name] = 0
            stats.append(zero_df.T)

        # Remember to clean up after ourselves
        if chop_to_roi and not outpath:
            remove_trace(cut_trace)

        try:
            ideal_trace = dimemas_idealise(trace, outpath)
            if chop_to_roi:
                if outpath:
                    tgtname = ".chop".join(splitext(basename(ideal_trace)))
                    outfile = os.path.join(outpath, tgtname)
                else:
                    outfile=None
                cut_ideal_trace = chop_prv_to_roi(ideal_trace, outfile)
            else:
                cut_ideal_trace = ideal_trace
            stats.extend(
                [
                    paramedir_analyze_any_of(
                        cut_ideal_trace,
                        cfg,
                        index_by_thread=True,
                        statistic_names=[name],
                    )
                    for name, cfg in ideal_configs.items()
                ]
            )

            # Keeping things tidy
            if not outpath:
                if chop_to_roi:
                    remove_trace(cut_ideal_trace)
                remove_trace(ideal_trace)
        except RuntimeError as err:
            warn(
                "Failed to run Dimemas: {}\n"
                "Continuing with reduced MPI detail.".format(err)
            )
            # Get an object with the correct layout
            skel = next(iter(stats))
            nan_df = pandas.DataFrame(index=skel.T.index)
            for name in ideal_configs:
                nan_df[name] = numpy.nan

            stats.append(nan_df.T)

        stats = pandas.concat(stats).T
        stats["IPC"] = stats["Useful Instructions"] / stats["Useful Cycles"]

        stats["Total Useful Computation"] = stats["Serial Useful Computation"]
        stats["Total Non-MPI Runtime"] = stats["Serial Useful Computation"]

        stats["Total Useful Computation"] += stats["OpenMP Useful Computation"]
        stats["Total Non-MPI Runtime"] += stats["OpenMP Total Runtime"]

        stats["Frequency"] = stats["Useful Cycles"] / stats["Total Useful Computation"]

        if (
            stats["Total Non-MPI Runtime"].loc[:, 1].max()
            > stats["Ideal Runtime"].loc[:, 1].max()
        ):
            raise RuntimeError(
                "Illegal Ideal Runtime value (less than useful computation)"
            )

        if cache_path:
            with open(cache_path, "wb") as fh:
                pickle.dump((metadata, stats), fh)
                
        # Clean up any temporary data
        if config._tmpdir_path:
            shutil.rmtree(os.path.join(config._tmpdir_path, "dimemas_tmpdir"),ignore_errors=True)
            shutil.rmtree(os.path.join(config._tmpdir_path, "paramedir_tmpdir"),ignore_errors=True)

        return RunData(metadata, stats, trace)
