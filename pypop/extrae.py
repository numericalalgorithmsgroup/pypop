#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""\
Extrae Trace Utilities
----------------------

Routines for analysis and management of Extrae traces, including Paramedir automation.
"""

import re
import os
from os.path import basename, dirname, normpath, splitext
from tempfile import mkdtemp, mkstemp
import subprocess as sp
import warnings

from pkg_resources import resource_filename

import pandas
import numpy

from .prv import PRV
from . import config

from pypop.utils.exceptions import ExtraePRVNoOnOffEventsError

floatmatch = re.compile(r"[0-9,]+\.[0-9]+")
keymatch = re.compile(r"@.*@")
tabmatch = re.compile(r"\t+")

ROI_FILTER_XML = "filters/tracing_state.xml"
CUTTER_SINGLE_SKEL = "cutters/single_region.skel"


def remove_trace(tracefile, failure_is_error=False):
    """Remove prv or dim file and its associated .row and .pcf files

    The function should be passed a filename ending in .dim or .prv and will
    attempt to delete this and the pcf and row file.  The filenames of these
    will be constructed by direct substitution of the file extension.

    Parameters
    ----------
    tracefile: str
        Path to tracefile, should end with .dim or prv
    """

    if not (tracefile.endswith(".prv") or tracefile.endswith(".dim")):
        raise ValueError("Expected a .prv or .dim file")

    try:
        os.remove(tracefile)
    except FileNotFoundError as err:
        if failure_is_error:
            raise err

    for ext in [".pcf", ".row"]:
        try:
            os.remove(splitext(tracefile)[0] + ext)
        except FileNotFoundError as err:
            if failure_is_error:
                raise err


def chop_prv_to_roi(prv_file, outfile=None):
    """Cut down a prv trace to just the region of interest

    This will cut a trace to just include the region of interest, which is
    assumed to be bracketed by calls to `Extrae_Restart()` and
    `Extrae_Shutdown()` which are located by their corresponding events
    (40000012) in the trace.

    Parameters
    ----------
    prv_file: str
        Trace file to be cut
    outfile: str or None
        Optional output file for chopped trace. (If not specified will be
        created in a temporary folder.)

    Returns
    -------
    chopped: str
        Path to chopped tracefile in prv format.
    """

    if outfile:
        workdir = dirname(normpath(outfile))
    else:
        if prv_file.endswith(".gz"):
            tgtname = ".chop".join(splitext(basename(prv_file[:-3])))
        else:
            tgtname = ".chop".join(splitext(basename(prv_file)))

        # Make sure config._tmpdir_path exists before using it
        if config._tmpdir_path:
            os.makedirs(config._tmpdir_path, exist_ok=True)
        workdir = mkdtemp(dir=config._tmpdir_path)
        outfile = os.path.join(workdir, tgtname)

    roi_filter = resource_filename(__name__, ROI_FILTER_XML)
    roi_prv = os.path.join(workdir, ".roifilter".join(splitext(basename(prv_file))))

    paramedir_binpath = "paramedir"
    if config._paramedir_path:
        paramedir_binpath = os.path.join(config._paramedir_path, paramedir_binpath)

    filter_cmds = [
        paramedir_binpath,
        "--filter",
        roi_filter,
        "--output-name",
        roi_prv,
        prv_file,
    ]

    result = sp.run(filter_cmds, stdout=sp.PIPE, stderr=sp.STDOUT)

    if result.returncode != 0 or not os.path.exists(roi_prv):
        raise RuntimeError(
            "Failed to filter ROI file:\n{}" "".format(result.stdout.decode())
        )

    try:
        starttime, endtime = _get_roi_times(roi_prv)
    except ValueError as err:
        raise ValueError("Error cutting trace to ROI: {}".format(str(err)))

    remove_trace(roi_prv)

    wfh, cutter_xml = mkstemp(dir=workdir, suffix=".xml", text=True)
    os.close(wfh)

    cutter_skel = resource_filename(__name__, CUTTER_SINGLE_SKEL)

    with open(cutter_skel, "rt") as rfh, open(cutter_xml, "wt") as wfh:
        for line in rfh:
            line = line.replace("@MIN_TIME@", "{:d}".format(starttime))
            line = line.replace("@MAX_TIME@", "{:d}".format(endtime))
            wfh.write(line)

    wfh.close()

    paramedir_binpath = "paramedir"
    if config._paramedir_path:
        paramedir_binpath = os.path.join(config._paramedir_path, paramedir_binpath)

    cutter_cmds = [
        paramedir_binpath,
        "--cutter",
        prv_file,
        cutter_xml,
        "--output-name",
        outfile,
    ]

    result = sp.run(cutter_cmds, stdout=sp.PIPE, stderr=sp.PIPE)
    os.remove(cutter_xml)

    if result.returncode != 0 or not os.path.exists(outfile):
        raise RuntimeError(
            "Failed to cut prv file {}:\n{}" "".format(outfile, result.stdout.decode())
        )

    return outfile


def _get_roi_times(roi_prv):
    """ Extract ROI timing information from a filtered trace

    Expects a trace containing only Extrae On/Off events and returns tuple of
    earliest and latest time
    """
    # Get dataframe of events from filtered trace
    data = PRV(roi_prv)
    df = data.event

    if df is None or df.empty:
        raise ExtraePRVNoOnOffEventsError("No valid Extrae ON-OFF bracket in trace")

    # Get the first on and last off events
    grouped = df.reset_index(level="time").groupby(level=["task", "thread"])
    ons = grouped.nth(1)
    offs = grouped.last()

    # Check the events have the expected values
    if not (ons["value"] == 1).all():
        raise ValueError("Unexpected event value: expected 40000012:1")
    if not (offs["value"] == 0).all():
        raise ValueError("Unexpected event value: expected 40000012:0")
    ontime, offtime = (ons["time"].min(), 1 + offs["time"].max())

    if ontime is numpy.nan or offtime is numpy.nan:
        raise ExtraePRVNoOnOffEventsError("No valid Extrae ON-OFF bracket in trace")

    return ontime, offtime


def paramedir_analyze(
    tracefile,
    paramedir_config,
    variables=None,
    index_by_thread=False,
    statistic_names=None,
):
    """Analyze a tracefile with paramedir

    Parameters
    ----------
    tracefile: str
        Path to `*.prv` tracefile from Extrae
    paramedir_config: str
        Path to Paraver/Paramedir `*.cfg`
    variables: dict or None
        Optional dict of key-value pairs for replacement in config file prior
        to running paramedir.
    index_by_thread: bool
        If True return data organised by a multilevel index of MPI ranks and
        threads.  Note that this discards Paramedir calculated statistical
        info.
    statistic_names: list of str or None
        Optional list of string names for the statistics returned by the config
        file.  If not provided names will be taken from paramedir output.

    Returns
    -------
    result: pandas.DataFrame
        Result data loaded from the resulting csv.
    """

    with open(paramedir_config, "r") as fh:
        confstring = " ".join(fh)

    if "Analyzer2D.3D" in confstring:
        datatype = "Hist3D"
    elif "Analyzer2D" in confstring:
        datatype = "Hist2D"
        return _analyze_hist2D(
            tracefile, paramedir_config, variables, index_by_thread, statistic_names
        )
    else:
        datatype = "Raw counts"

    raise ValueError('Unsupported analysis type "{}"'.format(datatype))


def paramedir_analyze_any_of(
    tracefile,
    paramedir_configs,
    variables=None,
    index_by_thread=False,
    statistic_names=None,
):
    """Analyze a tracefile with paramedir, returning first success from an interable of
    possible config files.

    This routine is intended to allow elegant use of fallback to simpler configs to work
    around a paramedir bug where all event types must be present to avoid a fatal error
    in analysis. (see https://github.com/bsc-performance-tools/paraver-kernel/issues/5)

    Parameters
    ----------
    tracefile: str
        Path to `*.prv` tracefile from Extrae
    paramedir_configs: iterable or str
        Path to Paraver/Paramedir `*.cfg` files.
    variables: dict or None
        Optional dict of key-value pairs for replacement in config file prior
        to running paramedir.
    index_by_thread: bool
        If True return data organised by a multilevel index of MPI ranks and
        threads.  Note that this discards Paramedir calculated statistical
        info.
    statistic_names: list of str or None
        Optional list of string names for the statistics returned by the config
        file.  If not provided names will be taken from paramedir output.

    Returns
    -------
    result: pandas.DataFrame
        Result data loaded from the resulting csv.
    """

    if isinstance(paramedir_configs, str):
        paramedir_configs = [paramedir_configs]

    for paramedir_config in paramedir_configs:
        try:
            return paramedir_analyze(
                tracefile, paramedir_config, variables, index_by_thread, statistic_names
            )
        except RuntimeError as err:
            error = err

    raise error


def _analyze_hist2D(tracefile, paramedir_config, variables, index_by_thread, stat_names):
    """Run a paramedir config producing a 2D histogram and return result DataFrame
    """

    histfile = run_paramedir(tracefile, paramedir_config, variables=variables)

    data = load_paraver_histdata(histfile)

    os.remove(histfile)

    if stat_names:
        data.index = pandas.Index(stat_names)

    if index_by_thread:
        return reindex_by_thread(data)

    return data


def reindex_by_thread(stats_dframe, thread_prefix="THREAD"):
    """Convert stats Dataframe index in-place to a rank,thread MultiIndex

    Parameters
    ----------
    stats_dframe: pandas.DataFrame
        Dataframe to reindex. Typically this will have been produced using
        paramedir_analyze().

    thread_prefix: str
        Prefix before thread number pattern in current index.  Should almost
        always be "THREAD". Paraver/Paramedir default is "THREAD a.r.t" with r
        the rank number and t the thread number.
    """

    if not isinstance(stats_dframe, pandas.DataFrame):
        raise TypeError("stats_dframe must be a Pandas DataFrame")

    oc_select = [c for c in stats_dframe.columns if c.startswith(thread_prefix)]
    newcols = pandas.MultiIndex.from_tuples(
        [tuple(int(x) for x in y.split(".")[1:]) for y in oc_select]
    )
    stats_dframe = stats_dframe[oc_select].set_axis(
        newcols, axis="columns", inplace=False
    )
    stats_dframe.columns.rename(["rank", "thread"], inplace=True)

    return stats_dframe


def run_paramedir(tracefile, paramedir_config, outfile=None, variables=None):
    """Run paramedir on a tracefile

    Parameters
    ----------
    tracefile: str
        Path to `*.prv` tracefile from Extrae
    paramedir_config: str
        Path to Paraver/Paramedir `*.cfg`
    outfile: str or None
        Path to output file. If None or "" a randomly named temporary file will
        be used.
    variables: dict of str:str
        Dict of variables to replace in the config file.  For a key-value pair
        "key":val any occurrence of @key@ in the file will be replaced with
        "val"

    Returns
    -------
    outfile: str
        Path to the output file.
    """

    # Make sure config._tmpdir_path exists before using it
    if config._tmpdir_path:
        os.makedirs(config._tmpdir_path, exist_ok=True)
        tmpdir = mkdtemp(dir=config._tmpdir_path)
    else:
        tmpdir = mkdtemp()

    # If variables is none, still sub with empty dict
    variables = variables if variables else {}
    tmp_config = _write_substituted_config(paramedir_config, tmpdir, variables)

    if not outfile:
        outfile = os.path.join(
            tmpdir, os.path.splitext(os.path.basename(paramedir_config))[0]
        )

    paramedir_binpath = "paramedir"
    if config._paramedir_path:
        paramedir_binpath = os.path.join(config._paramedir_path, paramedir_binpath)

    paramedir_params = [paramedir_binpath, tracefile, tmp_config, outfile]

    result = sp.run(paramedir_params, stdout=sp.PIPE, stderr=sp.STDOUT)

    if not os.path.exists(outfile) or result.returncode != 0:
        raise RuntimeError(
            "Paramedir execution failed:\n{}" "".format(result.stdout.decode())
        )

    return outfile


def _write_substituted_config(template_config, tmpdir, variables):
    """Copy config to tempfile, substituting placeholders from variables dict
    """
    newconfig = os.path.join(tmpdir, os.path.basename(template_config))

    with open(newconfig, "w") as newfh, open(template_config, "r") as oldfh:
        for line in oldfh:
            newline = line
            for match in keymatch.findall(line):
                if match[1:-1] in variables:
                    newline = newline.replace(match, variables[match[1:-1]])
                else:
                    raise ValueError(
                        "Unhandled key {} in {}" "".format(match, template_config)
                    )
            newfh.write(newline)

    return newconfig


def _split_binline(binline):
    """Internal function to read the line of bin specs

    Returns bin width and array of bin lower edges
    """
    bin_strings = tabmatch.split(binline.strip())

    # Grab the first float from each binspec, we'll return lower edges
    # Note that commas must be stripped from numbers...
    try:
        bins = numpy.fromiter(
            (floatmatch.findall(x)[0].replace(",", "") for x in bin_strings),
            dtype=numpy.float64,
        )
    except IndexError:
        bins = numpy.asarray(bin_strings)

    return bins


def _split_countline(countline, bins):
    """Internal function to read the line of counts

    Returns count name and array of counts
    """
    num_values = len(bins)
    count_strings = tabmatch.split(countline.strip())

    # Must be at least one more string than values to be the label
    if len(count_strings) <= num_values:
        raise ValueError("Malformed count line")

    extra_strings = len(count_strings) - num_values

    if extra_strings > 1:
        warnings.warn(
            "Warning, got more label strings ({}) than expected (1)"
            "".format(extra_strings)
        )

    # However many extra strings there are, join them to make the name
    count_name = " ".join(count_strings[0:extra_strings])

    counts = numpy.asarray(count_strings[extra_strings:], dtype=numpy.float64)

    return (count_name, pandas.Series(counts, index=bins))


def load_paraver_histdata(hist_file):
    """Read a Paraver histogram file and return pandas dataframe containing the
    data.

    Parameters
    ----------
    hist_file : str
        Path to the histogram data file
    """

    data_dict = {}
    with open(hist_file, "r") as fh:
        # First line should be bins
        bins = _split_binline(fh.readline())

        # Now process count lines
        for count_line in fh:
            if count_line.strip():
                name, data_dict[name] = _split_countline(count_line, bins)

    return pandas.DataFrame.from_dict(data_dict)


def is_extrae_tracefile(tracefile):
    if tracefile.endswith(".prv.gz") or tracefile.endswith(".prv"):
        return True
    return False
