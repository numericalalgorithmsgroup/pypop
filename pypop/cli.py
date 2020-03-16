#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""CLI Analysis scripts"""

from os import getcwd
from os.path import expanduser, join as path_join, normpath
from shutil import copytree, Error as shutil_error

from matplotlib import use

use("agg")

from .traceset import TraceSet
from .metrics import MPI_Metrics, MPI_OpenMP_Metrics, OpenMP_Metrics
from .dimemas import dimemas_idealise
from .config import set_dimemas_path, set_paramedir_path, set_tmpdir_path
from .examples import examples_directory

from argparse import ArgumentParser


def _dimemas_idealise_parse_args():

    # make an argument parser
    parser = ArgumentParser(description="Idealise Extrae traces with Dimemas")

    # First define collection of traces
    parser.add_argument(
        "traces", type=str, nargs="+", metavar="trace", help="tracefiles to idealise"
    )

    parser.add_argument(
        "--dimemas-path", type=str, metavar="PATH", help="Path to Dimemas executable"
    )

    return parser.parse_args()


def _mpi_parse_args():

    # make an argument parser
    parser = ArgumentParser(description="Calculate POP MPI strong scaling metrics")

    # First define collection of traces
    parser.add_argument(
        "traces", type=str, nargs="+", metavar="trace", help="tracefiles to analyze"
    )

    # Output disable
    parser.add_argument("--no-csv", action="store_true", help="Don't save metrics csv")
    parser.add_argument(
        "--no-metric-table", action="store_true", help="Don't save table plot"
    )
    parser.add_argument(
        "--no-scaling-plot", action="store_true", help="Don't save scaling plot"
    )

    parser.add_argument(
        "--paramedir-path", type=str, metavar="PATH", help="Path to Paramedir executable"
    )
    parser.add_argument(
        "--dimemas-path", type=str, metavar="PATH", help="Path to Dimemas executable"
    )

    # Output locations
    parser.add_argument(
        "--csv", "-c", type=str, default="metrics.csv", help="path to save metric csv"
    )
    parser.add_argument(
        "--metric-table",
        "-m",
        type=str,
        default="table.png",
        help="path to save table plot",
    )
    parser.add_argument(
        "--scaling-plot",
        "-s",
        type=str,
        default="scaling.png",
        help="path to save scaling plot",
    )

    # Output Customisation
    parser.add_argument("--metric-title", type=str, help="Title of metric table")
    parser.add_argument("--scaling-title", type=str, help="Title of scaling plot")

    return parser.parse_args()


def _preprocess_traces_parse_args():

    # make an argument parser
    parser = ArgumentParser(description="Preprocess traces for PyPOP Analysis")

    # First define collection of traces
    parser.add_argument(
        "traces", type=str, nargs="+", metavar="trace", help="Tracefiles to preprocess"
    )

    parser.add_argument(
        "--force-recalculation",
        action="store_true",
        help="Force recalculation & overwrite existing data",
    )
    parser.add_argument(
        "--chop-to-roi", action="store_true", help="Chop to region of interest"
    )
    parser.add_argument(
        "--paramedir-path", type=str, metavar="PATH", help="Path to Paramedir executable"
    )
    parser.add_argument(
        "--dimemas-path", type=str, metavar="PATH", help="Path to Dimemas executable"
    )
    parser.add_argument(
        "--outfile-path",
        type=str,
        metavar="PATH",
        help="Path in which to save chopped/ideal traces",
    )
    parser.add_argument(
        "--tmpdir-path",
        type=str,
        metavar="PATH",
        help="Path for PyPOP to save temporary files",
    )

    return parser.parse_args()


def mpi_cli_metrics():
    """Entrypoint for pypop-mpi-metrics script
    """

    config = _mpi_parse_args()

    if config.paramedir_path:
        set_paramedir_path(config.paramedir_path)

    if config.dimemas_path:
        set_dimemas_path(config.dimemas_path)

    statistics = TraceSet(config.traces)

    metrics = MPI_Metrics(statistics)

    # Create and save table
    if not config.no_metric_table:
        metric_table = metrics.plot_table(title=config.metric_title)
        metric_table.savefig(config.metric_table)

    # Create and save scaling plot
    if not config.no_scaling_plot:
        scaling_plot = metrics.plot_scaling(title=config.scaling_title)
        scaling_plot.savefig(config.scaling_plot)

    # Save metrics as csv
    if not config.no_csv:
        metrics.metric_data.to_csv(config.csv, index=False)


def openmp_cli_metrics():
    """Entrypoint for pypop-hybrid-metrics script
    """

    config = _mpi_parse_args()

    if config.paramedir_path:
        set_paramedir_path(config.paramedir_path)

    if config.dimemas_path:
        set_dimemas_path(config.dimemas_path)

    statistics = TraceSet(config.traces)

    metrics = OpenMP_Metrics(statistics)

    # Create and save table
    if not config.no_metric_table:
        metric_table = metrics.plot_table(title=config.metric_title)
        metric_table.savefig(config.metric_table)

    # Create and save scaling plot
    if not config.no_scaling_plot:
        scaling_plot = metrics.plot_scaling(title=config.scaling_title)
        scaling_plot.savefig(config.scaling_plot)

    # Save metrics as csv
    if not config.no_csv:
        metrics.metric_data.to_csv(config.csv, index=False)


def hybrid_cli_metrics():
    """Entrypoint for pypop-hybrid-metrics script
    """

    config = _mpi_parse_args()

    if config.paramedir_path:
        set_paramedir_path(config.paramedir_path)

    if config.dimemas_path:
        set_dimemas_path(config.dimemas_path)

    statistics = TraceSet(config.traces)

    metrics = MPI_OpenMP_Metrics(statistics)

    # Create and save table
    if not config.no_metric_table:
        metric_table = metrics.plot_table(title=config.metric_title)
        metric_table.savefig(config.metric_table)

    # Create and save scaling plot
    if not config.no_scaling_plot:
        scaling_plot = metrics.plot_scaling(title=config.scaling_title)
        scaling_plot.savefig(config.scaling_plot)

    # Save metrics as csv
    if not config.no_csv:
        metrics.metric_data.to_csv(config.csv, index=False)


def preprocess_traces():
    """Entrypoint for trace preprocessing
    """

    config = _preprocess_traces_parse_args()

    if config.paramedir_path:
        set_paramedir_path(config.paramedir_path)

    if config.dimemas_path:
        set_dimemas_path(config.dimemas_path)

    if config.tmpdir_path:
        set_tmpdir_path(config.tmpdir_path)

    TraceSet(
        config.traces,
        force_recalculation=config.force_recalculation,
        chop_to_roi=config.chop_to_roi,
        outpath=config.outfile_path,
    )


def dimemas_idealise():
    """Entrypoint for trace idealisation
    """

    config = _dimemas_idealise_parse_args()

    if config.dimemas_path:
        set_dimemas_path(config.dimemas_path)

    for tracefile in tqdm(config.traces, desc="Running Dimemas"):
        outfile = tracefile.split(".prv")[0] + ".sim.prv"
        dimemas_idealise(tracefile, outfile)


def _copy_examples_parse_args():

    # make an argument parser
    parser = ArgumentParser(description="Copy PyPOP example files to a user directory")

    # First define collection of traces
    parser.add_argument(
        "target_dir",
        type=str,
        nargs="?",
        metavar="dir",
        help="Target directory (default is current working dir)",
    )

    return parser.parse_args()


def copy_examples():
    """Entrypoint for example copy routine
    """

    config = _copy_examples_parse_args()

    outpath = (
        getcwd() if config.target_dir is None else expanduser(config.target_dir)
    )

    outpath = normpath(path_join(outpath, 'pypop_examples'))

    try:
        copytree(examples_directory(), outpath)
    except shutil_error as err:
        print("Copy failed: {}".format(str(err)))
        return -1
