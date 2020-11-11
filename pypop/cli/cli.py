#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""CLI Analysis scripts"""

from os import getcwd
from os.path import (
    exists as path_exists,
    expanduser,
    isabs,
    join as path_join,
    normpath,
    realpath,
    relpath,
)
from pkg_resources import resource_filename
from shutil import copytree, Error as shutil_error
from webbrowser import open_new_tab
import sys

from matplotlib import use

use("agg")

from pypop.traceset import TraceSet
from pypop.metrics import MPI_Metrics, MPI_OpenMP_Metrics, OpenMP_Metrics
from pypop.dimemas import dimemas_idealise
from pypop.config import set_dimemas_path, set_paramedir_path, set_tmpdir_path
from pypop.examples import examples_directory
from pypop.server import get_notebook_server_instance, construct_nb_url

from argparse import ArgumentParser
from tqdm import tqdm

import nbformat

latest_nbformat = getattr(nbformat, "v{}".format(nbformat.current_nbformat))
new_nb = latest_nbformat.new_notebook
code_cell = latest_nbformat.new_code_cell
md_cell = latest_nbformat.new_markdown_cell

GUI_MSG = """
        PyPOP GUI Ready:
        If the gui does not open automatically, please go to the following url
        in your web browser:

        {}
"""

OWN_SERVER_MSG = """
        A new notebook server was started for this PyPOP session.  When you are finished,
        press CTRL-C in this window to shut down the server.
"""


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


def _cli_metrics_parse_args(metric_name="MPI"):

    # make an argument parser
    parser = ArgumentParser(
        description="Calculate POP {} strong scaling metrics".format(metric_name)
    )

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
    parser.add_argument("--table-title", type=str, help="Title of metric table")
    parser.add_argument(
        "--table-key", type=str, default="auto", help="Key to use for table columns"
    )
    parser.add_argument(
        "--table-group",
        type=str,
        default="auto",
        help="Key to use for table column grouping",
    )

    parser.add_argument("--scaling-title", type=str, help="Title of scaling plot")
    parser.add_argument(
        "--scaling-key",
        type=str,
        default="auto",
        help="Key to use for independent variable in scaling plot",
    )

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
        "--tag", type=str, metavar="TAG", help="Tag to apply to trace(s)"
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

    config = _cli_metrics_parse_args("MPI")

    if config.paramedir_path:
        set_paramedir_path(config.paramedir_path)

    if config.dimemas_path:
        set_dimemas_path(config.dimemas_path)

    statistics = TraceSet(config.traces)

    metrics = MPI_Metrics(statistics)

    # Create and save table
    if not config.no_metric_table:
        metric_table = metrics.plot_table(
            title=config.table_title, columns_key=config.table_key
        )
        metric_table.save_png(config.metric_table)

    # Create and save scaling plot
    if not config.no_scaling_plot:
        scaling_plot = metrics.plot_scaling(
            title=config.scaling_title, x_key=config.scaling_key
        )
        scaling_plot.save_png(config.scaling_plot)

    # Save metrics as csv
    if not config.no_csv:
        metrics.metric_data.to_csv(config.csv, index=False)


def openmp_cli_metrics():
    """Entrypoint for pypop-hybrid-metrics script
    """

    config = _cli_metrics_parse_args("OpenMP")

    if config.paramedir_path:
        set_paramedir_path(config.paramedir_path)

    if config.dimemas_path:
        set_dimemas_path(config.dimemas_path)

    statistics = TraceSet(config.traces)

    metrics = OpenMP_Metrics(statistics)

    # Create and save table
    if not config.no_metric_table:
        metric_table = metrics.plot_table(
            title=config.table_title, columns_key=config.table_key
        )
        metric_table.save_png(config.metric_table)

    # Create and save scaling plot
    if not config.no_scaling_plot:
        scaling_plot = metrics.plot_scaling(
            title=config.scaling_title, x_key=config.scaling_key
        )
        scaling_plot.save_png(config.scaling_plot)

    # Save metrics as csv
    if not config.no_csv:
        metrics.metric_data.to_csv(config.csv, index=False)


def hybrid_cli_metrics():
    """Entrypoint for pypop-hybrid-metrics script
    """

    config = _cli_metrics_parse_args("hybrid MPI+OpenMP")

    if config.paramedir_path:
        set_paramedir_path(config.paramedir_path)

    if config.dimemas_path:
        set_dimemas_path(config.dimemas_path)

    statistics = TraceSet(config.traces)

    metrics = MPI_OpenMP_Metrics(statistics)

    print(config.metric_table)

    # Create and save table
    if not config.no_metric_table:
        metric_table = metrics.plot_table(
            title=config.table_title, columns_key=config.table_key
        )
        metric_table.save_png(config.metric_table)

    # Create and save scaling plot
    if not config.no_scaling_plot:
        scaling_plot = metrics.plot_scaling(
            title=config.scaling_title, x_key=config.scaling_key
        )
        scaling_plot.save_png(config.scaling_plot)

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
        tag=config.tag,
    )


def dimemas_idealise_cli():
    """Entrypoint for trace idealisation
    """

    config = _dimemas_idealise_parse_args()

    if config.dimemas_path:
        set_dimemas_path(config.dimemas_path)

    for tracefile in tqdm(config.traces, desc="Running Dimemas"):
        dimemas_idealise(tracefile, outpath=".")


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

    outpath = getcwd() if config.target_dir is None else expanduser(config.target_dir)

    outpath = normpath(path_join(outpath, "pypop_examples"))

    try:
        copytree(examples_directory(), outpath)
    except shutil_error as err:
        print("Copy failed: {}".format(str(err)))
        return -1


def _gui_launcher_parse_args():

    # make an argument parser
    parser = ArgumentParser(description="Launch the PyPOP GUI and Notebook server")

    # First define collection of traces
    parser.add_argument(
        "nb_path",
        type=str,
        nargs="?",
        metavar="NB_PATH",
        help="GUI Notebook name/path default is $PWD/pypop_gui.ipynb",
    )
    parser.add_argument(
        "-n",
        "--notebookdir",
        type=str,
        metavar="NB_DIR",
        help="Notebook server root directory (default is $PWD or inferred from "
        "NB_PATH), ignored if an existing server is used",
    )
    parser.add_argument(
        "-f",
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing file when creating GUI notebook.",
    )

    return parser.parse_args()


def _gui_exit(msg, own_server):
    print(msg, file=sys.stderr)
    if own_server:
        own_server.kill()
    sys.exit(-1)


def pypop_gui():
    """Entrypoint for launching Jupyter Notebook GUI
    """

    config = _gui_launcher_parse_args()

    notebookdir = getcwd()
    nb_name = "pypop_gui.ipynb"
    if config.notebookdir:
        notebookdir = realpath(notebookdir)

    nb_path = realpath(path_join(notebookdir, nb_name))
    if config.nb_path:
        if isabs(config.nb_path):
            if relpath(realpath(config.nb_path), notebookdir):
                gui_exit(
                    "Requested gui notebook file path is not in the notebook server "
                    "directory ({}).".format(config.nb_path, notebookdir),
                    None,
                )
            nb_path = os.realpath(config.nb_path)
        else:
            nb_name = os.realpath(join(notebookdir, config.nb_path))

    server_info, own_server = get_notebook_server_instance()

    real_nbdir = realpath(server_info["notebook_dir"])
    if relpath(nb_path, real_nbdir).startswith(".."):
        _gui_exit(
            "Requested gui notebook file path {} is not in the root of the "
            "notebook server ({}). You may need to specify a different working "
            "directory, change the server config, or allow PyPOP to start its own "
            "server.".format(nb_path, real_nbdir),
            own_server,
        )

    if not path_exists(nb_path) or config.force_overwrite:
        try:
            write_gui_nb(nb_path)
        except:
            _gui_exit("Failed to create gui notebook", own_server)

    nb_url = construct_nb_url(server_info, nb_path)

    open_new_tab(nb_url)

    print(GUI_MSG.format(nb_url))

    if own_server:
        print(OWN_SERVER_MSG)
        try:
            own_server.wait()
        except KeyboardInterrupt:
            own_server.terminate()


def hidden_code_cell(*args, **kwargs):

    hidden_cell = {"hide_input": True}

    if "metadata" in kwargs:
        kwargs["metadata"].update(hidden_cell)
    else:
        kwargs["metadata"] = hidden_cell

    return code_cell(*args, **kwargs)


def write_gui_nb(gui_nb_path):

    gui_nb = new_nb(metadata={})

    gui_code = """\
from pypop.notebook_interface import MetricsWizard
from pypop.metrics import MPI_OpenMP_Metrics
gui = MetricsWizard(MPI_OpenMP_Metrics)
display(gui)\
    """
    gui_cells = [(gui_code, hidden_code_cell)]

    for cell_text, cell_ctr in gui_cells:
        gui_nb.cells.append(cell_ctr(cell_text))

    nbformat.write(gui_nb, gui_nb_path)
