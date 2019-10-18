#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Routines for calculating and displaying the POP metrics.
"""

from pkg_resources import resource_filename

import numpy
import pandas

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib as mpl
import matplotlib.table as mt
import matplotlib.ticker as mtick

from .plotting import pypop_mpl_params, figparams
from .traceset import RunData

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


class Metric:
    """
    freq_corr: bool
        Correct performance metrics based on average clock frequency (use to
        correct for node dynamic clocking issues).
    """

    def __init__(self, key, level, displayname=None):
        self.key = key
        self.level = level

        if displayname:
            self.displayname = r"$\hookrightarrow$" * bool(self.level) + displayname
        else:
            self.displayname = r"$\hookrightarrow$" * bool(self.level) + self.key


class MetricSet:
    """Calculate POP MPI metrics for the given statistics

    Statistics data is expected to have been produced with `collect_statistics()`

    Parameters
    ----------
    stats_dict: dict or list of `pd.DataFrame`
        Statistics as collected with `collect_statistics()`. Dictionary keys will be
        used as the dataframe index. If a list, a dict will be constructed by
        enumeration.

    ref_key: scalar
        Key of stats_dict that should be used as the reference for calculation of
        scaling values.  If not specified, the lexical minimum key will be used (i.e
        ``min(stats_dict.keys())``.

    sort_keys: bool
        If true (default), lexically sort the keys in the returned DataFrame.
    """

    def __init__(self, stats_dict):

        if not isinstance(stats_dict, dict):
            stats_dict = {k: v for k, v in enumerate(stats_dict)}

        for df in stats_dict.values():
            if not isinstance(df, RunData):
                raise ValueError(
                    "stats_dict must be an iterable of " "pypop.traceset.RunData"
                )

        self._stats_dict = stats_dict
        self._metric_data = None

    def _calculate_metrics(self):
        raise NotImplementedError

    def _repr_html_(self):
        return self.metric_data._repr_html_()

    @property
    def metric_data(self):
        if self._metric_data is None:
            self._calculate_metrics()
        return self._metric_data

    @property
    def metrics(self):
        raise NotImplementedError

    def plot_table(
        self,
        columns_key=None,
        title=None,
        columns_label=None,
        good_thres=0.8,
        bad_thres=0.5,
        skipfirst=0,
        bwfirst=0,
    ):
        """Plot Metrics in colour coded Table

        Parameters
        ----------
        columns_key: str or None
            Key to pandas dataframe column containing column heading data (default
            "Number of Processes").
        title: str or None
            Title for table.
        columns_label: str or None
            Label to apply to column heading data (defaults to value of
            `columns_key`).
        good_thres: float [0.0 - 1.0]
            Threshold above which cells are shaded green.
        bad_thres: float [0.0 - 1.0]
            Threshold below which cells are shaded red.
        skipfirst: int
            Skip output of first N columns of metric data (default 0).
        bwfirst: int
            Skip coloring of first N columns of metric data (default 0).

        Returns
        -------
        figure: `matplotlib.figure.Figure`
            Figure containing the metrics table.
        """

        with mpl.rc_context(pypop_mpl_params):
            return self._plot_table(
                columns_key,
                title,
                columns_label,
                good_thres,
                bad_thres,
                skipfirst,
                bwfirst,
            )

    def _plot_table(
        self,
        columns_key,
        title,
        columns_label,
        good_thres,
        bad_thres,
        skipfirst,
        bwfirst,
    ):

        if not columns_key:
            columns_key = "Number of Processes"

        if not columns_label:
            columns_label = columns_key

        body_cell_width = 0.08
        body_cell_height = 0.1
        label_cell_width = 0.45
        level_pad = 0.075

        cmap_points = [
            (0.0, (0.690, 0.074, 0.074)),
            (bad_thres, (0.690, 0.074, 0.074)),
            (good_thres - 1e-20, (0.992, 0.910, 0.910)),
            (good_thres, (0.910, 0.992, 0.910)),
            (1.0, (0.074, 0.690, 0.074)),
        ]

        metric_cmap = mc.LinearSegmentedColormap.from_list(
            "POP_Metrics", colors=cmap_points, N=256, gamma=1
        )

        label_cell_kwargs = {
            "loc": "left",
            "width": label_cell_width,
            "height": body_cell_height,
        }
        body_cell_kwargs = {
            "loc": "center",
            "width": body_cell_width,
            "height": body_cell_height,
        }

        fig = plt.figure(figsize=figparams["single.figsize"])
        ax = [fig.add_axes(fp) for fp in figparams["single.axlayout"]]
        ax[0].set_axis_off()

        # Create empty table using full bounding box of axes
        metric_table = mt.Table(ax=ax[0], bbox=(0, 0, 1, 1))
        metric_table.auto_set_font_size(False)

        metric_table.add_cell(
            0,
            0,
            width=label_cell_width,
            height=body_cell_height,
            text=columns_label,
            loc="center",
        )

        for col_num, col_data in enumerate(self.metric_data[columns_key], start=1):
            if col_num <= skipfirst:
                continue
            metric_table.add_cell(
                0,
                col_num - skipfirst,
                text="{:d}".format(int(col_data)),
                **body_cell_kwargs
            )

        for row_num, metric in enumerate(self.metrics, start=1):
            c = metric_table.add_cell(
                row_num, 0, text=metric.displayname, **label_cell_kwargs
            )
            c.PAD = 0.05
            c.PAD += 0 if metric.level <= 1 else level_pad * (metric.level - 1)

            for col_num, col_data in enumerate(self.metric_data[metric.key], start=1):
                if col_num <= skipfirst:
                    continue
                if col_num <= bwfirst:
                    metric_table.add_cell(
                        row_num,
                        col_num,
                        text="{:1.02f}".format(col_data),
                        **body_cell_kwargs
                    )
                else:
                    metric_table.add_cell(
                        row_num,
                        col_num - skipfirst,
                        text="{:1.02f}".format(col_data),
                        facecolor=metric_cmap(col_data),
                        **body_cell_kwargs
                    )

        metric_table.set_fontsize(9)
        ax[0].add_table(metric_table)

        if title:
            ax[0].set_title(title)

        return fig

    def plot_scaling(
        self, x_key="Number of Processes", y_key="Speedup", label=None, title=None
    ):
        """Plot scaling graph with region shading.

        Plots scaling data from pandas dataframe(s). The 0-80% and 80-100% scaling
        regions are shaded for visual identification.  Multiple scaling lines may be
        plotted by passing a dict of dataframes.

        Parameters
        ----------
        data: `pd.DataFrame` or dict of DataFrames
            (Dict of) Pandas DataFrame containing scaling data.

        x_key: scalar
            Key of Dataframe column to use as x-axis.

        y_key: scalar
            key of Dataframe column to use as y-axis.

        label: str or None
            Label to be used for y-axis and data series. Defaults to `y_key`.

        title: str or None
            Optional title for plot.

        Returns
        -------
        figure: matplotlib.figure.Figure
            Figure containing complete scaling plot.
        """
        with mpl.rc_context(pypop_mpl_params):
            return self._plot_scaling(x_key, y_key, label, title)

    def _plot_scaling(self, x_key, y_key, label, title):

        if label is None:
            label = str(y_key)

        cores_min = numpy.nan
        cores_max = numpy.nan
        y_max = numpy.nan

        cores_min = numpy.nanmin([cores_min, self._metric_data[x_key].min()])
        cores_max = numpy.nanmax([cores_max, self._metric_data[x_key].max()])
        y_max = numpy.nanmax((y_max, self._metric_data[y_key].max()))

        y_max *= 1.2

        x_margin = 0.02*cores_max
        ideal_scaling_cores = numpy.linspace(cores_min-x_margin, cores_max+x_margin)
        ideal_scaling = ideal_scaling_cores / cores_min
        ideal_scaling_80pc = 0.2 + 0.8 * ideal_scaling

        fig = plt.figure(figsize=figparams["single.figsize"])
        ax = fig.add_axes(figparams["single.axlayout"][0])

        ax.fill_between(
            ideal_scaling_cores,
            0,
            ideal_scaling_80pc,
            label="80% Scaling",
            alpha=0.1,
            color="g",
            linestyle="-",
        )
        ax.fill_between(
            ideal_scaling_cores,
            ideal_scaling_80pc,
            ideal_scaling,
            label="Ideal Scaling",
            alpha=0.2,
            color="g",
            linestyle="-",
        )

        ax.plot(
            self._metric_data[x_key],
            self._metric_data[y_key],
            label=label,
            marker="x",
            linestyle="-",
            alpha=0.8,
        )

        ax.set_xlim(ideal_scaling_cores.min(), ideal_scaling_cores.max())
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Total cores")
        ax.set_ylabel(label)
        ax.xaxis.set_major_locator(mtick.FixedLocator(self._metric_data[x_key], 6))
        ax.legend()

        if title:
            ax.set_title(title)

        return fig


class MPI_Metrics(MetricSet):
    @property
    def metrics(self):
        return [
            Metric("Global Efficiency", 0),
            Metric("Parallel Efficiency", 1),
            Metric("MPI Load balance", 2, "Load balance"),
            Metric("MPI Communication Efficiency", 2),
            Metric("MPI Transfer Efficiency", 3),
            Metric("MPI Serialisation Efficiency", 3),
            Metric("Computational Scaling", 1),
            Metric("Instruction Scaling", 2),
            Metric("IPC Scaling", 2),
            Metric("Frequency Scaling", 2),
        ]

    def _calculate_metrics(self, ref_key=None, sort_keys=True):

        if not ref_key:
            ref_key = min(self._stats_dict.keys())

        metrics_by_key = {}

        if sort_keys:
            keys = sorted(self._stats_dict.keys())
        else:
            key = self._stats_dict.keys()

        for key in keys:
            traceinfo = self._stats_dict[key].traceinfo
            stats = self._stats_dict[key].stats
            metrics = {"Number of Processes": sum(traceinfo.procs_per_node)}

            metrics["MPI Communication Efficiency"] = (
                stats["Total Non-MPI Runtime"].loc[:, 1].max()
                / stats["Total Runtime"].max()
            )

            try:
                metrics["MPI Serialisation Efficiency"] = (
                    stats["Total Non-MPI Runtime"].loc[:, 1].max()
                    / stats["Ideal Runtime"].loc[:, 1].max()
                )
            except KeyError:
                metrics["MPI Serialisation Efficiency"] = numpy.nan

            try:
                metrics["MPI Transfer Efficiency"] = (
                    stats["Ideal Runtime"].loc[:, 1].max()
                    / stats["Total Runtime"].loc[:, 1].max()
                )
            except KeyError:
                metrics["MPI Transfer Efficiency"] = numpy.nan

            metrics["MPI Load balance"] = 1 - (
                (
                    stats["Total Useful Computation"].loc[:, 1].max()
                    - stats["Total Useful Computation"].loc[:, 1].mean()
                )
                / stats["Total Runtime"].max()
            )

            metrics["Parallel Efficiency"] = (
                stats["Total Useful Computation"].mean()
                / stats["Total Runtime"].max()  # avg all threads to include Amdahl
            )

            metrics["IPC Scaling"] = (
                stats["IPC"].mean() / self._stats_dict[ref_key].stats["IPC"].mean()
            )

            metrics["Instruction Scaling"] = (
                self._stats_dict[ref_key].stats["Useful Instructions"].sum()
                / stats["Useful Instructions"].sum()
            )

            metrics["Frequency Scaling"] = (
                stats["Frequency"].mean()
                / self._stats_dict[ref_key].stats["Frequency"].mean()
            )

            metrics["Computational Scaling"] = (
                self._stats_dict[ref_key].stats["Total Useful Computation"].sum()
                / stats["Total Useful Computation"].sum()
            )

            metrics["Global Efficiency"] = (
                metrics["Computational Scaling"] * metrics["Parallel Efficiency"]
            )

            metrics["Speedup"] = (
                self._stats_dict[ref_key].stats["Total Runtime"].max()
                / stats["Total Runtime"].max()
            )

            metrics["Runtime"] = stats["Total Runtime"].max()

            metrics_by_key[key] = metrics

        self._metric_data = pandas.DataFrame(metrics_by_key).T


class MPI_OpenMP_Metrics(MetricSet):
    @property
    def metrics(self):
        return [
            Metric("Global Efficiency", 0),
            Metric("Parallel Efficiency", 1),
            Metric("Process Level Efficiency", 2),
            Metric("MPI Load balance", 3, "Load balance"),
            Metric("MPI Communication Efficiency", 3),
            Metric("MPI Transfer Efficiency", 4),
            Metric("MPI Serialisation Efficiency", 4),
            Metric("Thread Level Efficiency", 2),
            Metric("OpenMP Region Efficiency", 3, "OpenMP Region Efficiency"),
            Metric("Serial Region Efficiency", 3),
            Metric("Computational Scaling", 1),
            Metric("Instruction Scaling", 2),
            Metric("IPC Scaling", 2, "IPC Scaling"),
        ]

    def _calculate_metrics(self, ref_key=None, sort_keys=True):
        if not ref_key:
            ref_key = min(self._stats_dict.keys())

        metrics_by_key = {}

        if sort_keys:
            keys = sorted(self._stats_dict.keys())
        else:
            key = self._stats_dict.keys()

        for key in keys:
            traceinfo = self._stats_dict[key].traceinfo
            stats = self._stats_dict[key].stats
            try:
                nthreads = traceinfo.application_layout.rank_threads[0][0]
                metrics = {"Number of Processes": sum(traceinfo.procs_per_node)}

                metrics["OpenMP Region Efficiency"] = 1 - (
                    (
                        (
                            stats["OpenMP Total Runtime"].loc[:, 1]
                            - stats["OpenMP Useful Computation"].mean(level="thread")
                        ).mean()
                    )
                    / stats["Total Runtime"].max()
                )

                metrics["Serial Region Efficiency"] = 1 - (
                    stats["Serial Useful Computation"].loc[:, 1].mean()
                    / stats["Total Runtime"].max()
                    * (1 - 1 / nthreads)
                )

                metrics["Thread Level Efficiency"] = 1 - (
                    (
                        stats["OpenMP Total Runtime"].loc[:, 1].mean()
                        - stats["OpenMP Useful Computation"].mean()
                        + stats["Serial Useful Computation"].loc[:, 1].mean()
                        * (1 - 1 / nthreads)
                    )
                    / stats["Total Runtime"].max()
                )

                metrics["MPI Communication Efficiency"] = (
                    stats["Total Non-MPI Runtime"].loc[:, 1].max()
                    / stats["Total Runtime"].max()
                )

                try:
                    metrics["MPI Serialisation Efficiency"] = (
                        stats["Total Non-MPI Runtime"].loc[:, 1].max()
                        / stats["Ideal Runtime"].loc[:, 1].max()
                    )
                except KeyError:
                    metrics["MPI Serialisation Efficiency"] = numpy.nan

                try:
                    metrics["MPI Transfer Efficiency"] = (
                        stats["Ideal Runtime"].loc[:, 1].max()
                        / stats["Total Runtime"].loc[:, 1].max()
                    )
                except KeyError:
                    metrics["MPI Transfer Efficiency"] = numpy.nan

                metrics["MPI Load balance"] = 1 - (
                    (
                        stats["Total Non-MPI Runtime"].loc[:, 1].max()
                        - stats["Total Non-MPI Runtime"].loc[:, 1].mean()
                    )
                    / stats["Total Runtime"].max()
                )

                metrics["Process Level Efficiency"] = (
                    stats["Total Non-MPI Runtime"].loc[:, 1].mean()
                ) / stats["Total Runtime"].max()

                metrics["Parallel Efficiency"] = (
                    stats["Total Useful Computation"].mean()
                    / stats["Total Runtime"].max()  # avg all threads to include Amdahl
                )

                metrics["IPC Scaling"] = (
                    stats["IPC"].mean() / self._stats_dict[ref_key].stats["IPC"].mean()
                )

                metrics["Instruction Scaling"] = (
                    self._stats_dict[ref_key].stats["Useful Instructions"].sum()
                    / stats["Useful Instructions"].sum()
                )

                metrics["Frequency Scaling"] = (
                    stats["Frequency"].mean()
                    / self._stats_dict[ref_key].stats["Frequency"].mean()
                )

                metrics["Computational Scaling"] = (
                    self._stats_dict[ref_key].stats["Total Useful Computation"].sum()
                    / stats["Total Useful Computation"].sum()
                )

                metrics["Global Efficiency"] = (
                    metrics["Computational Scaling"] * metrics["Parallel Efficiency"]
                )

                metrics["Speedup"] = (
                    self._stats_dict[ref_key].stats["Total Runtime"].max()
                    / stats["Total Runtime"].max()
                )

                metrics["Runtime"] = stats["Total Runtime"].max()

            except KeyError as err:
                raise ValueError(
                    "No '{}' statistic. (Wrong analysis type?)" "".format(err.args[0])
                )

            metrics_by_key[key] = metrics

        self._metric_data = pandas.DataFrame(metrics_by_key).T
