#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Shared routines for different Metric Sets
"""

from warnings import warn

import numpy
import pandas

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib as mpl
import matplotlib.table as mt
import matplotlib.ticker as mtick

from ..trace import Trace
from .._plotsettings import pypop_mpl_params, figparams

__all__ = ["Metric", "MetricSet"]


class Metric:
    """
    freq_corr: bool
        Correct performance metrics based on average clock frequency (use to
        correct for node dynamic clocking issues).
    """

    def __init__(self, key, level, displayname=None, desc=None, is_inefficiency=False):
        self.key = key
        self.level = level
        self.description = str(desc) if desc else ""
        self.is_inefficiency = is_inefficiency

        if displayname:
            self.displayname = r"$\hookrightarrow$" * bool(self.level) + displayname
        else:
            self.displayname = r"$\hookrightarrow$" * bool(self.level) + self.key


class MetricSet:
    """Calculate and plot POP MPI metrics

    Statistics data is expected to have been produced with `collect_statistics()`

    Attributes
    ----------
    metric_data
    metric_definition

    """

    _default_metric_key = None

    def __init__(self, stats_dict, ref_key=None, sort_keys=True):
        """
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

        if not isinstance(stats_dict, dict):
            stats_dict = {k: v for k, v in enumerate(stats_dict)}

        for df in stats_dict.values():
            if not isinstance(df, Trace):
                raise ValueError(
                    "stats_dict must be an iterable of pypop.traceset.RunData"
                )

        self._stats_dict = stats_dict
        self._metric_data = None
        self._ref_key = ref_key
        self._sort_keys = sort_keys

    def _calculate_metrics(self):
        raise NotImplementedError

    def _repr_html_(self):
        return self.metric_data._repr_html_()

    @property
    def metric_data(self):
        """pandas.DataFrame: Calculated metric data.
        """
        if self._metric_data is None:
            self._calculate_metrics(ref_key=self._ref_key)
        return self._metric_data

    @property
    def metrics(self):
        """List of :py:class:`pypop.metrics.Metric`: List of metrics that will be
        calculated.
        """
        return self._metric_list

    def _create_subdataframe(self, metadata, idxkey):
        layout = metadata.application_layout
        layout_keys = {
            "Number of Processes": pandas.Series(data=[layout.commsize], index=[idxkey]),
            "Threads per Process": pandas.Series(
                data=[layout.rank_threads[0][0]], index=[idxkey]
            ),
            "Total Threads": pandas.Series(
                data=[sum(x[0] for x in layout.rank_threads)], index=[idxkey]
            ),
            "Hybrid Layout": pandas.Series(
                data=["{}x{}".format(layout.commsize, layout.rank_threads[0][0])],
                index=[idxkey],
            ),
        }

        for metric in self._metric_list:
            layout_keys[metric.key] = pandas.Series(data=[0.0], index=[idxkey])

        return pandas.DataFrame(layout_keys)

    def plot_table(
        self,
        columns_key="auto",
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
            "Number of Processes"). If `'auto'`, a suitable default for the metric type
            is used, if `None` then the numerical index will be used.
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

        if columns_key == "auto":
            columns_key = self._default_metric_key

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
            columns_values = self.metric_data.index
            if columns_label is None:
                columns_label = "Index"
        else:
            columns_values = self.metric_data[columns_key]
            if not columns_label:
                columns_label = columns_key

        label_cell_width = 0.60
        body_cell_width = (1 - label_cell_width) / (len(columns_values) - skipfirst)
        body_cell_height = 0.1
        level_pad = 0.075

        pop_red = (0.690, 0.074, 0.074)
        pop_fade = (0.992, 0.910, 0.910)
        pop_green = (0.074, 0.690, 0.074)

        ineff_points = [
            (0.0, pop_green),
            (1 - good_thres, pop_fade),
            (1 - good_thres + 1e-20, pop_fade),
            (1 - bad_thres, pop_red),
            (1.0, pop_red),
        ]

        eff_points = [
            (0.0, pop_red),
            (bad_thres, pop_red),
            (good_thres - 1e-20, pop_fade),
            (good_thres, pop_fade),
            (1.0, pop_green),
        ]

        ineff_cmap = mc.LinearSegmentedColormap.from_list(
            "POP_Metrics", colors=ineff_points, N=256, gamma=1
        )

        eff_cmap = mc.LinearSegmentedColormap.from_list(
            "POP_Metrics", colors=eff_points, N=256, gamma=1
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
        metric_table.auto_set_font_size(True)
        #        metric_table.set_fontsize(8)

        metric_table.add_cell(
            0,
            0,
            width=label_cell_width,
            height=body_cell_height,
            text=columns_label,
            loc="center",
        )

        for col_num, col_data in enumerate(columns_values, start=1):
            if col_num <= skipfirst:
                continue
            metric_table.add_cell(
                0, col_num - skipfirst, text="{}".format(col_data), **body_cell_kwargs
            )

        for row_num, metric in enumerate(self.metrics, start=1):
            cmap = ineff_cmap if metric.is_inefficiency else eff_cmap
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
                        facecolor=cmap(col_data),
                        **body_cell_kwargs
                    )

        ax[0].add_table(metric_table)

        if title:
            ax[0].set_title(title)

        return fig

    def plot_scaling(self, x_key="auto", y_key="Speedup", label=None, title=None):
        """Plot scaling graph with region shading.

        Plots scaling data from pandas dataframe(s). The 0-80% and 80-100% scaling
        regions are shaded for visual identification.  Multiple scaling lines may be
        plotted by passing a dict of dataframes.

        Parameters
        ----------
        x_key: scalar
            Key of Dataframe column to use as x-axis. If 'auto' use a suitable default
            for the metric.

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
        if x_key == "auto":
            x_key = self._default_metric_key

        with mpl.rc_context(pypop_mpl_params):
            return self._plot_scaling(x_key, y_key, label, title)

    def _plot_scaling(self, x_key, y_key, label, title):

        if label is None:
            label = str(y_key)

        cores_min = numpy.nan
        cores_max = numpy.nan
        y_max = numpy.nan

        cores_min = numpy.nanmin([cores_min, self.metric_data[x_key].min()])
        cores_max = numpy.nanmax([cores_max, self.metric_data[x_key].max()])
        y_max = numpy.nanmax((y_max, self.metric_data[y_key].max()))

        y_max *= 1.2

        x_margin = 0.02 * cores_max
        ideal_scaling_cores = numpy.linspace(cores_min - x_margin, cores_max + x_margin)
        ideal_scaling = ideal_scaling_cores / cores_min
        ideal_scaling_80pc = 0.2 + 0.8 * ideal_scaling

        y_max = max(y_max, ideal_scaling.max())

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
            self.metric_data[x_key],
            self.metric_data[y_key],
            label=label,
            marker="x",
            linestyle="-",
            alpha=0.8,
        )

        ax.set_xlim(ideal_scaling_cores.min(), ideal_scaling_cores.max())
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Total cores")
        ax.set_ylabel(label)
        ax.xaxis.set_major_locator(mtick.FixedLocator(self.metric_data[x_key], 6))
        ax.legend(loc="upper left")

        if title:
            ax.set_title(title)

        return fig
