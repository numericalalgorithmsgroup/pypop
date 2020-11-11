#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

import numpy
import pandas

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.cm as mcm
import matplotlib.colors as mc
import matplotlib.table as mt
import matplotlib.ticker as mtick

from pypop.metrics.metricset import MetricSet
from pypop.utils.plot import approx_string_em_width, get_pop_logo_data

from .mplplotbase import MPLPlotBase

__all__ = ["MPLMetricTable", "MPLScalingPlot"]


def build_discrete_cmap(factors):

    if len(factors) <= 10:
        cmap = mcm.get_cmap("tab10", 10)
    else:
        cmap = mcm.get_cmap("rainbow", len(factors))

    if len(factors) == 1:
        return {factors[0]: cmap(0.0)}

    return {k: cmap(i / (len(factors) - 1)) for i, k in enumerate(factors)}


class MPLScalingPlot(MPLPlotBase):
    def __init__(
        self,
        metrics: MetricSet,
        scaling_variable="Speedup",
        independent_variable="auto",
        group_key="auto",
        group_label=None,
        title=None,
        fontsize=14,
        fit_data_only=False,
        **kwargs
    ):
        super().__init__()

        self._metrics = metrics

        self._group_key = (
            self._metrics._default_group_key if group_key == "auto" else group_key
        )

        self._group_label = group_label if group_label else self._group_key

        self._xaxis_key = (
            self._metrics._default_scaling_key
            if independent_variable == "auto"
            else independent_variable
        )

        self._yaxis_key = scaling_variable
        self.fontsize = fontsize
        self.fit_data_only = fit_data_only
        self.title = title
        self._dpi = 96

    def _build_plot(self):

        # Geometry calculations - currently fix width at 48em height at 32em
        pt_to_px = 96 / 72
        font_px = self.fontsize * pt_to_px
        width = int(48 * font_px)
        height = int(32 * font_px)

        self._plot_dims = (width, height)

        if self._group_key:
            plot_data = (
                self._metrics.metric_data[
                    [self._xaxis_key, self._yaxis_key, self._group_key]
                ]
                .sort_values(self._group_key)
                .copy()
            )
            plot_data["Plotgroups"] = plot_data[self._group_key].apply(
                lambda x: "{} {}".format(x, self._group_label)
            )
        else:
            plot_data = self._metrics.metric_data[
                [self._xaxis_key, self._yaxis_key]
            ].copy()
            plot_data["Plotgroups"] = "NULL"

        color_map = build_discrete_cmap(plot_data["Plotgroups"].unique())
        x_lims = numpy.asarray(
            [plot_data[self._xaxis_key].min(), plot_data[self._xaxis_key].max()]
        )
        y_lims = numpy.asarray(
            [plot_data[self._yaxis_key].min(), plot_data[self._yaxis_key].max()]
        )

        xrange_ideal = x_lims + numpy.asarray([0, numpy.diff(x_lims)[0] * 0.1])
        yrange_ideal = xrange_ideal / x_lims[0]
        yrange_80pc = 0.8 * yrange_ideal + 0.2

        x_axis_range = x_lims
        if self.fit_data_only:
            y_axis_range = y_lims
        else:
            y_axis_range = (
                min(y_lims[0], yrange_ideal[0]),
                max(y_lims[1], yrange_ideal[1]),
            )

        x_expand = numpy.asarray([-0.1, 0.1]) * numpy.abs(numpy.diff(x_lims))
        y_expand = numpy.asarray([-0.1, 0.1]) * numpy.abs(numpy.diff(y_lims))
        x_axis_range = x_axis_range + x_expand
        y_axis_range = y_axis_range + y_expand

        self._figure = Figure(
            figsize=tuple(x / self._dpi for x in self._plot_dims),
            dpi=self._dpi,
            tight_layout=False,
            constrained_layout=False,
        )

        self._figax = self._figure.add_axes([0.125, 0.125, 0.8, 0.78])

        self._figax.grid(True, which="major", axis="both", alpha=0.4)

        self._canvas = FigureCanvasAgg(self._figure)

        self._figax.fill_between(
            xrange_ideal, y1=yrange_ideal, y2=yrange_80pc, color="green", alpha=0.2, lw=0
        )
        self._figax.fill_between(
            xrange_ideal,
            y1=yrange_80pc,
            y2=numpy.ones_like(yrange_80pc),
            color="orange",
            alpha=0.2,
            lw=0,
        )
        self._figax.fill_between(
            xrange_ideal,
            y1=numpy.ones_like(xrange_ideal),
            y2=numpy.zeros_like(xrange_ideal),
            color="red",
            alpha=0.2,
            lw=0,
        )

        for group, groupdata in plot_data.groupby("Plotgroups", sort=False):
            groupdata = groupdata.sort_values(self._xaxis_key)
            self._figax.plot(
                groupdata[self._xaxis_key],
                groupdata[self._yaxis_key],
                label=group,
                color=color_map[group],
                markersize=10,
                marker="D",
            )

        self._figax.set_xlabel(self._xaxis_key, labelpad=self._figax.xaxis.labelpad + 8)
        self._figax.set_ylabel(self._yaxis_key, labelpad=self._figax.yaxis.labelpad + 8)

        self._figax.set_xlim(*x_axis_range)
        self._figax.set_ylim(*y_axis_range)

        self._figax.xaxis.set_major_locator(mtick.MaxNLocator(5))
        self._figax.xaxis.set_minor_locator(mtick.MaxNLocator(5))
        self._figax.yaxis.set_major_locator(mtick.MaxNLocator(8))
        self._figax.yaxis.set_minor_locator(mtick.MaxNLocator(5))

        for item in (
            [self._figax.title, self._figax.xaxis.label, self._figax.yaxis.label]
            + self._figax.get_xticklabels()
            + self._figax.get_yticklabels()
        ):
            item.set_fontsize(20)

        if self._group_key is not None:
            self._figax.legend(loc="upper left")


class MPLMetricTable(MPLPlotBase):
    def __init__(
        self,
        metrics: MetricSet,
        columns_key="auto",
        group_key="auto",
        group_label=None,
        title=None,
        columns_label=None,
        fontsize=14,
        pop_logo=True,
        **kwargs
    ):
        self._metrics = metrics
        super().__init__()

        self._columns_key = (
            self._metrics._default_metric_key if columns_key == "auto" else columns_key
        )
        self._group_key = (
            self._metrics._default_group_key if group_key == "auto" else group_key
        )

        self._group_label = group_label if group_label else self._group_key

        self.fontsize = fontsize
        self._pop_logo = pop_logo

        self._metric_name_column_text = []
        self._metric_descriptions = []
        self._eff_cmap = None
        self._dpi = 96
        self._setup_geometry()

    def _setup_geometry(self):

        # Geometry calculations, N.B don't include a group-by row here:
        self._nrows = len(self._metrics.metrics) + 1
        self._ncols = len(self._metrics.metric_data.index)

        pt_to_px = self._dpi / 72
        font_px = self.fontsize * pt_to_px
        self._logo_height = 50
        self._logo_subpad = 10
        self._cell_height = font_px * 2.2
        self._border_pad = 10  # px
        self._left_pad = font_px / 2
        self._right_pad = font_px / 3
        self._row_locs = numpy.linspace(
            0, -self._cell_height * (self._nrows - 1), self._nrows
        )

        # Offset rows if there will be a POP Logo included
        if self._pop_logo:
            self._row_locs -= self._logo_height + self._logo_subpad

        # Offset rows if there will be a header row
        if self._group_key is not None:
            self._row_locs -= self._cell_height

        # Calculate required key column width
        self._metric_name_column_text.append(self._columns_key)
        self._metric_descriptions.append(
            self._metrics._key_descriptions[self._columns_key]
        )
        max_metric_em_width = approx_string_em_width(self._columns_key)

        # group_key width
        if self._group_key is not None:
            max_metric_em_width = max(
                max_metric_em_width, approx_string_em_width(self._group_key)
            )

        # metric name widths
        for metric in self._metrics.metrics:
            self._metric_name_column_text.append(metric.displayname)
            self._metric_descriptions.append(metric.description)
            max_metric_em_width = max(
                max_metric_em_width, approx_string_em_width(metric.displayname)
            )

        # Calculate required value column width
        max_value_em_width = approx_string_em_width("0.00")
        for keyname in self._metrics.metric_data[self._columns_key]:
            max_value_em_width = max(
                max_value_em_width, approx_string_em_width("{}".format(keyname))
            )

        self._metric_column_width = 1.1 * (
            max_metric_em_width * font_px + self._left_pad + self._right_pad
        )
        self._value_column_width = 1.1 * (
            max_value_em_width * font_px + self._left_pad + self._right_pad
        )

    def _build_plot(self):
        plot_width = (
            int(self._metric_column_width + self._ncols * self._value_column_width)
            + 2 * self._border_pad
        )  # px
        plot_height = int(
            0.0 - self._row_locs.min() + self._cell_height + 2 * self._border_pad
        )  # px

        # Total plot pane size
        self._plot_dims = (plot_width, plot_height)

        # Now calculate axis size as a fraction of total
        table_ax_layout = [
            self._border_pad / plot_width,  # left instep
            self._border_pad / plot_height,  # Bottom instep
            (plot_width - 2 * self._border_pad) / plot_width,  # Table width
            (self._row_locs.max() - self._row_locs.min() + self._cell_height)
            / plot_height,  # Table height
        ]

        figsize = tuple(x / self._dpi for x in self._plot_dims)

        # create a new plot with a title and axis labels
        self._figure = Figure(
            figsize=figsize, dpi=self._dpi, tight_layout=False, constrained_layout=False,
        )

        if self._pop_logo:
            pop_data = get_pop_logo_data()
            img_h = pop_data.shape[0]
            img_w = pop_data.shape[1]
            logo_aspect = img_w / img_h

            logo_ax_layout = [
                self._border_pad / plot_width,  # Left instep
                1
                - (self._border_pad + self._logo_height) / plot_height,  # Bottom instep
                self._logo_height * logo_aspect / plot_width,  # Logo width
                self._logo_height / plot_height,  # Logo height
            ]

            self._logo_ax = self._figure.add_axes(logo_ax_layout, frame_on=False)
            self._logo_ax.xaxis.set_visible(False)
            self._logo_ax.yaxis.set_visible(False)

            self._logo_ax.imshow(pop_data, origin="lower")

        self._table_ax = self._figure.add_axes(table_ax_layout, frame_on=False)

        self._table_ax.xaxis.set_visible(False)
        self._table_ax.yaxis.set_visible(False)

        metric_cmap = self.efficiency_cmap

        if self._group_key is not None:
            group_iter = [
                (k, v)
                for k, v in self._metrics.metric_data.groupby(
                    self._group_key, axis="rows"
                )
            ]
        else:
            group_iter = [(1, self._metrics.metric_data)]

        # list of dataframes we will concatenate later
        plotdata = []

        # Optional grouping header row
        if self._group_key is not None:
            groupsizes = [g[1].index.shape[0] for g in group_iter]
            num_groups = len(groupsizes) + 1
            edges = numpy.cumsum(
                [0.0, self._metric_column_width]
                + [self._value_column_width * nc for nc in groupsizes]
            )
            left_edges = edges[:-1]
            right_edges = edges[1:]
            plotdata.append(
                pandas.DataFrame(
                    {
                        "left_edges": left_edges,
                        "right_edges": right_edges,
                        "top_edges": numpy.full(
                            num_groups, self._row_locs[0] + self._cell_height
                        ),
                        "text_inset": self._left_pad,
                        "bottom_edges": numpy.full(num_groups, self._row_locs[0]),
                        "cell_fills": [(1.0, 1.0, 1.0)] * num_groups,
                        "data": [self._group_label]
                        + ["{}".format(g[0]) for g in group_iter],
                        "short_desc": [self._group_key] * num_groups,
                        "long_desc": [self._metrics._key_descriptions[self._group_key]]
                        * num_groups,
                    }
                )
            )

        metric_insets = [self._left_pad] + [
            self._left_pad * (1 + 1.5 * m.level) for m in self._metrics.metrics
        ]

        # Label column
        plotdata.append(
            pandas.DataFrame(
                {
                    "left_edges": numpy.zeros(self._nrows),
                    "right_edges": numpy.full(self._nrows, self._metric_column_width),
                    "top_edges": self._row_locs,
                    "text_inset": metric_insets,
                    "bottom_edges": self._row_locs - self._cell_height,
                    "cell_fills": [(1.0, 1.0, 1.0)] * self._nrows,
                    "data": self._metric_name_column_text,
                    "short_desc": self._metric_name_column_text,
                    "long_desc": self._metric_descriptions,
                }
            ),
        )

        right_edges = plotdata[-1]["right_edges"].values

        for grouplabel, metricgroup in group_iter:
            metricgroup = metricgroup.sort_values(self._columns_key)
            for _, coldata in metricgroup.iterrows():
                left_edges = right_edges
                right_edges = left_edges + self._value_column_width
                plotdata.append(
                    pandas.DataFrame(
                        {
                            "left_edges": left_edges,
                            "right_edges": right_edges,
                            "top_edges": self._row_locs,
                            "text_inset": self._left_pad,
                            "bottom_edges": self._row_locs - self._cell_height,
                            "cell_fills": [(1.0, 1.0, 1.0)]
                            + [
                                metric_cmap(coldata[metric.key])
                                for metric in self._metrics.metrics
                            ],
                            "data": ["{}".format(coldata[self._columns_key])]
                            + [
                                "{:.2f}".format(coldata[metric.key])
                                for metric in self._metrics.metrics
                            ],
                            "short_desc": self._metric_name_column_text,
                            "long_desc": self._metric_descriptions,
                        }
                    ),
                )

        plotdata = pandas.concat(plotdata)

        plotdata["widths"] = plotdata["right_edges"] - plotdata["left_edges"]
        plotdata["heights"] = plotdata["top_edges"] - plotdata["bottom_edges"]

        for idx, row in plotdata.iterrows():
            cell = mt.Cell(
                (row["left_edges"], row["bottom_edges"]),
                row["widths"],
                row["heights"],
                text=row["data"],
                loc="left",
                facecolor=row["cell_fills"],
            )
            cell.PAD = row["text_inset"] / row["widths"]
            self._table_ax.add_artist(cell)

        xlim = (plotdata["left_edges"].min(), plotdata["right_edges"].max())
        ylim = (plotdata["bottom_edges"].min(), plotdata["top_edges"].max())

        self._table_ax.set_xlim(*xlim)
        self._table_ax.set_ylim(*ylim)

    def efficiency_cmap(self, value):
        if self._eff_cmap is None:
            bad_thres = 0.5
            good_thres = 0.8

            cmap_points = [
                (0.0, (0.690, 0.074, 0.074)),
                (bad_thres, (0.690, 0.074, 0.074)),
                (good_thres - 1e-5, (0.992, 0.910, 0.910)),
                (good_thres, (0.910, 0.992, 0.910)),
                (1.0, (0.074, 0.690, 0.074)),
            ]

            self._eff_cmap = mc.LinearSegmentedColormap.from_list(
                "POP_Metrics", colors=cmap_points, N=256, gamma=1
            )

        if numpy.any(numpy.isnan(value)):
            return self._eff_cmap(-1)

        return self._eff_cmap(value)
