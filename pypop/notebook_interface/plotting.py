#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from io import BytesIO
import numpy
import pandas

from bokeh.plotting import figure
from bokeh.colors import RGB
from bokeh.palettes import all_palettes, linear_palette
from bokeh.models import HoverTool
from bokeh.io.export import get_screenshot_as_png

from matplotlib.colors import LinearSegmentedColormap

from ..metrics.metricset import MetricSet

from .bokeh_widget import BokehWidget

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions


def approx_string_em_width(string):
    """Give the approximate string width in 'em'

    Basically assume 'w' and 'm' have em width and everything else has
    a width of 0.6*em
    """

    return 0.6 * len(string) + 0.4 * sum(string.count(x) for x in ["w", "m", "~"])


def build_discrete_cmap(factors):

    if len(factors) <= 10:
        cmap = all_palettes["Category10"][10]
    else:
        cmap = linear_palette("Turbo256", len(factors))

    return {k: v for k, v in zip(factors, cmap)}


class MetricTable(BokehWidget):
    def __init__(
        self,
        metrics: MetricSet,
        columns_key="auto",
        group_key="auto",
        title=None,
        columns_label=None,
        fontsize=14,
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

        self._fontsize = fontsize
        self._metric_name_column_text = []
        self._metric_descriptions = []
        self._setup_geometry()

    def _setup_geometry(self):

        # Geometry calculations, N.B don't include a group-by row here:
        self._nrows = len(self._metrics.metrics) + 1
        self._ncols = len(self._metrics.metric_data.index)

        pt_to_px = 96 / 72
        font_px = self._fontsize * pt_to_px
        self._cell_height = font_px * 2.2
        self._row_locs = numpy.linspace(
            0, -self._cell_height * (self._nrows - 1), self._nrows
        )

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
            self._metric_name_column_text.append(metric.key)
            self._metric_descriptions.append(metric.description)
            max_metric_em_width = max(
                max_metric_em_width, approx_string_em_width(metric.key)
            )

        # Calculate required value column width
        max_value_em_width = approx_string_em_width("0.00")
        for keyname in self._metrics.metric_data[self._columns_key]:
            max_value_em_width = max(
                max_value_em_width, approx_string_em_width("{}".format(keyname))
            )

        self._border_pad = 20  # px
        self._left_pad = font_px / 2
        self._right_pad = font_px / 3
        self._metric_column_width = 1.1 * (
            max_metric_em_width * font_px + self._left_pad + self._right_pad
        )
        self._value_column_width = 1.1 * (
            max_value_em_width * font_px + self._left_pad + self._right_pad
        )

    def _init_figure(self):
        plot_width = (
            int(self._metric_column_width + self._ncols * self._value_column_width)
            + self._border_pad
        )  # px
        plot_height = int(
            0.0 - self._row_locs.min() + self._cell_height + self._border_pad
        )  # px

        tooltip_template = """
<div style="width:{}px;color:#060684;font-family:sans;font-weight:bold;font-size:{}pt;">
@short_desc
</div>
<div style="width:{}px;padding-top:5px;font-size:{}pt">
@long_desc
</div>
        """.format(
            plot_width / 2, self._fontsize - 2, plot_width / 2, self._fontsize - 4
        )

        hover_tool = HoverTool(
            tooltips=tooltip_template, names=["quads"], toggleable=True
        )
        # create a new plot with a title and axis labels
        self._figure = figure(
            plot_height=plot_height,
            plot_width=plot_width,
            tools=[hover_tool, "save"],
            min_border=0,
            x_range=(-self._border_pad, plot_width),
            y_range=(-plot_height, self._border_pad),
            sizing_mode="fixed",
        )

        self._figure.min_border = 0
        self._figure.grid.visible = False
        self._figure.axis.visible = False
        self._figure.outline_line_color = None

    def _plot_table(self):

        bad_thres = 0.5
        good_thres = 0.8

        cmap_points = [
            (0.0, (0.690, 0.074, 0.074)),
            (bad_thres, (0.690, 0.074, 0.074)),
            (good_thres - 1e-20, (0.992, 0.910, 0.910)),
            (good_thres, (0.910, 0.992, 0.910)),
            (1.0, (0.074, 0.690, 0.074)),
        ]

        metric_cmap = LinearSegmentedColormap.from_list(
            "POP_Metrics", colors=cmap_points, N=256, gamma=1
        )

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
                        "bottom_edges": numpy.full(num_groups, self._row_locs[0]),
                        "cell_fills": numpy.full(num_groups, RGB(255, 255, 255)),
                        "data": [self._group_key]
                        + ["{}".format(g[0]) for g in group_iter],
                        "short_desc": [self._group_key] * num_groups,
                        "long_desc": [self._metrics._key_descriptions[self._group_key]]
                        * num_groups,
                    }
                )
            )

        # Label column
        plotdata.append(
            pandas.DataFrame(
                {
                    "left_edges": numpy.zeros(self._nrows),
                    "right_edges": numpy.full(self._nrows, self._metric_column_width),
                    "top_edges": self._row_locs,
                    "bottom_edges": self._row_locs - self._cell_height,
                    "cell_fills": numpy.full(self._nrows, RGB(255, 255, 255)),
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
                            "bottom_edges": self._row_locs - self._cell_height,
                            "cell_fills": [RGB(255, 255, 255)]
                            + [
                                RGB(
                                    *(
                                        int(255 * x)
                                        for x in metric_cmap(coldata[metric.key])[:3]
                                    )
                                )
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

        self.figure.quad(
            left="left_edges",
            right="right_edges",
            top="top_edges",
            bottom="bottom_edges",
            source=plotdata,
            line_color="black",
            line_width=1,
            fill_color="cell_fills",
            name="quads",
        )

        self.figure.text(
            x="left_edges",
            y="top_edges",
            x_offset=self._left_pad,
            y_offset=self._cell_height / 2,
            text="data",
            source=plotdata,
            text_baseline="middle",
            text_font_size="{}pt".format(self._fontsize),
        )

        self.update()

    def _repr_png_(self):
        options = ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        imgbuffer = BytesIO()
        img = get_screenshot_as_png(self.figure, driver=driver)
        img.save(imgbuffer, format="png")
        return imgbuffer.getvalue()



class ScalingPlot(BokehWidget):
    def __init__(
        self,
        metrics: MetricSet,
        scaling_variable="Speedup",
        independent_variable="auto",
        group_key="auto",
        title="None",
        fontsize=14,
        **kwargs
    ):

        self._metrics = metrics
        super().__init__()

        self._group_key = (
            self._metrics._default_group_key if group_key == "auto" else group_key
        )

        self._xaxis_key = (
            self._metrics._default_scaling_key
            if independent_variable == "auto"
            else independent_variable
        )
        self._yaxis_key = scaling_variable

        self._fontsize = fontsize

    def _build_plot(self):

        # Geometry calculations - currently fix width at 60em height at 40em
        pt_to_px = 96 / 72
        font_px = self._fontsize * pt_to_px
        width = int(60 * font_px)
        height = int(40 * font_px)

        plot_data = (
            self._metrics.metric_data[
                [self._xaxis_key, self._yaxis_key, self._group_key]
            ]
            .sort_values(self._group_key)
            .copy()
        )

        plot_data["Plotgroups"] = plot_data[self._group_key].apply(
            lambda x: "{} {}".format(x, self._group_key)
        )

        color_map = build_discrete_cmap(plot_data["Plotgroups"].unique())

        x_lims = plot_data[self._xaxis_key].min(), plot_data[self._xaxis_key].max()
        x_range = x_lims[1] - x_lims[0]
        x_range = x_lims[0] - 0.1 * x_range, x_lims[1] + 0.1 * x_range
        y_lims = plot_data[self._yaxis_key].min(), plot_data[self._yaxis_key].max()
        y_range = y_lims[1] - y_lims[0]
        y_range = y_lims[0] - 0.1 * y_range, y_lims[1] + 0.1 * y_range

        self._figure = figure(
            plot_width=width,
            plot_height=height,
            tools=["save"],
            min_border=0,
            sizing_mode="scale_both",
            x_range=x_range,
            y_range=y_range,
        )

        self._figure.xaxis.axis_label_text_font_size = "{}pt".format(self._fontsize)
        self._figure.xaxis.major_label_text_font_size = "{}pt".format(self._fontsize - 1)
        self._figure.yaxis.axis_label_text_font_size = "{}pt".format(self._fontsize)
        self._figure.yaxis.major_label_text_font_size = "{}pt".format(self._fontsize - 1)

        self._figure.xaxis.axis_label = self._xaxis_key
        self._figure.yaxis.axis_label = self._yaxis_key

        for group, groupdata in plot_data.groupby("Plotgroups", sort=False):
            groupdata = groupdata.sort_values(self._xaxis_key)
            self._figure.square(
                x=self._xaxis_key,
                y=self._yaxis_key,
                legend_label=group,
                source=groupdata,
                color=color_map[group],
                size=10,
                angle=numpy.pi / 4,
            )
            self._figure.line(
                x=self._xaxis_key,
                y=self._yaxis_key,
                legend_label=group,
                source=groupdata,
                line_color=color_map[group],
                line_width=1.5,
                alpha=0.6,
            )

        self._figure.legend.location = "top_left"

        self.update()
