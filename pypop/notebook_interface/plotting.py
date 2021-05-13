#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from os import environ
from io import BytesIO
from sys import float_info
import numpy
import pandas
from warnings import filterwarnings, warn

from bokeh.plotting import figure
from bokeh.colors import RGB
from bokeh.palettes import all_palettes, linear_palette
from bokeh.models import HoverTool
from bokeh.embed import file_html
from bokeh.resources import INLINE
from bokeh.io.export import get_screenshot_as_png
from bokeh.transform import linear_cmap

from matplotlib.colors import LinearSegmentedColormap

from ..metrics.metricset import MetricSet
from .palettes import efficiency_red_green
from ..utils.plot import approx_string_em_width, get_pop_logo_data
from pypop.utils.exceptions import NoWebDriverError

filterwarnings(
    "ignore",
    ".* are being repeated",
    category=UserWarning,
    module="bokeh.plotting.helpers",
)


def get_any_webdriver():

    try:
        import selenium.webdriver
        from selenium.common.exceptions import WebDriverException
    except ImportError:
        raise NoWebDriverError("Selenium is not installed.")

    driver = None
    for drivername in ["Chrome", "Firefox"]:
        try:
            options = getattr(selenium.webdriver, "{}Options".format(drivername))()
            options.headless = True
            driver = getattr(selenium.webdriver, drivername)(options=options)
            break
        except WebDriverException:
            pass

    if driver:
        return driver

    raise NoWebDriverError("Failed to load suitable webdriver")


def build_discrete_cmap(factors):

    if len(factors) <= 10:
        cmap = all_palettes["Category10"][10]
    else:
        cmap = linear_palette("Turbo256", len(factors))

    return {k: v for k, v in zip(factors, cmap)}


class BokehBase:
    def __init__(self, *args, **kwargs):

        self._figure = None
        self._mpl_fallback = None
        self._update_callback = None

    def install_update_callback(self, callback):
        self._update_callback = callback

    def update(self):
        if self._update_callback is not None:
            self._update_callback()

    @property
    def figure(self):
        if self._figure is None:
            self._build_plot()

        return self._figure

    def _repr_html_(self):
        if environ.get("PYPOP_HEADLESS"):
            return None

        return file_html(self.figure, INLINE, "")

    def _repr_png_(self):

        try:
            driver = get_any_webdriver()
        except NoWebDriverError:
            warn("Webdriver unavailable - falling back to matplotlib interface")
            return self._mpl_fallback_png()

        self.figure.min_border = 10

        window_size = [1.1 * self.figure.plot_width, 1.1 * self.figure.plot_height]

        self.figure.toolbar_location = None

        driver.set_window_size(*window_size)

        try:
            img = get_screenshot_as_png(self.figure, driver=driver, timeout=10)
        except:
            driver.quit()
            warn("Webdriver failed - falling back to matplotlib interface")
            return self._mpl_fallback_png()

        imgbuffer = BytesIO()
        img.save(imgbuffer, format="png")
        return imgbuffer.getvalue()

    def _mpl_fallback_png(self):
        from pypop.mplplotting.mplplotbase import MPLPlotBase

        if isinstance(self._mpl_fallback, MPLPlotBase):
            return self._mpl_fallback._repr_png_()
        else:
            raise RuntimeError(
                "No Matplotlib fallback provided for class {}".format(self.__name__)
            )

    def save_html(self, path):
        imgcode = self._repr_html_()

        with open(path, "wt") as fh:
            fh.write(imgcode)

    def save_png(self, path):
        imgdata = self._repr_png_()

        with open(path, "wb") as fh:
            fh.write(imgdata)


class MetricTable(BokehBase):
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

        # Create matplotlib fallback object. This is lightweight unless used for e.g png
        # generation
        from pypop.mplplotting import MPLMetricTable

        self._mpl_fallback = MPLMetricTable(
            metrics,
            columns_key,
            group_key,
            group_label,
            title,
            columns_label,
            fontsize,
            pop_logo,
        )

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
        self._setup_geometry()
        self._eff_cmap = None

    def _setup_geometry(self):

        # Geometry calculations, N.B don't include a group-by row here:
        self._nrows = len(self._metrics.metrics) + 1
        self._ncols = len(self._metrics.metric_data.index)

        pt_to_px = 96 / 72
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
            + self._border_pad
        )  # px
        plot_height = int(
            0.0 - self._row_locs.min() + self._cell_height + self._border_pad
        )  # px

        self._plot_dims = (plot_width, plot_height)

        tooltip_template = """
<div style="width:{}px;color:#060684;font-family:sans;font-weight:bold;font-size:{}pt;">
@short_desc
</div>
<div style="width:{}px;padding-top:5px;font-size:{}pt">
@long_desc
</div>
        """.format(
            plot_width / 2, self.fontsize - 2, plot_width / 2, self.fontsize - 4
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
                        "cell_fills": numpy.full(num_groups, RGB(255, 255, 255)),
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
                            "text_inset": self._left_pad,
                            "bottom_edges": self._row_locs - self._cell_height,
                            "cell_fills": [RGB(255, 255, 255)]
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

        self._figure.quad(
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

        self._figure.text(
            x="left_edges",
            y="top_edges",
            x_offset="text_inset",
            y_offset=self._cell_height / 2,
            text="data",
            source=plotdata,
            text_baseline="middle",
            text_font_size="{}pt".format(self.fontsize),
        )

        if self._pop_logo:
            pop_data = get_pop_logo_data()
            img_h = pop_data.shape[0]
            img_w = pop_data.shape[1]

            render_h = self._logo_height
            render_w = self._logo_height * img_w / img_h
            self._figure.image_rgba(
                image=[pop_data.view(dtype=numpy.uint32).reshape((img_h, img_w))],
                x=20,
                y=-self._logo_height,
                dw=render_w,
                dh=render_h,
            )

        self.update()

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

            self._eff_cmap = LinearSegmentedColormap.from_list(
                "POP_Metrics", colors=cmap_points, N=256, gamma=1
            )

        if numpy.any(numpy.isnan(value)):
            return self.to_RGB(self._eff_cmap(-1))

        return self.to_RGB(self._eff_cmap(value))

    @staticmethod
    def to_RGB(value):

        return RGB(*(int(255 * x) for x in value[:3]))


class ScalingPlot(BokehBase):
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

        self._metrics = metrics
        super().__init__()

        # Create matplotlib fallback object. This is lightweight unless used for e.g png
        # generation
        from pypop.mplplotting import MPLScalingPlot

        self._mpl_fallback = MPLScalingPlot(
            metrics,
            scaling_variable,
            independent_variable,
            group_key,
            group_label,
            title,
            fontsize,
            fit_data_only,
            **kwargs
        )

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

    def _build_plot(self):

        # Geometry calculations - currently fix width at 60em height at 40em
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

        self._figure = figure(
            tools=["save"],
            min_border=0,
            aspect_ratio=1.5,
            sizing_mode="fixed",
            x_range=x_axis_range,
            y_range=y_axis_range,
            title=self.title,
        )

        self._figure.varea(
            xrange_ideal,
            y1=yrange_ideal,
            y2=yrange_80pc,
            fill_color="green",
            fill_alpha=0.2,
        )
        self._figure.varea(
            xrange_ideal,
            y1=yrange_80pc,
            y2=numpy.ones_like(yrange_80pc),
            fill_color="orange",
            fill_alpha=0.2,
        )
        self._figure.varea(
            xrange_ideal,
            y1=numpy.ones_like(xrange_ideal),
            y2=numpy.zeros_like(xrange_ideal),
            fill_color="red",
            fill_alpha=0.2,
        )

        self._figure.xaxis.axis_label_text_font_size = "{}pt".format(self.fontsize)
        self._figure.xaxis.major_label_text_font_size = "{}pt".format(self.fontsize - 1)
        self._figure.yaxis.axis_label_text_font_size = "{}pt".format(self.fontsize)
        self._figure.yaxis.major_label_text_font_size = "{}pt".format(self.fontsize - 1)

        self._figure.xaxis.axis_label = self._xaxis_key
        self._figure.yaxis.axis_label = self._yaxis_key

        xmin = float_info.max
        xmax = 0
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

        if self._group_key is None:
            self._figure.legend.visible = False

        self.update()


class TimelinePlot(BokehBase):
    def __init__(
        self, timelinedata, y_variable, color_by, tooltip=None, title=None, palette=None
    ):
        self._data = timelinedata
        super().__init__()

        self._y_axis_key = y_variable
        self._tooltip_template = tooltip
        self._title = title
        self._palette = palette if palette else efficiency_red_green
        self._color_by = color_by

    def _build_plot(self):

        self._figure = figure(
            tools=["save", "xwheel_zoom", "ywheel_zoom", "zoom_in", "zoom_out", "reset"],
            tooltips=self._tooltip_template,
        )

        self._figure.hbar(
            y=self._y_axis_key,
            left="Region Start",
            right="Region End",
            height=0.9,
            color=linear_cmap(self._color_by, self._palette, 0.0, 1.0),
            source=self._data,
        )
