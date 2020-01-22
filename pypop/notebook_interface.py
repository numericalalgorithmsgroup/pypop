#!/usr/bin/env/python3

from ipywidgets import VBox, GridBox, Button, Layout, Label, Tab
from ipyfilechooser import FileChooser

from pypop.traceset import TraceSet
from pypop.metrics import Judit_Hybrid_Metrics

from .bokeh_widget import BokehWidget

from .metrics.metricset import MetricSet

import numpy
import pandas

import matplotlib.colors as mc

from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.colors import RGB


class FileSelector(VBox):

    _add_row_button_layout = Layout(width="12em", margin="1em 0em 0em 0em")
    _label_layout = Layout(width="2em")
    _remove_button_layout = Layout(width="3em")
    _grid_layout = Layout(grid_template_columns="2em auto 3em", margin="1em 0em 0em 0em")
    _calculate_button_layout = Layout(width="12em", margin="1em 0em 0em 0em")

    def __init__(
        self, base_dir=".", starting_files=None, calculation_callback=None, **kwargs
    ):
        self._base_dir = base_dir
        self._files = [None] if starting_files is None else starting_files
        self._calculation_callback = calculation_callback

        self._add_row_button = None
        self._filechooser_grid = None
        self._filechoosers = None

        self._create_filechoosers()
        self._create_buttons()
        self._update_filechooser_grid()

        super().__init__(
            children=[
                self._add_row_button,
                self._filechooser_grid,
                self._calculate_button,
            ],
            layout=Layout(width="auto"),
            **kwargs
        )

    def _create_buttons(self):
        self._add_row_button = Button(
            description="Add File",
            button_style="info",
            layout=self._add_row_button_layout,
        )
        self._add_row_button.on_click(self._add_filechooser_row)

        self._calculate_button = Button(
            description="Calculate Metrics",
            button_style="success",
            layout=self._calculate_button_layout,
        )
        if self._calculation_callback:
            self._calculate_button.on_click(self._calculation_callback)

    def _add_filechooser_row(self, callback_reference=None):

        self._files.append(None)
        self._update_filechooser_grid()

    def _create_filechoosers(self):

        self._filechoosers = [
            FileChooser(filename=fname, select_default=True) if fname else FileChooser()
            for fname in self._files
        ]
        [fc.register_callback(self._update_files) for fc in self._filechoosers]

    def _update_filechooser_grid(self):
        if self._filechooser_grid is None:
            self._filechooser_grid = GridBox([], layout=self._grid_layout)

        self._create_filechoosers()

        labels = [
            Label(value="{:d}".format(i + 1), layout=self._label_layout)
            for i in range(len(self._filechoosers))
        ]
        buttons = [
            Button(
                description="🗙", button_style="danger", layout=self._remove_button_layout
            )
            for i in range(len(self._filechoosers))
        ]
        [
            button.on_click(self._create_row_deleter(i))
            for i, button in enumerate(buttons)
        ]

        self._filechooser_grid.children = sum(
            zip(labels, self._filechoosers, buttons), ()
        )

    def _create_row_deleter(self, row_num):
        def deleter(callback_reference=None):
            self._files.pop(row_num)
            self._update_filechooser_grid()

        return deleter

    def _update_files(self, callback_reference=None):
        self._files = [fc.selected for fc in self._filechoosers]

    @property
    def filenames(self):
        self._update_files()
        return [file for file in self._files if file is not None]


class AutoMetricsGUI(Tab):
    def __init__(self, metric_calc, base_dir=".", starting_files=None, **kwargs):
        self._fileselector = FileSelector(
            base_dir=base_dir,
            starting_files=starting_files,
            calculation_callback=self._calculate_callback_hook,
        )

        self._metrics_display = None
        self._metric_calculator = metric_calc

        super().__init__(
            children=[self._fileselector], layout=Layout(width="auto"), **kwargs
        )

        self.set_title(0, "Trace Files")

    def _calculate_callback_hook(self, callback_reference=None):

        statistics = TraceSet(
            self._fileselector.filenames, ignore_cache=False, chop_to_roi=True
        )
        metrics = self._metric_calculator(statistics.by_commsize())
        metrics_display = MetricTable(metrics)

        if self._metrics_display is None:
            self.children = self.children + (metrics_display,)
            self.set_title(1, "Judit Metrics")
        else:
            self._metrics_display.close()
            new_children = list(self.children)
            new_children[1] = metrics_display
            self.children = new_children

        self._metrics_display = metrics_display
        metrics_display._plot_table()


def approx_string_em_width(string):
    """Give the approximate string width in 'em'

    Basically assume 'w' and 'm' have em width and everything else has
    a width of 0.6*em
    """

    return 0.6 * len(string) + 0.4 * sum(string.count(x) for x in ["w", "m", "~"])


class MetricTable(BokehWidget):
    def __init__(self, metrics: MetricSet, **kwargs):
        self._metrics = metrics
        super().__init__()

        self._setup_geometry()

    def _setup_geometry(self):

        # Geometry calculations:
        self._nrows = len(self._metrics.metrics) + 1
        self._ncols = len(self._metrics.metric_data.index)

        pt_to_px = 96 / 72
        self._fontsize = 16
        font_px = self._fontsize * pt_to_px
        self._cell_height = font_px * 2.2
        self._row_locs = numpy.linspace(
            0, -self._cell_height * (self._nrows - 1), self._nrows
        )

        metric_name_column_width = 0
        self._metric_name_column_text = ["Number of Processes"]
        self._metric_descriptions = [
            "The number of processes the application was run on"
        ]
        for i, metric in enumerate(self._metrics.metrics):
            self._metric_name_column_text.append(metric.key)
            self._metric_descriptions.append(metric.description)
            metric_name_column_width = max(
                metric_name_column_width, approx_string_em_width(metric.key)
            )

        self._border_pad = 20  # px
        self._left_pad = font_px / 2
        self._right_pad = font_px / 3
        self._metric_column_width = 1.1 * (
            metric_name_column_width * font_px + self._left_pad + self._right_pad
        )
        self._value_column_width = 1.1 * (
            approx_string_em_width("0.00") * font_px + self._left_pad + self._right_pad
        )

    def _init_figure(self):
        plot_width = (
            int(self._metric_column_width + self._ncols * self._value_column_width)
            + self._border_pad
        )  # 20 px padding at edge
        plot_height = int(self._cell_height * self._nrows) + self._border_pad  # px

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

        metric_cmap = mc.LinearSegmentedColormap.from_list(
            "POP_Metrics", colors=cmap_points, N=256, gamma=1
        )

        # Label column
        plotdata = pandas.DataFrame(
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
        )

        right_edges = plotdata["right_edges"].values

        for icol, idx in enumerate(self._metrics.metric_data.index):
            left_edges = right_edges
            right_edges = left_edges + self._value_column_width
            plotdata = pandas.concat(
                (
                    plotdata,
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
                                        for x in metric_cmap(
                                            self._metrics.metric_data[metric.key][idx]
                                        )[:3]
                                    )
                                )
                                for metric in self._metrics.metrics
                            ],
                            "data": ["{}".format(idx)]
                            + [
                                "{:.2f}".format(
                                    self._metrics.metric_data[metric.key][idx]
                                )
                                for metric in self._metrics.metrics
                            ],
                            "short_desc": self._metric_name_column_text,
                            "long_desc": self._metric_descriptions,
                        }
                    ),
                )
            )

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
