#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from ipywidgets import Tab, Layout

from ..traceset import TraceSet
from .fileselector import FileSelector
from .plotting import MetricTable, ScalingPlot
from .reporting import ReportGenerator


class MetricsWizard(Tab):
    def __init__(self, metric_calc, base_dir=".", starting_files=None, **kwargs):
        self._fileselector = FileSelector(
            base_dir=base_dir,
            starting_files=starting_files,
            calculation_callback=self._calculate_callback_hook,
        )

        self._metrics_display = None
        self._scaling_plot = None
        self._report_generator = None
        self._metric_calculator = metric_calc

        super().__init__(
            children=[self._fileselector], layout=Layout(width="auto"), **kwargs
        )

        self.set_title(0, "Trace Files")

    def _calculate_callback_hook(self, callback_reference=None):

        statistics = TraceSet(
            self._fileselector.filenames, force_recalculation=False, chop_to_roi=True
        )
        metrics = self._metric_calculator(statistics)
        metrics_display = MetricTable(metrics)

        if self._metrics_display is None:
            self.children = self.children + (metrics_display,)
            self.set_title(1, "Metrics Table")
        else:
            self._metrics_display.close()
            new_children = list(self.children)
            new_children[1] = metrics_display
            self.children = new_children

        self._metrics_display = metrics_display
        metrics_display._plot_table()

        scaling_plot = ScalingPlot(metrics)

        if self._scaling_plot is None:
            self.children = self.children + (scaling_plot,)
            self.set_title(2, "Scaling Plot")
        else:
            self._scaling_plot.close()
            new_children = list(self.children)
            new_children[2] = scaling_plot
            self.children = new_children

        self._scaling_plot = scaling_plot
        scaling_plot._build_plot()

        report_generator = ReportGenerator()

        if self._report_generator is None:
            self.children = self.children + (report_generator,)
            self.set_title(3, "Report Generation")
        else:
            self._report_generator.close()
            new_children = list(self.children)
            new_children[3] = report_generator
            self.children = new_children
