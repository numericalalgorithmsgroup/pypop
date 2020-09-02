#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from string import Formatter

from ipywidgets import Button, Text, VBox

from ipyfilechooser import FileChooser

import nbformat

latest_nbformat = getattr(nbformat, "v{}".format(nbformat.current_nbformat))
new_nb = latest_nbformat.new_notebook
code_cell = latest_nbformat.new_code_cell
md_cell = latest_nbformat.new_markdown_cell


def quiet_code_cell(*args, **kwargs):

    quiet_cell = {"hide_output": True}

    if "metadata" in kwargs:
        kwargs["metadata"].update(quiet_cell)
    else:
        kwargs["metadata"] = quiet_cell

    return code_cell(*args, **kwargs)


class ReportGenerator(VBox):

    _text_header = """
"""

    _import_header = """
from pypop.traceset import TraceSet
from pypop.metrics import {metric_class}
from pypop.notebook_interface.plotting import MetricTable, ScalingPlot
"""

    _file_load = """
files_list = [{trace_file_list}]
statistics = TraceSet(files_list)
metrics = {metric_class}(statistics)
"""

    _appl_info = """
# Application Information
* Application Name:
* Applicant:
* Language:
* Programming Model: {programming_model}
* Application description:
* Run Parameters:
* Machine Environment:
"""

    _scaling_header = """
# Application Scaling
"""

    _scaling_plot = """
scaling_plot = ScalingPlot(metrics)
display(scaling_plot)
"""

    _scaling_discussion = """
_Discussion of scaling results._
"""

    _metrics_header = """
# Application Efficiency Metrics
"""

    _metrics_table = """
metric_table = MetricTable(metrics)
display(metric_table)
"""

    _metrics_discussion = """
_Discussion of metrics._
"""

    _conclusion_cell = """
# Conclusions

_Key findings and recommendations._
"""

    _notebook_layout = [
        (_text_header, md_cell),
        (_import_header, quiet_code_cell),
        (_file_load, quiet_code_cell),
        (_appl_info, md_cell),
        (_scaling_header, md_cell),
        (_scaling_plot, code_cell),
        (_scaling_discussion, md_cell),
        (_metrics_header, md_cell),
        (_metrics_table, code_cell),
        (_metrics_discussion, md_cell),
        (_conclusion_cell, md_cell),
    ]

    def __init__(self, analysis_state, **kwargs):

        self._analysis_state = analysis_state

        self._filechooser = FileChooser(
            title="Report Notebook Filename",
            filename="report.ipynb",
            select_default=True,
        )

        self._codename = Text(description="Application:", placeholder="")
        self._author = Text(description="Report Author:", placeholder="")
        self._contributors = Text(description="Contributors:", placeholder="")
        self._report_id = Text(description="Report ID:", value="DEMO_001")

        self._button_generate_report = Button(
            description="Generate", button_style="success"
        )
        self._button_generate_report.on_click(self._generate_report)

        super().__init__(
            children=[
                self._filechooser,
                self._codename,
                self._author,
                self._contributors,
                self._report_id,
                self._button_generate_report,
            ],
            **kwargs
        )

    def _generate_subst_dict(self):

        keys = set()
        for cell_text, _ in self._notebook_layout:
            keys.update([k[1] for k in Formatter().parse(cell_text) if k[1] is not None])

        #       Debug condition
        #        if "" in keys or any(x.isnumeric() for x in keys):
        #            raise ValueError("")

        return {key: getattr(self, "_get_{}".format(key))() for key in keys}

    def _generate_report_nb_metadata(self):

        return {
            "reprefnum": self._report_id.value,
            "title": "Performance Report --- {}".format(self._codename.value),
            "author": self._author.value,
            "contributors": self._contributors.value,
        }

    def _generate_report(self, callback_reference=None):

        subst_dict = self._generate_subst_dict()

        report_nb = new_nb(
            metadata={
                "pop_metadata": self._generate_report_nb_metadata(),
                "celltoolbar": "Edit Metadata",
            }
        )

        for cell_text, cell_ctr in self._notebook_layout:
            report_nb.cells.append(cell_ctr(cell_text.strip().format(**subst_dict)))

        nbformat.write(report_nb, self._filechooser.selected)

    def _get_metric_class(self):
        return self._analysis_state["metrics_object"].__class__.__name__

    def _get_trace_file_list(self):
        return ", ".join('"{}"'.format(x) for x in self._analysis_state["trace_files"])

    def _get_programming_model(self):
        return self._analysis_state["metrics_object"]._programming_model
