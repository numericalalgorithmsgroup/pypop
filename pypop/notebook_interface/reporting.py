#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from string import Formatter

from ipywidgets import Button, VBox

from ipyfilechooser import FileChooser

import nbformat

latest_nbformat = getattr(nbformat, "v{}".format(nbformat.current_nbformat))
new_nb = latest_nbformat.new_notebook
code_cell = latest_nbformat.new_code_cell
md_cell = latest_nbformat.new_markdown_cell


class ReportGenerator(VBox):

    _text_header = """\
# PyPOP Metrics Report\
"""

    _import_header = """\
from pypop.traceset import TraceSet
from pypop.metrics import {metric_class}\
"""

    _file_load = """\
files_list = [{trace_file_list}]
statistics = TraceSet(files_list)\
    """

    _appl_info = """\
# Application Information
* Application Name:
* Applicant:
* Language:
* Programming Model: {programming_model}
* Application description:
* Run Parameters:
* Machine Environment:\
"""

    _metric_calc = """\
metrics = {metric_class}(statistics)\
"""

    _table_plot = """\
metric_table = metrics.plot_table()
display(metric_table)\
"""

    _scaling_plot = """\
scaling_plot = metrics.plot_scaling()
display(scaling_plot)\
"""

    _notebook_layout = [
        (_text_header, md_cell),
        (_import_header, code_cell),
        (_file_load, code_cell),
        (_appl_info, md_cell),
        (_metric_calc, code_cell),
        (_table_plot, code_cell),
        (_scaling_plot, code_cell),
    ]

    def __init__(self, analysis_state, **kwargs):

        self._analysis_state = analysis_state

        self._filechooser = FileChooser(
            title="Report Notebook Filename",
            filename="report.ipynb",
            select_default=True,
        )

        self._button_generate_report = Button(
            description="Generate", button_style="success"
        )
        self._button_generate_report.on_click(self._generate_report)

        super().__init__(
            children=[self._filechooser, self._button_generate_report], **kwargs
        )

    def _generate_subst_dict(self):

        keys = set()
        for cell_text, _ in self._notebook_layout:
            keys.update([k[1] for k in Formatter().parse(cell_text) if k[1] is not None])

        #       Debug condition
        #        if "" in keys or any(x.isnumeric() for x in keys):
        #            raise ValueError("")

        return {key: getattr(self, "_get_{}".format(key))() for key in keys}

    def _generate_report(self, callback_reference=None):

        subst_dict = self._generate_subst_dict()

        report_nb = new_nb()

        for cell_text, cell_ctr in self._notebook_layout:
            report_nb.cells.append(cell_ctr(cell_text.format(**subst_dict)))

        nbformat.write(report_nb, self._filechooser.selected)

    def _get_metric_class(self):
        return self._analysis_state['metrics_object'].__class__.__name__

    def _get_trace_file_list(self):
        return ", ".join('"{}"'.format(x) for x in self._analysis_state["trace_files"])

    def _get_programming_model(self):
        return self._analysis_state['metrics_object']._programming_model
