#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from ipywidgets import Button, VBox

from ipyfilechooser import FileChooser


class ReportGenerator(VBox):
    def __init__(self, **kwargs):

        self._filechooser = FileChooser(
            title="Report Notebook Filename",
            filename="report.ipynb",
            select_default=True,
        )

        self._button_generate_report = Button(
            description="Generate", button_style="success"
        )

        super().__init__(
            children=[self._filechooser, self._button_generate_report], **kwargs
        )
