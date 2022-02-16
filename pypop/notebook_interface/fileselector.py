#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

import os

from ipyfilechooser import FileChooser
from ipywidgets import (
    Accordion,
    Box,
    HBox,
    VBox,
    Checkbox,
    GridBox,
    Button,
    Layout,
    Label,
)


class ValidatingChooser(VBox):
    """IPyWidgets GUI Element for file selection with validation
    """

    _container_layout = Layout(margin="1em 0em 1em 0em")
    _msgbox_layout = Layout()

    def __init__(self, starting_path=None, **kwargs):
        """\
        Parameters
        ----------
        starting_path: str or None
            If a file, start with that file selected. If a directory, use as the starting
            directory for filechooser. If none, defaults to value of `os.os.getcwd()`
        """


        if starting_path is None:
            starting_dir = os.getcwd()
            starting_file = ""
        else:
            starting_path = os.path.os.path.normpath(starting_path)
            if os.path.isfile(starting_path):
                starting_file = os.path.basename(starting_path)
                starting_dir = os.path.os.path.dirname(starting_path)
            else:
                starting_file = ""
                starting_dir = starting_path

        self._filechooser = FileChooser(
            filename=starting_file,
            path=starting_dir,
            select_default=bool(starting_file),
            use_dir_icons=True,
        )
        self._msgbox = HBox(children=[], layout=self._msgbox_layout)
        self._validity = None

        super().__init__(children=[self._filechooser], **kwargs)

    def set_file_select_callback(self, callback):

        self._filechooser.register_callback(callback)

    @property
    def validity(self):
        return self._validity

    @property
    def files(self):
        raise NotImplementedError

    @property
    def selected(self):
        return self._filechooser.selected


class FileSelector(VBox):
    """IPyWidgets GUI Element for selecting arbitrary numbers of files with callback
    support
    """

    _button_container_layout = Layout(
        display="flex",
        width="100%",
        justify_content="space-between",
        margin="1em 0em 1em 0em",
        padding="0em 0em 0em 0em",
    )
    _control_button_layout = Layout(width="10em", margin="0em 2em 0em 0em")
    _label_layout = Layout(width="2em")
    _remove_button_layout = Layout(width="2.5em")
    _grid_layout = Layout(grid_template_columns="2em auto 3em", margin="1em 0em 1em 0em")
    _container_layout = Layout(padding="0em 1em 0em 1em", width="100%")

    def __init__(
        self,
        base_dir=None,
        starting_files=None,
        calculation_callback=None,
        analysis_state=None,
        **kwargs
    ):
        """\
        Parameters
        ----------

        base_dir: str or None
            Starting directory for filechooser. If None defaults to `os.os.getcwd()`.
        starting_files: list of str or None
            List of files to be initially selected.
        calculation_callback: callable
            Function to be executed when calculation button is pressed. Should have the
            standard IPyWidgets Button callback signature `f(WidgetInstance)`
        analysis_state: AnalysisState
            Saved analysis state to be passed in (overrides default options
            with those saved)
        """

        self._base_dir = base_dir if base_dir else os.getcwd()
        self._files = (
            [None]
            if starting_files is None
            else [os.path.normpath(os.path.expanduser(path)) for path in starting_files]
        )
        self._calculation_callback = calculation_callback
        self._analysis_state = analysis_state

        self._filechooser_grid = None
        self._filechoosers = None
        self._create_filechoosers()
        self._update_filechooser_grid()

        self._button_container = None
        self._add_row_button = None
        self._create_buttons()

        self._advanced_config_controls = {}
        self._advanced_config_box = None
        self._advanced_config_accordion = None
        self._create_advanced_config_box()

        super().__init__(
            children=[
                self._filechooser_grid,
                self._button_container,
                self._advanced_config_accordion,
            ],
            layout=self._container_layout,
            **kwargs
        )

    def _create_buttons(self):
        self._add_row_button = Button(
            description="Add File", layout=self._control_button_layout,
        )
        self._add_row_button.on_click(self._add_filechooser_row)

        self._calculate_button = Button(
            description="Analyze",
            button_style="success",
            layout=self._control_button_layout,
        )
        if self._calculation_callback:
            self._calculate_button.on_click(self._calculation_callback)

        self._button_container = Box(
            children=[self._calculate_button, self._add_row_button],
            layout=self._button_container_layout,
        )

    def _create_advanced_config_box(self):
        self._advanced_config_controls["Chop to ROI"] = Checkbox(
            value=True, description="Chop to ROI"
        )

        self._advanced_config_controls["Delete Cache"] = Checkbox(
            value=False, description="Delete Cache"
        )

        self._advanced_config_box = VBox(
            children=tuple(self._advanced_config_controls.values())
        )
        self._advanced_config_accordion = Accordion(children=[self._advanced_config_box])
        self._advanced_config_accordion.set_title(0, "Advanced Configuration")
        self._advanced_config_accordion.selected_index = None

    def _add_filechooser_row(self, callback_reference=None):

        self._files.append(None)
        self._update_filechooser_grid()

    def _create_filechoosers(self):

        self._filechoosers = [
            ValidatingChooser(starting_path=fname)
            if fname
            else ValidatingChooser(starting_path=os.path.expanduser('~'))
            for fname in self._files
        ]
        [fc.set_file_select_callback(self._update_files) for fc in self._filechoosers]

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

        if self._analysis_state is not None:
            self._analysis_state["trace_files"] = [
                file for file in self._files if file is not None
            ]

    @property
    def filenames(self):
        self._update_files()
        return [file for file in self._files if file is not None]
