#!/usr/bin/env/python3

from ipywidgets import VBox, GridBox, Button, Layout, Label
from ipyfilechooser import FileChooser


class FileSelector(VBox):

    _label_layout = Layout(width="2em")
    _button_layout = Layout(width="6em")
    _grid_layout = Layout(grid_template_columns="2em auto 8em")

    def __init__(self, base_dir=".", starting_files=None, **kwargs):
        self._base_dir = base_dir
        self._files = [None] if starting_files is None else starting_files

        self._add_row_button = None
        self._filechooser_grid = None
        self._filechoosers = None

        self._create_filechoosers()
        self._create_add_row_button()
        self._update_filechooser_grid()

        super().__init__(
            children=[self._add_row_button, self._filechooser_grid],
            layout=Layout(width="auto"),
            **kwargs
        )

    def _assign_contents(self):

        self.children = [self._add_row_button, self._filechooser_grid]

    def _create_add_row_button(self):
        self._add_row_button = Button(description="Add File", button_style="info")
        self._add_row_button.on_click(self._add_filechooser_row)

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
                description="Remove", button_style="danger", layout=self._button_layout
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
