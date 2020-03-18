#!/usr/bin/env python3

from ipywidgets import Output

from bokeh.io import output_notebook, push_notebook
from bokeh.plotting import figure, show
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import MISSING_RENDERERS

silence(MISSING_RENDERERS)


class BokehWidget(Output):

    def __init__(self, *args, **kwargs):
        output_notebook(hide_banner=True)
        super().__init__(*args, **kwargs)
        self._figure = None
        self._handle = None

    def _initialize(self):
        with self:
            self._handle = show(self.figure, notebook_handle=True)

    def _init_figure(self):
        self._figure = figure(sizing_mode='stretch_both')

    @property
    def figure(self):
        if self._figure is None:
            self._init_figure()

        return self._figure

    def update(self):
        if self._handle is None:
            with self:
                self._handle = show(self.figure, notebook_handle=True)
        push_notebook(handle=self._handle)
