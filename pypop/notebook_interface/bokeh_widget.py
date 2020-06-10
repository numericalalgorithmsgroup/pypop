#!/usr/bin/env python3

from ipywidgets import Output

from bokeh.io import output_notebook, push_notebook
from bokeh.plotting import show
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import MISSING_RENDERERS

silence(MISSING_RENDERERS)


class BokehWidgetWrapper(Output):
    def __init__(self, plot_object):
        output_notebook(hide_banner=True)
        super().__init__()
        self._plot_object = plot_object
        self._figure = plot_object.figure
        self._plot_object.install_update_callback(self.update)
        with self:
            self._handle = show(self._figure, notebook_handle=True)

    def update(self):
        if self._handle is None:
            with self:
                self._handle = show(self.figure, notebook_handle=True)
        push_notebook(handle=self._handle)
