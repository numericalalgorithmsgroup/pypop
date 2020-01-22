#!/usr/bin/python3

import sys

from tqdm import tqdm

from ipywidgets import FloatProgress, HBox, HTML, Layout
from html import escape


class ProgressWidget(HBox, tqdm):
    """
    Experimental IPython/Jupyter Notebook widget using tqdm!
    """

    _layout = Layout(width="100%", display="inline-flex", flex_flow="row wrap")
    _progress_bar_layout = Layout(flex="2")

    def __init__(self, iterable=None, tqdm_kwargs=None):

        if tqdm_kwargs is None:
            tqdm_kwargs = {}

        # Setup tqdm output
        if tqdm_kwargs.get("file", sys.stderr) in (sys.stderr, None):
            tqdm_kwargs["file"] = sys.stdout  # avoid the red block in IPython

        # Initialize parent class + avoid printing by using gui=True
        tqdm_kwargs["gui"] = True
        tqdm_kwargs.setdefault("bar_format", "{l_bar}{bar}{r_bar}")
        tqdm_kwargs["bar_format"] = tqdm_kwargs["bar_format"].replace("{bar}", "<bar/>")
        tqdm.__init__(self, iterable=iterable, **tqdm_kwargs)

        # Replace with IPython progress bar display (with correct total)
        self._unit_scale = 1 if self.unit_scale is True else self.unit_scale or 1
        self._total = self.total * self._unit_scale if self.total else 1
        self.n = 0

        self._progress_bar = FloatProgress(
            min=0, max=self._total, layout=self._progress_bar_layout
        )
        if not self.total:
            self._progress_bar.value = 1
            self._progress_bar.bar_style = "info"

        self._progress_msg = HTML()

        HBox.__init__(
            self, children=[self._progress_bar, self._progress_msg], layout=self._layout
        )
        self.iterable = iterable

        # Print initial bar state
        if not self.disable:
            self.display()

    def display(self):

        msg = tqdm.__repr__(self)

        self._progress_bar.value = self.n

        if msg:
            # html escape special characters (like '&')
            if "<bar/>" in msg:
                left, right = map(escape, msg.split("<bar/>", 1))
            else:
                left, right = "", escape(msg)

            # remove inesthetical pipes
            if left and left[-1] == "|":
                left = left[:-1]
            if right and right[0] == "|":
                right = right[1:]

            # Update description
            self._progress_bar.description = left
            self._progress_bar.style.description_width = "initial"

            # and rhs message
            if right:
                self._progress_msg.value = right

    def __iter__(self, *args, **kwargs):
        try:
            for obj in super(ProgressWidget, self).__iter__(*args, **kwargs):
                # return super(tqdm...) will not catch exception
                yield obj
        # NB: except ... [ as ...] breaks IPython async KeyboardInterrupt
        except:  # NOQA
            raise

    def update(self, *args, **kwargs):
        try:
            tqdm.update(self, *args, **kwargs)
        except Exception as exc:
            # cannot catch KeyboardInterrupt when using manual tqdm
            # as the interrupt will most likely happen on another statement
            raise exc
