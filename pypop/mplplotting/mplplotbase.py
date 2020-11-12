#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from io import BytesIO


class MPLPlotBase(object):
    def __init__(self):
        self._figure = None

    def _build_plot(self):
        raise NotImplementedError("MPLPlotBase should not be used directly!")

    @property
    def figure(self):
        if not self._figure:
            self._build_plot()
        return self._figure

    def _repr_png_(self):
        imgbuffer = BytesIO()
        self.figure.savefig(imgbuffer, format="png")
        return imgbuffer.getvalue()

    def save_png(self, filename):
        self.figure.savefig(filename, format="png")
