#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

import numpy

from bokeh.plotting import figure
from bokeh.transform import linear_cmap

from pypop.prv import PRV

from .plotting import BokehBase
from .palettes import efficiency_red_green


class OMPRegionExplorer(BokehBase):

    _omp_tooltip_template = [
        ("Function(s)", "@{Region Function Fingerprint}"),
        ("Location(s)", "@{Region Location Fingerprint}"),
        ("Load Bal.", "@{Load Balance}{0.00}"),
        ("Length", "@{Region Length}{0.000 a}s"),
        ("Avg. Comp.", "@{Average Computation Time}{0.000 a}s"),
        ("Max. Comp.", "@{Maximum Computation Time}{0.000 a}s"),
        ("Sum Comp.", "@{Region Total Computation}{0.000 a}s"),
    ]

    def __init__(self, prv: PRV, fontsize=14):
        super().__init__()
        self._prv = prv
        self.fontsize = fontsize

    def _build_plot(self):

        omp_region_stats = self._prv.profile_openmp_regions().copy()

        for tm in [
            "Region Start",
            "Region End",
            "Region Length",
            "Average Computation Time",
            "Maximum Computation Time",
            "Computation Delay Time",
            "Region Total Computation",
            "Region Delay Time",
        ]:
            omp_region_stats[tm] /= 1e9

        # Geometry calculations - start at 48 em width and 2em plus 2em per bar height up
        # to a maximum of 15 bars, then start shrinking the bars
        pt_to_px = 96 / 72
        font_px = self.fontsize * pt_to_px
        width = int(48 * font_px)
        height = int(32 * font_px)

        self._plot_dims = (width, height)

        self._figure = figure(
            plot_width=900,
            plot_height=600,
            tools="xwheel_zoom,zoom_in,zoom_out,pan,reset,save",
            tooltips=self._omp_tooltip_template,
        )

        for rank, rankdata in omp_region_stats.groupby(level="rank"):
            self._figure.hbar(
                y=rank,
                left="Region Start",
                right="Region End",
                height=0.9,
                color=linear_cmap("Load Balance", efficiency_red_green, 0, 1),
                source=rankdata,
            )

        self._figure.outline_line_color = None

        for ax in (self._figure.xaxis, self._figure.yaxis):
            ax.minor_tick_line_color = None

        n_ranks = len(omp_region_stats.index.unique(level="rank"))
        n_rankticks = n_ranks if n_ranks < 10 else 10
        rankticks = [int(x) for x in numpy.linspace(1, n_ranks, n_rankticks)]

        self._figure.yaxis.ticker = rankticks

        self._figure.ygrid.visible = False
        self._figure.yaxis.major_tick_line_color = None
        self._figure.yaxis.axis_line_color = None

        self._figure.xaxis.axis_label = "Time (s)"
        self._figure.xaxis.axis_label_text_font_size = "16pt"
        self._figure.xaxis.major_label_text_font_size = "16pt"
        self._figure.yaxis.axis_label = "Processes"
        self._figure.yaxis.axis_label_text_font_size = "16pt"
        self._figure.yaxis.major_label_text_font_size = "16pt"

        self.update()
