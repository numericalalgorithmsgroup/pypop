#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from bokeh.palettes import diverging_palette, all_palettes

__all__ = ["efficiency_red_green", "inefficiency_green_red"]

k_breakover_threshold = 0.8


def build_efficiency_red_green():

    return diverging_palette(
        all_palettes["Reds"][256],
        all_palettes["Greens"][256],
        256,
        k_breakover_threshold,
    )


def build_inefficiency_green_red():

    return diverging_palette(
        all_palettes["Greens"][256],
        all_palettes["Reds"][256],
        256,
        1 - k_breakover_threshold,
    )


efficiency_red_green = build_efficiency_red_green()
inefficiency_green_red = build_inefficiency_green_red()
