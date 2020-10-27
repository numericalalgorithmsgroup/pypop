#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

import numpy
from pkg_resources import resource_filename

__all__ = ["approx_string_em_width", "get_pop_logo_data"]

_POP_LOGO_PNG = "pop_logo.npy"


def approx_string_em_width(string):
    """Give the approximate string width in 'em'

    Basically assume 'w' and 'm' have em width and everything else has
    a width of 0.6*em
    """

    return 0.6 * len(string) + 0.4 * sum(string.count(x) for x in ["w", "m", "~"])


def get_pop_logo_data():
    """Return numpy array containing POP logo pixels
    """

    return numpy.load(resource_filename(__name__, _POP_LOGO_PNG))
