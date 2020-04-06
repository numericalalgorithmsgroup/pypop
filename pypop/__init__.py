#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Tools for POP assessments in Python scripts and IPython notebooks
"""

from . import traceset
from . import metrics
from . import prv
from . import config

from .version import get_version

__all__ = ["__version__"]

__version__ = get_version()
