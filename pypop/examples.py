#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Support module for locating the pypop example files
"""

from pkg_resources import resource_filename


def examples_directory():
    """Location of the examples directory

    Returns
    -------
    examples_dir: str
        Path to examples directory
    """
    return resource_filename(__name__, "examples")


if __name__ == "__main__":
    print(examples_directory())
