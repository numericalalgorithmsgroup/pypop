#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Configuration options for PyPOP
"""

from os.path import normpath, expanduser

__all__ = ["set_paramedir_path", "set_dimemas_path", "set_tmpdir_path"]

_dimemas_path = None
_paramedir_path = None
_tmpdir_path = None


def set_dimemas_path(path):
    global _dimemas_path
    if not isinstance(path, str):
        raise ValueError("path must be a string")

    _dimemas_path = normpath(expanduser(path))


def set_paramedir_path(path):
    global _paramedir_path
    if not isinstance(path, str):
        raise ValueError("path must be a string")

    _paramedir_path = normpath(expanduser(path))


def set_tmpdir_path(path):
    global _tmpdir_path
    if not isinstance(path, str):
        raise ValueError("path must be a string")

    _tmpdir_path = normpath(expanduser(path))
