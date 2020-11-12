#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""\
Miscellaneous utility routines
------------------------------

General utility and helper routines not specific to a particular tool.
"""

from hashlib import md5
import gzip


def zipopen(path, mode="rt"):

    # Default modes of open() and gzip.open() are not the same ('t' and 'b' respectively)
    # so ensure a consistent default of text mode
    if not any(x in mode for x in ("t", "b")):
        mode = "{}t".format(mode)

    try:
        if gzip.open(path, mode=mode).read(1):
            return gzip.open(path, mode=mode)
    except OSError:
        return open(path, mode=mode)


def chunked_md5sum(filename, blocksize=8388608):
    """ Chunked md5 checksum using hashlib

    Parameters
    ----------
    filename: str
        File to calculate checksum.
    blocksize: int
        Blocksize in bytes to iterate with, (default 8MB).

    Returns
    -------
    hexdigest: str
        String containing hexadecimal representation of the checksum.
    """
    hashobj = md5()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(blocksize), b""):
            hashobj.update(block)
    return hashobj.hexdigest()


def return_first_arg(arg, *args, **kwargs):
    """Return first argument and disregard remaining, intended for silently replacing
    tqdm when unavailable

    Parameters
    ----------
    arg: *
        Argument to be returned

    *args: Will be ignored

    **kwargs: Will be ignored

    Returns
    -------
    arg:
        First argument passed to function
    """

    return arg
