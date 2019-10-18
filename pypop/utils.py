#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Miscellaneous utility routines
"""

from hashlib import md5


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
