#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""\
Dimemas Automation
------------------

Helper routines to run Dimemas from Python, including the special case of
ideal trace generation.
"""

import os
import shutil
import warnings
from os.path import basename, splitext
from tempfile import mkdtemp
import subprocess as sp

from pkg_resources import resource_filename

from .prv import PRV
from .utils.io import zipopen
from .extrae import remove_trace
from . import config

IDEAL_CONF_PATH = resource_filename(__name__, "cfgs/dimemas_ideal.cfg")
IDEAL_COLL_PATH = resource_filename(__name__, "cfgs/ideal.collectives")


def dimemas_idealise(tracefile, outpath=None):
    """Idealise a tracefile using Dimemas

    Parameters
    ----------
    tracefile: str
        Path to Extrae tracefile (`*.prv`)
    outpath: str
        Optional path to output idealised trace. (If not specified will be
        created in a temporary folder.)

    Returns
    -------
    idealised: str
        Path to idealised tracefile in prv format.
    """

    # First we need the application layout info
    metadata = PRV(tracefile, lazy_load=True).metadata

    # Populate run specfic config data
    subs = {
        "@NUM_NODES@": metadata.num_processes,
        "@PROCS_PER_NODE@": max(metadata.cores_per_node),
        "@RANKS_PER_NODE@": 1,
        "@COLLECTIVES_PATH@": IDEAL_COLL_PATH,
    }

    # Pass trace, run config and path to idealisation skeleton config and let
    # dimemas_analyze work its subtle magic(k)s
    return dimemas_analyse(tracefile, IDEAL_CONF_PATH, outpath, subs)


def dimemas_analyse(tracefile, configfile, outpath=None, substrings=None):
    """Run a Dimemas simulation given a configuration and tracefile

    The configuration file may be modifed with key->value substitutions prior
    to running dimemas using the substrings parameter. This allows the use of a
    predefined skeleton configuration file rather than requiring programmatic
    production of the complete config.

    Parameters
    ----------
    tracefile: str
        Path to Extrae tracefile (`*.prv`)
    configfile: str
        Path to Dimemas configfile
    outpath: str or None
        Optional path to output idealised trace. (If not specified, will be
        created in a temporary folder.)
    substrings: dict
        Dict of keys in the config file to be replaced with corresponding
        values.

    Returns
    -------
    simulated: str
        Path to simulated tracefile in prv format.
    """

    # Perform all work in a tempdir with predictable names,
    # this works around a series of weird dimemas bugs

    # Make sure config._tmpdir_path exists before using it
    if config._tmpdir_path:
        os.makedirs(config._tmpdir_path, exist_ok=True)
        workdir = mkdtemp(dir=config._tmpdir_path)
    else:
        workdir = mkdtemp()

    # Create temporary config from supplied config and substitution dict
    dimconfig = os.path.join(workdir, ".tmpconfig".join(splitext(basename(configfile))))

    with open(configfile, "rt") as ifh, open(dimconfig, "wt") as ofh:
        for line in ifh:
            if substrings:
                for key, val in substrings.items():
                    line = line.replace(key, str(val))
            ofh.write(line)

    # Now copy temporary prv file:
    tmp_prv = os.path.join(workdir, "input.prv")
    with zipopen(tracefile, "rb") as ifh, open(tmp_prv, "wb") as ofh:
        while True:
            buff = ifh.read(536870912)
            if not buff:
                break
            ofh.write(buff)

    # And also copy row and pcf if available
    for ext in [".row", ".pcf"]:
        tracestem = tracefile[:-3] if tracefile.endswith(".gz") else tracefile
        infile = splitext(tracestem)[0] + ext
        outfile = splitext(tmp_prv)[0] + ext
        try:
            with open(infile, "rb") as ifh, open(outfile, "wb") as ofh:
                while True:
                    buff = ifh.read(536870912)
                    if not buff:
                        break
                    ofh.write(buff)
        except FileNotFoundError:
            warnings.warn(
                "Could not find {}, dimemas may fail or produce invalid data"
                "".format(infile)
            )

    # Now create the dim file for dimemas
    tmp_dim = os.path.join(workdir, splitext(basename(tmp_prv))[0] + ".dim")

    prv2dim_binpath = "prv2dim"
    if config._dimemas_path:
        prv2dim_binpath = os.path.join(config._dimemas_path, prv2dim_binpath)

    # Use basenames as running in workdir
    prv2dim_params = [
        prv2dim_binpath,
        "--prv-trace",
        basename(tmp_prv),
        "--dim-trace",
        basename(tmp_dim),
    ]

    # Run prv2dim and check for success
    try:
        result = sp.run(prv2dim_params, stdout=sp.PIPE, stderr=sp.PIPE, cwd=workdir)
        if not os.path.exists(tmp_dim) or result.returncode != 0:
            raise RuntimeError(
                "prv2dim execution failed:\n{}" "".format(result.stderr.decode())
            )
    except Exception as err:
        raise RuntimeError("prv2dim execution_failed: {}".format(err))

    dimemas_binpath = "Dimemas"
    if config._dimemas_path:
        dimemas_binpath = os.path.join(config._dimemas_path, dimemas_binpath)

    # Run in workdir to workaround Dimemas path bug, output to relpath
    sim_prv = ".sim".join(splitext(tmp_prv))
    dimemas_params = [
        dimemas_binpath,
        "-S 32k",
        "--dim",
        basename(tmp_dim),
        "--prv-trace",
        basename(sim_prv),
        "--config-file",
        basename(dimconfig),
    ]

    try:
        result = sp.run(dimemas_params, stdout=sp.PIPE, stderr=sp.STDOUT, cwd=workdir)
        if not os.path.exists(sim_prv) or result.returncode != 0:
            raise RuntimeError(
                "Dimemas execution failed:\n{}\n"
                "Input was:\n{}\n"
                "See {}"
                "".format(result.stdout.decode(), dimemas_params, workdir)
            )
    except Exception as err:
        raise RuntimeError("dimemas execution failed:\n{}".format(err))

    # remove all the temporary files we created
    os.remove(dimconfig)
    os.remove(tmp_dim)
    remove_trace(tmp_prv)

    # If no outpath specified then we are done
    if outpath is None:
        return sim_prv

    # Otherwise copy back to requested location
    filestem = basename(splitext(tracefile)[0])
    outfile = os.path.join(outpath, filestem + ".sim.prv")
    with open(sim_prv, "rb") as ifh, open(outfile, "wb") as ofh:
        while True:
            buff = ifh.read(536870912)
            if not buff:
                break
            ofh.write(buff)

    # And also copy row and pcf
    for ext in [".row", ".pcf"]:
        infile = splitext(sim_prv)[0] + ext
        outfile = os.path.join(outpath, filestem + ".sim" + ext)
        with open(infile, "rb") as ifh, open(outfile, "wb") as ofh:
            while True:
                buff = ifh.read(536870912)
                if not buff:
                    break
                ofh.write(buff)

    # and then delete temps
    remove_trace(sim_prv)
    shutil.rmtree(workdir, ignore_errors=True)

    # finally return outpath as promised
    return os.path.join(outpath, filestem + ".sim.prv")
