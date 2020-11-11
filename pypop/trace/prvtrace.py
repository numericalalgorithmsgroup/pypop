#!/usr/bin/env python3

from hashlib import sha1
from pkg_resources import resource_filename
from warnings import warn

from os import makedirs
from os.path import splitext, basename, join as path_join

import numpy
import pandas

from .trace import Trace

from ..utils.io import zipopen
from ..utils.exceptions import WrongLoaderError, ExtraePRVNoOnOffEventsError

from ..dimemas import dimemas_idealise
from ..extrae import paramedir_analyze_any_of, chop_prv_to_roi, remove_trace

base_configs = {
    k: tuple(resource_filename("pypop", w) for w in v)
    for k, v in {
        "Serial Useful Computation": (
            "cfgs/serial_useful_computation.cfg",
            "cfgs/serial_useful_computation_omp_loop.cfg",
            "cfgs/serial_useful_computation_omp_task.cfg",
            "cfgs/serial_useful_computation_no_omp.cfg",
        ),
        "Total Runtime": (
            "cfgs/total_runtime_excl_disabled.cfg",
            "cfgs/total_runtime.cfg",
        ),
        "Useful Instructions": ("cfgs/useful_instructions.cfg",),
        "Useful Cycles": ("cfgs/useful_cycles.cfg",),
    }.items()
}

omp_configs = {
    k: tuple(resource_filename("pypop", w) for w in v)
    for k, v in {
        "OpenMP Total Runtime": (
            "cfgs/omp_total_runtime.cfg",
            "cfgs/omp_total_runtime_loop.cfg",
            "cfgs/omp_total_runtime_loop.cfg",
        ),
        "OpenMP Useful Computation": (
            "cfgs/omp_useful_computation.cfg",
            "cfgs/omp_useful_computation_loop.cfg",
            "cfgs/omp_useful_computation_task.cfg",
        ),
    }.items()
}

ideal_configs = {
    k: tuple(resource_filename("pypop", w) for w in v)
    for k, v in {
        "Ideal Useful Computation": ("cfgs/total_useful_computation.cfg",),
        "Ideal Runtime": ("cfgs/total_runtime.cfg",),
    }.items()
}


class PRVTrace(Trace):
    def _gather_metadata(self):

        if ".sim." in self._tracefile:
            warn(
                "Filename {} suggests this trace has already been idealised. This will "
                "likely cause the PyPOP analysis to fail!".format(self._tracefile)
            )

        if ".chop" in self._tracefile:
            warn(
                "Filename {} suggests this trace has been chopped before analysis. In "
                "some cases this can cause Dimemas idealisation to fail. It is "
                "recommended to use the trace chopping support built into PyPOP (see "
                "the documentation for more details)".format(self._tracefile)
            )

        try:
            with zipopen(self._tracefile, "rt") as fh:
                headerline = fh.readline().strip()
        except IsADirectoryError:
            raise WrongLoaderError("Not a valid prv file")

        if not headerline.startswith("#Paraver"):
            raise WrongLoaderError("Not a valid prv file")

        elem = headerline.replace(":", ";", 1).split(":", 4)

        self._metadata.capture_time = elem[0][
            elem[0].find("(") + 1 : elem[0].find(")")
        ].replace(";", ":")

        self._metadata.elapsed_seconds = float(elem[1][:-3]) * 1e-9

        self._metadata.num_nodes, self._metadata.cores_per_node = self._split_nodestring(
            elem[2]
        )

        # elem 3 is the number of applications, currently only support 1
        if int(elem[3]) != 1:
            raise ValueError("Multi-application traces are not supported")

        # elem 4 contains the application layout processes/threads
        (
            self._metadata.num_processes,
            self._metadata.threads_per_process,
        ) = self._split_layoutstring(elem[4])

        self._metadata.tracefile_name = self._tracefile

        hasher = sha1()
        hasher.update(headerline.encode())
        self._metadata.fingerprint = hasher.hexdigest()

    def _gather_statistics(self):
        self._statistics = self._analyze_tracefile(
            self._tracefile,
            self._kwargs.get("chop_to_roi", False),
            self._kwargs.get("outpath", None),
            self._kwargs.get("chop_fail_is_error", False)
        )

    @staticmethod
    def _split_nodestring(prv_td):
        num_nodes, plist = prv_td.split("(", 2)
        num_nodes = int(num_nodes)
        cores_per_node = tuple(int(x) for x in plist.rstrip(",)").split(","))

        return (num_nodes, cores_per_node)

    @staticmethod
    def _split_layoutstring(prv_td):

        prv_td = prv_td.split(")")[0].strip()
        commsize, layoutstring = prv_td.split("(")

        commsize = int(commsize)
        threads = [int(x.split(":")[0]) for x in layoutstring.split(",")]

        return (commsize, threads)

    @staticmethod
    def _analyze_tracefile(trace, chop_to_roi, outpath, chop_fail_is_error):
        if outpath:
            makedirs(outpath, exist_ok=True)

        try:
            if chop_to_roi:
                if outpath:
                    tgtname = ".chop".join(splitext(basename(trace)))
                    outfile = path_join(outpath, tgtname)
                else:
                    outfile = None
                cut_trace = chop_prv_to_roi(trace, outfile)
            else:
                cut_trace = trace
        except ExtraePRVNoOnOffEventsError as err:
            if chop_fail_is_error:
                raise err
            warn("Unable to chop to ROI: ({}) - Continuing without chopping".format(err))
            cut_trace = trace
            chop_to_roi = False

        stats = [
            paramedir_analyze_any_of(
                cut_trace, cfg, index_by_thread=True, statistic_names=[name]
            )
            for name, cfg in base_configs.items()
        ]

        try:
            omp_stats = [
                paramedir_analyze_any_of(
                    cut_trace, cfg, index_by_thread=True, statistic_names=[name]
                )
                for name, cfg in omp_configs.items()
            ]
            stats += omp_stats
        except RuntimeError:
            skel = next(iter(stats))
            zero_df = pandas.DataFrame(index=skel.T.index)
            for name in omp_configs:
                zero_df[name] = 0
            stats.append(zero_df.T)

        # Remember to clean up after ourselves
        if chop_to_roi and not outpath:
            remove_trace(cut_trace)

        try:
            ideal_trace = dimemas_idealise(trace, outpath)
            if chop_to_roi:
                if outpath:
                    tgtname = ".chop".join(splitext(basename(ideal_trace)))
                    outfile = path_join(outpath, tgtname)
                else:
                    outpath = None
                cut_ideal_trace = chop_prv_to_roi(ideal_trace, outfile)
            else:
                cut_ideal_trace = ideal_trace

            stats.extend(
                [
                    paramedir_analyze_any_of(
                        cut_ideal_trace,
                        cfg,
                        index_by_thread=True,
                        statistic_names=[name],
                    )
                    for name, cfg in ideal_configs.items()
                ]
            )

            # Keeping things tidy
            if not outpath:
                if chop_to_roi:
                    remove_trace(cut_ideal_trace)
                remove_trace(ideal_trace)
        except RuntimeError as err:
            warn(
                "Failed to run Dimemas: {}\n"
                "Continuing with reduced MPI detail.".format(err)
            )
            # Get an object with the correct layout
            skel = next(iter(stats))
            nan_df = pandas.DataFrame(index=skel.T.index)
            for name in ideal_configs:
                nan_df[name] = numpy.nan

            stats.append(nan_df.T)

        stats = pandas.concat(stats).T
        stats["IPC"] = stats["Useful Instructions"] / stats["Useful Cycles"]

        stats["Total Useful Computation"] = stats["Serial Useful Computation"]
        stats["Total Non-MPI Runtime"] = stats["Serial Useful Computation"]

        stats["Total Useful Computation"] += stats["OpenMP Useful Computation"]
        stats["Total Non-MPI Runtime"] += stats["OpenMP Total Runtime"]

        stats["Frequency"] = stats["Useful Cycles"] / stats["Total Useful Computation"]

        if (
            stats["Total Non-MPI Runtime"].loc[:, 1].max()
            > stats["Ideal Runtime"].loc[:, 1].max()
        ):
            warn(
                "Dimemas has provided an invalid Ideal Runtime value (less than Useful "
                "Computation)\ntracefile:{}\nIR:{}\nUC:{}".format(
                    trace,
                    stats["Total Non-MPI Runtime"].loc[:, 1].max(),
                    stats["Ideal Runtime"].loc[:, 1].max(),
                )
            )

        return stats


PRVTrace.register_loader()
