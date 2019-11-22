#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from warnings import warn

import pandas

from pkg_resources import resource_filename

from ..dimemas import dimemas_idealise
from ..extrae import paramedir_analyze, chop_prv_to_roi, remove_trace
from ..prv import get_prv_header_info

from .tracestats import TraceStats, TraceStatsMetadata

base_configs = {
    k: resource_filename(__name__, v)
    for k, v in {
        "Serial Useful Computation": "paramedircfg/serial_useful_computation.cfg",
        "Total Runtime": "paramedircfg/total_runtime.cfg",
        "Useful Instructions": "paramedircfg/useful_instructions.cfg",
        "Useful Cycles": "paramedircfg/useful_cycles.cfg",
    }.items()
}

omp_configs = {
    k: resource_filename(__name__, v)
    for k, v in {
        "OpenMP Total Runtime": "paramedircfg/omp_total_runtime.cfg",
        "OpenMP Useful Computation": "paramedircfg/omp_useful_computation.cfg",
    }.items()
}

ideal_configs = {
    k: resource_filename(__name__, v)
    for k, v in {
        "Ideal Useful Computation": "paramedircfg/total_useful_computation.cfg",
        "Ideal Runtime": "paramedircfg/total_runtime.cfg",
    }.items()
}


class PRVStats(TraceStats):
    """Summary statistics for PRV traces

    """

    _extra_metadata_keys = ['chop_to_roi']

    @staticmethod
    def can_handle(tracefile):
        if tracefile.endswith(".prv") or tracefile.endswith(".prv.gz"):
            return True

        return False

    def _build_metadata(self, tracefile, **kwargs):

        prvmetadata = get_prv_header_info(tracefile)

        metadata_extra = {
            key: kwargs[key] for key in PRVStats._extra_metadata_keys if key in kwargs
        }

        metadata = TraceStatsMetadata(
            tracefile,
            prvmetadata.nodes,
            prvmetadata.application_layout.commsize,
            prvmetadata.application_layout.rank_threads[0][1],
            prvmetadata.ns_elapsed * 1e-9,
            **metadata_extra
        )

        return metadata

    def _analyze_tracefile(self, tracefile, **kwargs):
        try:
            chop_to_roi = bool(kwargs["chop_to_roi"])
        except KeyError:
            chop_to_roi = False

        if chop_to_roi:
            cut_trace = chop_prv_to_roi(tracefile)
        else:
            cut_trace = tracefile

        stats = [
            paramedir_analyze(
                cut_trace, cfg, index_by_thread=True, statistic_names=[name]
            )
            for name, cfg in base_configs.items()
        ]

        hybrid = False
        try:
            omp_stats = [
                paramedir_analyze(
                    cut_trace, cfg, index_by_thread=True, statistic_names=[name]
                )
                for name, cfg in omp_configs.items()
            ]
            stats += omp_stats
            hybrid = True
        except RuntimeError:
            pass

        # Remember to clean up after ourselves
        if chop_to_roi:
            remove_trace(cut_trace)

        try:
            ideal_trace = dimemas_idealise(tracefile)
            if chop_to_roi:
                cut_ideal_trace = chop_prv_to_roi(ideal_trace)
            else:
                cut_ideal_trace = ideal_trace
            stats.extend(
                [
                    paramedir_analyze(
                        cut_ideal_trace,
                        cfg,
                        index_by_thread=True,
                        statistic_names=[name],
                    )
                    for name, cfg in ideal_configs.items()
                ]
            )
            # Keeping things tidy
            if chop_to_roi:
                remove_trace(cut_ideal_trace)
            remove_trace(ideal_trace)
        except RuntimeError as err:
            warn(
                "Failed to run Dimemas:\n{}"
                "Continuing with reduced MPI detail.".format(err)
            )

        stats = pandas.concat(stats).T
        stats["IPC"] = stats["Useful Instructions"] / stats["Useful Cycles"]

        if hybrid:
            stats["Serial Useful Computation"].loc[:, 2:] = 0

        stats["Total Useful Computation"] = stats["Serial Useful Computation"]
        stats["Total Non-MPI Runtime"] = stats["Serial Useful Computation"]

        if hybrid:
            stats["Total Useful Computation"] += stats["OpenMP Useful Computation"]
            stats["Total Non-MPI Runtime"] += stats["OpenMP Total Runtime"]

        stats["Frequency"] = stats["Useful Cycles"] / stats["Total Useful Computation"]

        return stats


TraceStats.register_handler(PRVStats)
