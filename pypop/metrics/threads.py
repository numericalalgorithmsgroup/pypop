#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Shared Memory threaded metrics
"""

import pandas

from .metricset import MetricSet, Metric

__all__ = ["Thread_Metrics"]

k_GE_desc = (
    "The overall quality of the parallelisation.  This is the product of the "
    "Parallel Efficiency and Computational Scaling"
)
k_PE_desc = (
    "The overall efficiency with which the computation is parallelised between "
    "different threads. This is further divided into the OpenMP Region "
    "Efficiency and the Serial Region Efficiency"
)
k_PARRE_desc = (
    "The efficiency with which CPU resources are used within OpenMP regions. "
    "This includes the effect of any load imbalance within OpenMP regions "
    "as well as any scheduling overhead and critical or single regions."
)
k_SERRE_desc = (
    "The effects of loss of efficiency due to computation being performed outside of "
    "Parallel Regions.  This is calculated by comparing with the effective speedup that "
    "would be achieved if the serial computation were perfectly parallelised over all "
    "available threads."
)
k_COMPSC_desc = (
    "The way in which the total computational cost varies with the applied parallelism. "
    "This is a combination of the increased cost due to additional calculations "
    "performed, and increased costs due to reduced instructions per cycle."
)
k_INSSC_desc = (
    "Inefficiencies introduced due to an increase in the total computational work done, "
    "measured by the total CPU instructions. Ideally, there would be no additional "
    "computation required when parallelising, but there is normally some additional "
    "cost to manage the distribution of work. The Instruction Scaling metric "
    "represents this by calculating the relative difference in total instructions "
    "between runs."
)
k_IPCSC_desc = (
    "Inefficiencies due to changes in the instructions per cycle executed by the CPUs. "
    "The IPC rate can be reduced due to CPU data starvation, inefficient cache usage or "
    "high rates of branch misprediction."
)


class Thread_Metrics(MetricSet):
    """Proposed Hybrid MPI+OpenMP Metrics.
    """

    _required_stats = [
        "Serial Useful",
        "Total Runtime",
        "Useful Instructions",
        "Useful Cycles",
        "Parallel Region Time",
        "Parallel Useful",
    ]

    _metric_list = [
        Metric("Global Efficiency", 0, desc=k_GE_desc),
        Metric("Parallel Efficiency", 1, desc=k_PE_desc),
        Metric("Parallel Region Efficiency", 2, desc=k_PARRE_desc),
        Metric("Serial Region Efficiency", 2, desc=k_SERRE_desc),
        Metric("Computational Scaling", 1, desc=k_COMPSC_desc),
        Metric("Instruction Scaling", 2, desc=k_INSSC_desc),
        Metric("IPC Scaling", 2, "IPC Scaling", desc=k_IPCSC_desc),
    ]

    def _calculate_metrics(self, ref_key=None, sort_keys=True):

        total_useful = {
            k: self._stats_dict[k].stats["Serial Useful"]
            + self._stats_dict[k].stats["Parallel Useful"]
            for k in self._stats_dict
        }

        ipc = {
            k: self._stats_dict[k].stats["Useful Instructions"]
            / self._stats_dict[k].stats["Useful Cycles"]
            for k in self._stats_dict
        }

        frequency = {
            k: self._stats_dict[k].stats["Useful Cycles"] / total_useful[k]
            for k in self._stats_dict
        }

        if not ref_key:
            ref_key = min(self._stats_dict.keys())

        metrics_by_key = {}

        if sort_keys:
            keys = sorted(self._stats_dict.keys())
        else:
            keys = self._stats_dict.keys()

        for curr_key in keys:
            metadata = self._stats_dict[curr_key].metadata
            stats = self._stats_dict[curr_key].stats
            nthreads = metadata.application_layout.rank_threads[0][0]
            metrics = {
                "Number of Processes": metadata.application_layout.commsize,
                "Threads per Process": metadata.application_layout.rank_threads[0][0],
                "Total Threads": sum(
                    x[0] for x in metadata.application_layout.rank_threads
                ),
            }
            try:

                metrics["Parallel Region Efficiency"] = 1 - (
                    (
                        (
                            stats["Parallel Region Time"].loc[:, 1]
                            - stats["Parallel Useful"].mean(level="process")
                        ).mean()
                    )
                    / stats["Total Runtime"].max()
                )

                metrics["Serial Region Efficiency"] = 1 - (
                    stats["Serial Useful"].loc[:, 1].mean()
                    / stats["Total Runtime"].max()
                    * (1 - 1 / nthreads)
                )

                metrics["Parallel Efficiency"] = (
                    total_useful[curr_key].mean()
                    / stats["Total Runtime"].max()  # avg all threads to include Amdahl
                )

                metrics["IPC Scaling"] = ipc[curr_key].mean() / ipc[ref_key].mean()

                metrics["Instruction Scaling"] = (
                    self._stats_dict[ref_key].stats["Useful Instructions"].sum()
                    / stats["Useful Instructions"].sum()
                )

                metrics["Frequency"] = frequency[curr_key].loc[1, 1]

                metrics["Frequency Scaling"] = (
                    frequency[curr_key].mean()
                    / frequency[ref_key].mean()
                )

                metrics["Computational Scaling"] = (
                    total_useful[ref_key].sum() / total_useful[curr_key].sum()
                )

                metrics["Global Efficiency"] = (
                    metrics["Computational Scaling"] * metrics["Parallel Efficiency"]
                )

                metrics["Speedup"] = (
                    self._stats_dict[ref_key].stats["Total Runtime"].max()
                    / stats["Total Runtime"].max()
                )

                metrics["Runtime"] = stats["Total Runtime"].max()

            except KeyError as err:
                raise ValueError(
                    "No '{}' statistic. (Wrong analysis type?)" "".format(err.args[0])
                )

            metrics_by_key[curr_key] = metrics

        self._metric_data = pandas.DataFrame(metrics_by_key).T
