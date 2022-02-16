#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""OpenMP only metrics
"""

import pandas

from .metricset import MetricSet, Metric

__all__ = ["OpenMP_Metrics"]

k_GE_desc = (
    "The overall quality of the parallelisation.  This is the product of the "
    "Parallel Efficiency and Computational Scaling"
)
k_PE_desc = (
    "The overall efficiency with which the computation is parallelised between "
    "different threads. This is further divided into the OpenMP Region "
    "Efficiency and the Serial Region Efficiency"
)
k_OMPRE_desc = (
    "The efficiency with which CPU resources are used within OpenMP regions. "
    "This includes the effect of any load imbalance within OpenMP regions "
    "as well as any scheduling overhead and critical or single regions."
)
k_SERRE_desc = (
    "The effects of loss of efficiency due to computation being performed outside of "
    "OpenMP Regions.  This is calculated by comparing with the effective speedup that "
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
k_FREQSC_desc = (
    "Inefficiencies due to processor execution frequency changes. The frequency is typically "
    "reduced due to either incorrect choice of CPU scheduler or overheating of the CPU."
)


class OpenMP_Metrics(MetricSet):
    """Proposed Hybrid MPI+OpenMP Metrics."""

    _metric_list = [
        Metric("Global Efficiency", 0, desc=k_GE_desc),
        Metric("Parallel Efficiency", 1, desc=k_PE_desc),
        Metric("OpenMP Region Efficiency", 2, desc=k_OMPRE_desc),
        Metric("Serial Region Efficiency", 2, desc=k_SERRE_desc),
        Metric("Computational Scaling", 1, desc=k_COMPSC_desc),
        Metric("Instruction Scaling", 2, desc=k_INSSC_desc),
        Metric("IPC Scaling", 2, "IPC Scaling", desc=k_IPCSC_desc),
        Metric("Frequency Scaling", 2, "Frequency Scaling", desc=k_FREQSC_desc),
    ]

    _programming_model = "OpenMP"

    _default_metric_key = "Total Threads"

    def _calculate_metrics(self, ref_key=None, sort_keys=True):
        if not ref_key:
            ref_key = min(self._stats_dict.keys())

        metrics_by_key = {}

        if sort_keys:
            keys = sorted(self._stats_dict.keys())
        else:
            key = self._stats_dict.keys()

        for key in keys:
            metadata = self._stats_dict[key].metadata
            stats = self._stats_dict[key].statistics
            nthreads = metadata.threads_per_process[0]
            metrics = self._create_subdataframe(metadata, key)

            try:

                metrics["OpenMP Region Efficiency"] = 1 - (
                    (
                        (
                            stats["OpenMP Total Runtime"].loc[:, 1]
                            - stats["OpenMP Useful Computation"]
                            .groupby(level="rank")
                            .mean()
                        ).mean()
                    )
                    / stats["Total Runtime"].max()
                )

                metrics["Serial Region Efficiency"] = 1 - (
                    stats["Serial Useful Computation"].loc[:, 1].mean()
                    / stats["Total Runtime"].max()
                    * (1 - 1 / nthreads)
                )

                metrics["Parallel Efficiency"] = (
                    stats["Total Useful Computation"].mean()
                    / stats["Total Runtime"].max()  # avg all threads to include Amdahl
                )

                metrics["IPC Scaling"] = (
                    stats["Useful Instructions"].sum() / stats["Useful Cycles"].sum()
                ) / (
                    self._stats_dict[ref_key].statistics["Useful Instructions"].sum()
                    / self._stats_dict[ref_key].statistics["Useful Cycles"].sum()
                )

                metrics["Instruction Scaling"] = (
                    self._stats_dict[ref_key].statistics["Useful Instructions"].sum()
                    / stats["Useful Instructions"].sum()
                )

                metrics["Frequency Scaling"] = (
                    stats["Useful Cycles"].sum()
                    / stats["Total Useful Computation"].sum()
                ) / (
                    self._stats_dict[ref_key].statistics["Useful Cycles"].sum()
                    / self._stats_dict[ref_key]
                    .statistics["Total Useful Computation"]
                    .sum()
                )

                metrics["Computational Scaling"] = (
                    self._stats_dict[ref_key]
                    .statistics["Total Useful Computation"]
                    .sum()
                    / stats["Total Useful Computation"].sum()
                )

                metrics["Frequency Scaling"] = (
                    stats["Useful Cycles"].sum()
                    / stats["Total Useful Computation"].sum()
                ) / (
                    self._stats_dict[ref_key].statistics["Useful Cycles"].sum()
                    / self._stats_dict[ref_key]
                    .statistics["Total Useful Computation"]
                    .sum()
                )

                metrics["Global Efficiency"] = (
                    metrics["Computational Scaling"] * metrics["Parallel Efficiency"]
                )

                metrics["Speedup"] = (
                    self._stats_dict[ref_key].statistics["Total Runtime"].max()
                    / stats["Total Runtime"].max()
                )

                metrics["Runtime"] = stats["Total Runtime"].max()

            except KeyError as err:
                raise ValueError(
                    "No '{}' statistic. (Wrong analysis type?)" "".format(err.args[0])
                )

            metrics_by_key[key] = metrics

        self._metric_data = pandas.concat(metrics_by_key.values())
