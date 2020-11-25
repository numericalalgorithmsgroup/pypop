#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Pure MPI metrics
"""

import numpy
import pandas

from .metricset import MetricSet, Metric

k_GE_desc = (
    "The overall quality of the parallelisation.  This is the product of the "
    "Parallel Efficiency and Computational Scaling"
)
k_PE_desc = (
    "The overall efficiency with which the computation is parallelised between "
    "different processes and threads. This is further divided into the Process Level"
    "Efficiency and the Thread Level Efficiency"
)
k_PLE_desc = (
    "The efficiency of the application as viewed at the process level, including the "
    "MPI communication and process-level load balance."
)
k_MPILB_desc = (
    "The efficiency with which the total amount of computational work is shared "
    "between the different MPI processes. Low values indicate that there is significant "
    "imbalance between the most and least loaded processes."
)
k_MPICE_desc = (
    "The efficiency with which the application carries out MPI communication.  An ideal "
    "application will spend no time communicating and 100% of time in computation. Low "
    "values indicate that too much communication is being performed for the amount of "
    "computation."
)
k_MPITE_desc = (
    "The efficiency of the actual transfer of data via MPI. This reflects the size of "
    "the data being communicated and the speed of the underlying communication network. "
    "Low values indicate that the network bandwidth is insufficient for the required "
    "communication rate, or that too much data is being communicated."
)
k_MPISE_desc = (
    "The efficiency with which the MPI communications are synchronised and carried out. "
    "Low values indicate that there is significant irregularity in the timings of "
    "different processes arrivals at MPI calls, reducing efficiency due to waiting "
    "for MPI calls to complete."
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
    "Inefficiencies due to changes in the rate at which the CPU executes instructions. "
    "This is typically due to thermal management in the CPU reducing the overall clock "
    "speed."
)


class MPI_Metrics(MetricSet):
    """Pure MPI Metrics  (additive version).
    """

    _metric_list = [
        Metric("Global Efficiency", 0, desc=k_GE_desc),
        Metric("Parallel Efficiency", 1, desc=k_PE_desc),
        Metric("MPI Load Balance", 2, "Load balance", desc=k_MPILB_desc),
        Metric("MPI Communication Efficiency", 2, desc=k_MPICE_desc),
        Metric("MPI Transfer Efficiency", 3, desc=k_MPITE_desc),
        Metric("MPI Serialisation Efficiency", 3, desc=k_MPISE_desc),
        Metric("Computation Scaling", 1, desc=k_COMPSC_desc),
        Metric("Instruction Scaling", 2, desc=k_INSSC_desc),
        Metric("IPC Scaling", 2, desc=k_IPCSC_desc),
        Metric("Frequency Scaling", 2, desc=k_FREQSC_desc),
    ]

    _programming_model = "MPI"

    def _calculate_metrics(self, ref_key=None, sort_keys=True):

        if not ref_key:
            ref_key = min(self._stats_dict.keys())

        metrics_by_key = {}

        if sort_keys:
            keys = sorted(self._stats_dict.keys())
        else:
            key = self._stats_dict.keys()

        for idx, key in enumerate(keys):
            metadata = self._stats_dict[key].metadata
            stats = self._stats_dict[key].statistics
            metrics = self._create_subdataframe(metadata, key)

            metrics["MPI Communication Efficiency"] = (
                stats["Total Non-MPI Runtime"].loc[:, 1].max()
                / stats["Total Runtime"].max()
            )

            try:
                metrics["MPI Serialisation Efficiency"] = 1 - (
                    (
                        stats["Ideal Runtime"].loc[:, 1].max()
                        - stats["Total Non-MPI Runtime"].loc[:, 1].max()
                    )
                    / stats["Total Runtime"].max()
                )
            except KeyError:
                metrics["MPI Serialisation Efficiency"] = numpy.nan

            try:
                metrics["MPI Transfer Efficiency"] = (
                    stats["Ideal Runtime"].loc[:, 1].max()
                    / stats["Total Runtime"].loc[:, 1].max()
                )
            except KeyError:
                metrics["MPI Transfer Efficiency"] = numpy.nan

            metrics["MPI Load Balance"] = 1 - (
                (
                    stats["Total Useful Computation"].loc[:, 1].max()
                    - stats["Total Useful Computation"].loc[:, 1].mean()
                )
                / stats["Total Runtime"].max()
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
                stats["Useful Cycles"].sum() / stats["Total Useful Computation"].sum()
            ) / (
                self._stats_dict[ref_key].statistics["Useful Cycles"].sum()
                / self._stats_dict[ref_key].statistics["Total Useful Computation"].sum()
            )

            metrics["Computation Scaling"] = (
                self._stats_dict[ref_key].statistics["Total Useful Computation"].sum()
                / stats["Total Useful Computation"].sum()
            )

            metrics["Global Efficiency"] = (
                metrics["Computation Scaling"] * metrics["Parallel Efficiency"]
            )

            metrics["Speedup"] = (
                self._stats_dict[ref_key].statistics["Total Runtime"].max()
                / stats["Total Runtime"].max()
            )

            metrics["Runtime"] = stats["Total Runtime"].max()

            metrics_by_key[key] = metrics

        self._metric_data = pandas.concat(metrics_by_key.values())


class MPI_Multiplicative_Metrics(MetricSet):
    """Pure MPI Metrics (multiplicative version).
    """

    _metric_list = [
        Metric("Global Efficiency", 0),
        Metric("Parallel Efficiency", 1),
        Metric("MPI Load balance", 2, "Load balance"),
        Metric("MPI Communication Efficiency", 2),
        Metric("MPI Transfer Efficiency", 3),
        Metric("MPI Serialisation Efficiency", 3),
        Metric("Computation Scaling", 1),
        Metric("Instruction Scaling", 2),
        Metric("IPC Scaling", 2),
        Metric("Frequency Scaling", 2),
    ]

    _programming_model = "MPI"

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
            metrics = self._create_subdataframe(metadata, key)

            metrics["MPI Communication Efficiency"] = (
                stats["Total Non-MPI Runtime"].loc[:, 1].max()
                / stats["Total Runtime"].max()
            )

            try:
                metrics["MPI Serialisation Efficiency"] = (
                    stats["Total Non-MPI Runtime"].loc[:, 1].max()
                    / stats["Ideal Runtime"].loc[:, 1].max()
                )
            except KeyError:
                metrics["MPI Serialisation Efficiency"] = numpy.nan

            try:
                metrics["MPI Transfer Efficiency"] = (
                    stats["Ideal Runtime"].loc[:, 1].max()
                    / stats["Total Runtime"].loc[:, 1].max()
                )
            except KeyError:
                metrics["MPI Transfer Efficiency"] = numpy.nan

            metrics["MPI Load balance"] = (
                stats["Total Useful Computation"].loc[:, 1].mean()
                / stats["Total Useful Computation"].loc[:, 1].max()
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
                stats["Useful Cycles"].sum() / stats["Total Useful Computation"].sum()
            ) / (
                self._stats_dict[ref_key].statistics["Useful Cycles"].sum()
                / self._stats_dict[ref_key].statistics["Total Useful Computation"].sum()
            )

            metrics["Computation Scaling"] = (
                self._stats_dict[ref_key].statistics["Total Useful Computation"].sum()
                / stats["Total Useful Computation"].sum()
            )

            metrics["Global Efficiency"] = (
                metrics["Computation Scaling"] * metrics["Parallel Efficiency"]
            )

            metrics["Speedup"] = (
                self._stats_dict[ref_key].statistics["Total Runtime"].max()
                / stats["Total Runtime"].max()
            )

            metrics["Runtime"] = stats["Total Runtime"].max()

            metrics_by_key[key] = metrics

        self._metric_data = pandas.concat(metrics_by_key.values())
