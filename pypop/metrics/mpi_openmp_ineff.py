#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Hybrid MPI_OpenMP Inefficiency metrics
"""

import numpy
import pandas

from .metricset import MetricSet, Metric

__all__ = [
    "MPI_OpenMP_Ineff_Metrics",
]


class MPI_OpenMP_Ineff_Metrics(MetricSet):
    """Proposed Hybrid MPI+OpenMP Inefficiency Metrics (additive version)."""

    _metric_list = [
        Metric("Global Inefficiency", 0, is_inefficiency=True),
        Metric("Parallel Inefficiency", 1, is_inefficiency=True),
        Metric("Process Level Inefficiency", 2, is_inefficiency=True),
        Metric("MPI Load Balance Inefficiency", 3, is_inefficiency=True),
        Metric("MPI Communication Inefficiency", 3, is_inefficiency=True),
        Metric("MPI Transfer Inefficiency", 4, is_inefficiency=True),
        Metric("MPI Serialisation Inefficiency", 4, is_inefficiency=True),
        Metric("Thread Level Inefficiency", 2, is_inefficiency=True),
        Metric("OpenMP Region Inefficiency", 3, is_inefficiency=True),
        Metric("Serial Region Inefficiency", 3, is_inefficiency=True),
        Metric("Computational Scaling", 1),
        Metric("Instruction Scaling", 2),
        Metric("IPC Scaling", 2, "IPC Scaling"),
    ]

    _programming_model = "MPI + OpenMP"

    _default_metric_key = "Number of Processes"
    _default_group_key = "Threads per Process"

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

                metrics["OpenMP Region Inefficiency"] = (
                    (
                        stats["OpenMP Total Runtime"].loc[:, 1]
                        - stats["OpenMP Useful Computation"].groupby(level="rank").mean()
                    ).mean()
                ) / stats["Total Runtime"].max()

                metrics["Serial Region Inefficiency"] = (
                    stats["Serial Useful Computation"].loc[:, 1].mean()
                    / stats["Total Runtime"].max()
                    * (1 - 1 / nthreads)
                )

                metrics["Thread Level Inefficiency"] = (
                    stats["OpenMP Total Runtime"].loc[:, 1].mean()
                    - stats["OpenMP Useful Computation"].mean()
                    + stats["Serial Useful Computation"].loc[:, 1].mean()
                    * (1 - 1 / nthreads)
                ) / stats["Total Runtime"].max()

                metrics["MPI Communication Inefficiency"] = 1 - (
                    stats["Total Non-MPI Runtime"].loc[:, 1].max()
                    / stats["Total Runtime"].max()
                )

                try:
                    metrics["MPI Serialisation Inefficiency"] = (
                        stats["Ideal Runtime"].loc[:, 1].max()
                        - stats["Total Non-MPI Runtime"].loc[:, 1].max()
                    ) / stats["Total Runtime"].max()
                except KeyError:
                    metrics["MPI Serialisation Inefficiency"] = numpy.nan

                try:
                    metrics["MPI Transfer Inefficiency"] = 1 - (
                        stats["Ideal Runtime"].loc[:, 1].max()
                        / stats["Total Runtime"].max()
                    )
                except KeyError:
                    metrics["MPI Transfer Inefficiency"] = numpy.nan

                metrics["MPI Load Balance Inefficiency"] = (
                    stats["Total Non-MPI Runtime"].loc[:, 1].max()
                    - stats["Total Non-MPI Runtime"].loc[:, 1].mean()
                ) / stats["Total Runtime"].max()

                metrics["Process Level Inefficiency"] = (
                    1
                    - (stats["Total Non-MPI Runtime"].loc[:, 1].mean())
                    / stats["Total Runtime"].max()
                )

                metrics["Parallel Inefficiency"] = 1 - (
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

                metrics["Global Inefficiency"] = 1 - (
                    metrics["Computational Scaling"]
                    * (1 - metrics["Parallel Inefficiency"])
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
