#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Hybrid metrics
"""

import numpy
import pandas

from .metricset import MetricSet, Metric

__all__ = [
    "MPI_OpenMP_Metrics",
    "MPI_OpenMP_Multiplicative_Metrics",
    "MPI_OpenMP_Ineff_Metrics",
]


class MPI_OpenMP_Ineff_Metrics(MetricSet):
    """Proposed Hybrid MPI+OpenMP Inefficiency Metrics.
    """

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
            stats = self._stats_dict[key].stats
            try:
                nthreads = metadata.application_layout.rank_threads[0][0]
                metrics = {"Number of Processes": sum(metadata.procs_per_node)}

                metrics["OpenMP Region Inefficiency"] = (
                    (
                        stats["OpenMP Total Runtime"].loc[:, 1]
                        - stats["OpenMP Useful Computation"].mean(level="rank")
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

                try:
                    if (
                        stats["Total Non-MPI Runtime"].loc[:, 1].max()
                        > stats["Ideal Runtime"].loc[:, 1].max()
                    ):
                        raise RuntimeError("Illegal Ideal Runtime value")
                except RuntimeError:
                    metrics["MPI Serialisation Inefficiency"] = numpy.nan
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
                    stats["IPC"].mean() / self._stats_dict[ref_key].stats["IPC"].mean()
                )

                metrics["Instruction Scaling"] = (
                    self._stats_dict[ref_key].stats["Useful Instructions"].sum()
                    / stats["Useful Instructions"].sum()
                )

                metrics["Frequency Scaling"] = (
                    stats["Frequency"].mean()
                    / self._stats_dict[ref_key].stats["Frequency"].mean()
                )

                metrics["Computational Scaling"] = (
                    self._stats_dict[ref_key].stats["Total Useful Computation"].sum()
                    / stats["Total Useful Computation"].sum()
                )

                metrics["Global Inefficiency"] = 1 - (
                    metrics["Computational Scaling"]
                    * (1 - metrics["Parallel Inefficiency"])
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

            metrics_by_key[key] = metrics

        self._metric_data = pandas.DataFrame(metrics_by_key).T


class MPI_OpenMP_Metrics(MetricSet):
    """Proposed Hybrid MPI+OpenMP Metrics.
    """

    _metric_list = [
        Metric("Global Efficiency", 0),
        Metric("Parallel Efficiency", 1),
        Metric("Process Level Efficiency", 2),
        Metric("MPI Load balance", 3, "Load balance"),
        Metric("MPI Communication Efficiency", 3),
        Metric("MPI Transfer Efficiency", 4),
        Metric("MPI Serialisation Efficiency", 4),
        Metric("Thread Level Efficiency", 2),
        Metric("OpenMP Region Efficiency", 3, "OpenMP Region Efficiency"),
        Metric("Serial Region Efficiency", 3),
        Metric("Computational Scaling", 1),
        Metric("Instruction Scaling", 2),
        Metric("IPC Scaling", 2, "IPC Scaling"),
    ]

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
            stats = self._stats_dict[key].stats
            try:
                nthreads = metadata.application_layout.rank_threads[0][0]
                metrics = {"Number of Processes": sum(metadata.procs_per_node)}

                metrics["OpenMP Region Efficiency"] = 1 - (
                    (
                        (
                            stats["OpenMP Total Runtime"].loc[:, 1]
                            - stats["OpenMP Useful Computation"].mean(level="rank")
                        ).mean()
                    )
                    / stats["Total Runtime"].max()
                )

                metrics["Serial Region Efficiency"] = 1 - (
                    stats["Serial Useful Computation"].loc[:, 1].mean()
                    / stats["Total Runtime"].max()
                    * (1 - 1 / nthreads)
                )

                metrics["Thread Level Efficiency"] = 1 - (
                    (
                        stats["OpenMP Total Runtime"].loc[:, 1].mean()
                        - stats["OpenMP Useful Computation"].mean()
                        + stats["Serial Useful Computation"].loc[:, 1].mean()
                        * (1 - 1 / nthreads)
                    )
                    / stats["Total Runtime"].max()
                )

                metrics["MPI Communication Efficiency"] = (
                    stats["Total Non-MPI Runtime"].loc[:, 1].max()
                    / stats["Total Runtime"].max()
                )

                try:
                    metrics["MPI Serialisation Efficiency"] = (
                        stats["Total Runtime"].loc[:, 1].max()
                        - stats["Total Non-MPI Runtime"].loc[:, 1].max()
                        + stats["Ideal Runtime"].loc[:, 1].max()
                    ) / stats["Total Runtime"].max()
                except KeyError:
                    metrics["MPI Serialisation Efficiency"] = numpy.nan

                try:
                    metrics["MPI Transfer Efficiency"] = (
                        stats["Ideal Runtime"].loc[:, 1].max()
                        / stats["Total Runtime"].max()
                    )
                except KeyError:
                    metrics["MPI Transfer Efficiency"] = numpy.nan

                try:
                    if (
                        stats["Total Non-MPI Runtime"].loc[:, 1].max()
                        > stats["Ideal Runtime"].loc[:, 1].max()
                    ):
                        raise RuntimeError("Illegal Ideal Runtime value")
                except RuntimeError:
                    metrics["MPI Serialisation Efficiency"] = numpy.nan
                    metrics["MPI Transfer Efficiency"] = numpy.nan

                metrics["MPI Load balance"] = 1 - (
                    (
                        stats["Total Non-MPI Runtime"].loc[:, 1].max()
                        - stats["Total Non-MPI Runtime"].loc[:, 1].mean()
                    )
                    / stats["Total Runtime"].max()
                )

                metrics["Process Level Efficiency"] = (
                    stats["Total Non-MPI Runtime"].loc[:, 1].mean()
                ) / stats["Total Runtime"].max()

                metrics["Parallel Efficiency"] = (
                    stats["Total Useful Computation"].mean()
                    / stats["Total Runtime"].max()  # avg all threads to include Amdahl
                )

                metrics["IPC Scaling"] = (
                    stats["IPC"].mean() / self._stats_dict[ref_key].stats["IPC"].mean()
                )

                metrics["Instruction Scaling"] = (
                    self._stats_dict[ref_key].stats["Useful Instructions"].sum()
                    / stats["Useful Instructions"].sum()
                )

                metrics["Frequency Scaling"] = (
                    stats["Frequency"].mean()
                    / self._stats_dict[ref_key].stats["Frequency"].mean()
                )

                metrics["Computational Scaling"] = (
                    self._stats_dict[ref_key].stats["Total Useful Computation"].sum()
                    / stats["Total Useful Computation"].sum()
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

            metrics_by_key[key] = metrics

        self._metric_data = pandas.DataFrame(metrics_by_key).T


class MPI_OpenMP_Multiplicative_Metrics(MetricSet):
    """Proposed Hybrid MPI+OpenMP Metrics (multiplicative version).
    """

    _metric_list = [
        Metric("Global Efficiency", 0),
        Metric("Parallel Efficiency", 1),
        Metric("Process Level Efficiency", 2),
        Metric("MPI Load balance", 3, "Load balance"),
        Metric("MPI Communication Efficiency", 3),
        Metric("MPI Transfer Efficiency", 4),
        Metric("MPI Serialisation Efficiency", 4),
        Metric("Thread Level Efficiency", 2),
        Metric("OpenMP Region Efficiency", 3, "OpenMP Region Efficiency"),
        Metric("Serial Region Efficiency", 3),
        Metric("Computational Scaling", 1),
        Metric("Instruction Scaling", 2),
        Metric("IPC Scaling", 2, "IPC Scaling"),
    ]

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
            stats = self._stats_dict[key].stats
            try:
                metrics = {"Number of Processes": sum(metadata.procs_per_node)}

                metrics["OpenMP Region Efficiency"] = (
                    stats["OpenMP Useful Computation"].mean()
                    + stats["Serial Useful Computation"].loc[:, 1].mean()
                ) / (
                    stats["OpenMP Total Runtime"].loc[:, 1].mean()
                    + stats["Serial Useful Computation"].loc[:, 1].mean()
                )

                metrics["Serial Region Efficiency"] = (
                    stats["Total Useful Computation"].mean()
                ) / (
                    stats["OpenMP Useful Computation"].mean()
                    + stats["Serial Useful Computation"].loc[:, 1].mean()
                )

                metrics["Thread Level Efficiency"] = (
                    stats["Total Useful Computation"].mean()
                ) / (
                    stats["OpenMP Total Runtime"].loc[:, 1].mean()
                    + stats["Serial Useful Computation"].loc[:, 1].mean()
                )

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
                    stats["OpenMP Total Runtime"].loc[:, 1].mean()
                    + stats["Serial Useful Computation"].loc[:, 1].mean()
                ) / stats["Total Non-MPI Runtime"].loc[:, 1].max()

                metrics["Process Level Efficiency"] = (
                    stats["Total Non-MPI Runtime"].loc[:, 1].mean()
                ) / stats["Total Runtime"].max()

                metrics["Parallel Efficiency"] = (
                    stats["Total Useful Computation"].mean()
                    / stats["Total Runtime"].max()  # avg all threads to include Amdahl
                )

                metrics["IPC Scaling"] = (
                    stats["IPC"].mean() / self._stats_dict[ref_key].stats["IPC"].mean()
                )

                metrics["Instruction Scaling"] = (
                    self._stats_dict[ref_key].stats["Useful Instructions"].sum()
                    / stats["Useful Instructions"].sum()
                )

                metrics["Frequency Scaling"] = (
                    stats["Frequency"].mean()
                    / self._stats_dict[ref_key].stats["Frequency"].mean()
                )

                metrics["Computational Scaling"] = (
                    self._stats_dict[ref_key].stats["Total Useful Computation"].sum()
                    / stats["Total Useful Computation"].sum()
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

            metrics_by_key[key] = metrics

        self._metric_data = pandas.DataFrame(metrics_by_key).T
