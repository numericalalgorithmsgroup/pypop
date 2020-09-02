#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Judit Hybrid metrics
"""

import pandas

from .metricset import MetricSet, Metric

__all__ = ["Judit_Hybrid_Metrics"]


class Judit_Hybrid_Metrics(MetricSet):
    """Judit Multiplicative MPI+OpenMP Metrics.
    """

    _metric_list = [
        Metric("Global Efficiency", 0),
        Metric("Hybrid Parallel Efficiency", 1),
        Metric("MPI Parallel Efficiency", 2),
        Metric("MPI Load Balance", 3),
        Metric("MPI Communication Efficiency", 3),
        Metric("OpenMP Parallel Efficiency", 2),
        Metric("OpenMP Communication Efficiency", 3),
        Metric("OpenMP Load Balance", 3),
        Metric("Hybrid Communication Efficiency", 2),
        Metric("Hybrid Load Balance", 2),
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
            metrics = self._create_subdataframe(metadata, key)

            try:
                metrics["MPI Communication Efficiency"] = (
                    stats["Total Non-MPI Runtime"].loc[:, 1].max()
                    / stats["Total Runtime"].max()
                )

                metrics["MPI Load Balance"] = (
                    stats["Total Non-MPI Runtime"].loc[:, 1].mean()
                    / stats["Total Non-MPI Runtime"].loc[:, 1].max()
                )

                metrics["MPI Parallel Efficiency"] = (
                    stats["Total Non-MPI Runtime"].loc[:, 1].mean()
                ) / stats["Total Runtime"].max()

                metrics["Hybrid Parallel Efficiency"] = (
                    stats["Total Useful Computation"].mean()
                    / stats["Total Runtime"].max()  # avg all threads to include Amdahl
                )

                metrics["Hybrid Load Balance"] = (
                    stats["Total Useful Computation"].mean()
                    / stats["Total Useful Computation"].max()
                )

                metrics["Hybrid Communication Efficiency"] = (
                    stats["Total Useful Computation"].max()
                    / stats["Total Runtime"].max()
                )

                metrics["OpenMP Parallel Efficiency"] = (
                    metrics["Hybrid Parallel Efficiency"]
                    / metrics["MPI Parallel Efficiency"]
                )

                metrics["OpenMP Load Balance"] = (
                    metrics["Hybrid Load Balance"] / metrics["MPI Load Balance"]
                )

                metrics["OpenMP Communication Efficiency"] = (
                    metrics["Hybrid Communication Efficiency"]
                    / metrics["MPI Communication Efficiency"]
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

                metrics["Global Efficiency"] = (
                    metrics["Computational Scaling"]
                    * metrics["Hybrid Parallel Efficiency"]
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
