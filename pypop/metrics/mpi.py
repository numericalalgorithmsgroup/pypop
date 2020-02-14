#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Pure MPI metrics
"""

import numpy
import pandas

from .metricset import MetricSet, Metric


class MPI_Metrics(MetricSet):
    """Pure MPI Metrics.
    """

    _metric_list = [
        Metric("Global Efficiency", 0),
        Metric("Parallel Efficiency", 1),
        Metric("MPI Load balance", 2, "Load balance"),
        Metric("MPI Communication Efficiency", 2),
        Metric("MPI Transfer Efficiency", 3),
        Metric("MPI Serialisation Efficiency", 3),
        Metric("Computational Scaling", 1),
        Metric("Instruction Scaling", 2),
        Metric("IPC Scaling", 2),
        Metric("Frequency Scaling", 2),
    ]

    _default_metric_key = "Number of Processes"

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
            metrics = _create_layout_keys(metadata)

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

            metrics["MPI Load balance"] = 1 - (
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

            metrics_by_key[key] = metrics

        self._metric_data = pandas.DataFrame(metrics_by_key).T
