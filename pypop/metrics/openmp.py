#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""OpenMP only metrics
"""

import pandas

from .metricset import MetricSet, Metric

__all__ = ['OpenMP_Metrics']


class OpenMP_Metrics(MetricSet):
    """Proposed Hybrid MPI+OpenMP Metrics.
    """

    _metric_list = [
        Metric("Global Efficiency", 0),
        Metric("Parallel Efficiency", 1),
        Metric("OpenMP Region Efficiency", 2, "OpenMP Region Efficiency"),
        Metric("Serial Region Efficiency", 2),
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
