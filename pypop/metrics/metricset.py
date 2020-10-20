#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Shared routines for different Metric Sets
"""

from warnings import warn

import numpy
import pandas

from ..trace import Trace
from ..traceset import TraceSet
from .._plotsettings import pypop_mpl_params, figparams

__all__ = ["Metric", "MetricSet"]


class Metric:
    """Individual performance metrics to be used within a metricset. Defines metric name,
    properties and method of calculation.
    """

    def __init__(
        self,
        key,
        level,
        displayname=None,
        desc=None,
        is_inefficiency=False,
        freq_corr=False,
    ):
        """
        Parameters
        ----------
        key: str
            Key by which to identify metric.

        level: int
            Level at which to display metric in the stack.

        displayname: str or None
            Display name to use for metric in table etc. Defaults to key.

        desc: str or None
            Detailed description of the metric.

        is_inefficiency: bool
            Tag metric as an inefficiency (rather than efficiency) for correct display
            and shading. Default False.

        freq_corr: bool
            Correct performance metrics based on average clock frequency (use to
            correct for node dynamic clocking issues). Default False.
        """
        self.key = key
        self.level = level
        self.description = str(desc) if desc else ""
        self.is_inefficiency = is_inefficiency

        if displayname:
            self.displayname = r"↪ " * bool(self.level) + displayname
        else:
            self.displayname = r"↪ " * bool(self.level) + self.key


class MetricSet:
    """Calculate and plot POP MPI metrics

    Statistics data is expected to have been produced with `collect_statistics()`

    Attributes
    ----------
    metric_data
    metric_definition

    """

    _programming_model = None

    _default_metric_key = "Number of Processes"
    _default_group_key = None
    _default_scaling_key = "Total Threads"

    _key_descriptions = {
        "Number of Processes": "",
        "Threads per Process": "",
        "Total Threads": "",
        "Hybrid Layout": "",
        "Tag": "",
    }

    def __init__(self, stats_data, ref_key=None, sort_keys=True):
        """
        Parameters
        ----------
        stats_data: TraceSet instance, dict, iterable or instance of Trace
            Statistics as collected with `collect_statistics()`. Dictionary keys will be
            used as the dataframe index. If a list, a dict will be constructed by
            enumeration.

        ref_key: str or None
            Key of stats_dict that should be used as the reference for calculation of
            scaling values.  By default the trace with smallest number of processes and
            smallest number of threads per process will be used.

        sort_keys: bool
            If true (default), lexically sort the keys in the returned DataFrame.
        """

        self._stats_dict = MetricSet._dictify_stats(stats_data)
        self._metric_data = None
        self._sort_keys = sort_keys
        self._ref_key = (
            self._choose_ref_key(self._stats_dict) if ref_key is None else ref_key
        )

    def _calculate_metrics(self):
        raise NotImplementedError

    def _repr_html_(self):
        return self.metric_data._repr_html_()

    @staticmethod
    def _choose_ref_key(stats_dict):
        """ Take the stats dict and choose an appropriate reference trace.

        As a default choice choose the smallest number of total threads, breaking ties
        with smallest number of threads per process
        """

        return min(
            stats_dict.items(),
            key=lambda x: "{:05}_{:05}_{}".format(
                sum(x[1].metadata.threads_per_process),
                max(x[1].metadata.threads_per_process),
                x[1].metadata.tag,
            ),
        )[0]

    @property
    def metric_data(self):
        """pandas.DataFrame: Calculated metric data.
        """
        if self._metric_data is None:
            self._calculate_metrics(ref_key=self._ref_key)
        return self._metric_data

    @staticmethod
    def _dictify_stats(stats_data):
        if isinstance(stats_data, TraceSet):
            return {k: v for k, v in enumerate(stats_data.traces)}
        else:
            if isinstance(stats_data, Trace):
                return {0: stats_data}
            if not isinstance(stats_data, dict):
                stats_data = {k: v for k, v in enumerate(stats_data)}

        for df in stats_data.values():
            if not isinstance(df, Trace):
                raise ValueError("stats_dict must be an iterable of pypop.trace.Trace")

            return stats_data

    @property
    def metrics(self):
        """List of :py:class:`pypop.metrics.Metric`: List of metrics that will be
        calculated.
        """
        return self._metric_list

    def _create_subdataframe(self, metadata, idxkey):
        if len(set(metadata.threads_per_process)) != 1:
            warn(
                "The supplied trace has a varying number of threads per process. "
                "The PyPOP metrics were designed assuming a homogenous number of "
                "threads per process -- analysis results may be inaccurate."
            )

        layout_keys = {
            "Number of Processes": pandas.Series(
                data=[metadata.num_processes], index=[idxkey]
            ),
            "Threads per Process": pandas.Series(
                data=[metadata.threads_per_process[0]], index=[idxkey]
            ),
            "Total Threads": pandas.Series(
                data=[sum(metadata.threads_per_process)], index=[idxkey]
            ),
            "Hybrid Layout": pandas.Series(
                data=[
                    "{}x{}".format(
                        metadata.num_processes, metadata.threads_per_process[0]
                    )
                ],
                index=[idxkey],
            ),
            "Tag": pandas.Series(data=[metadata.tag], index=[idxkey]),
        }

        for metric in self._metric_list:
            layout_keys[metric.key] = pandas.Series(data=[0.0], index=[idxkey])

        return pandas.DataFrame(layout_keys)

    def plot_table(
        self,
        columns_key="auto",
        group_key="auto",
        title=None,
        columns_label=None,
        fontsize=14,
        **kwargs
    ):
        from pypop.notebook_interface.plotting import MetricTable

        return MetricTable(
            self,
            columns_key=columns_key,
            group_key=group_key,
            title=title,
            columns_label=columns_label,
            fontsize=fontsize,
        )

    def plot_scaling(
        self,
        scaling_variable="Speedup",
        independent_variable="auto",
        group_key="auto",
        title=None,
        fontsize=14,
        fit_data_only=False,
        **kwargs
    ):
        from pypop.notebook_interface.plotting import ScalingPlot

        return ScalingPlot(
            self,
            scaling_variable=scaling_variable,
            independent_variable=independent_variable,
            group_key=group_key,
            title=title,
            fontsize=fontsize,
            fit_data_only=fit_data_only,
        )
