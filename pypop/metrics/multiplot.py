#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""Shared plot routines for multiple Metric Sets
"""

import numpy
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from .._plotsettings import pypop_mpl_params, figparams

from .metricset import MetricSet
from ..traceset import TraceSet


def _get_dataframe(obj):
    if isinstance(obj, (TraceSet)):
        return obj.statistics

    if isinstance(obj, MetricSet):
        return obj.metric_data

    return obj


def plot_scalings(
    scalings,
    x_key="Number of Processes",
    y_key="Speedup",
    x_label=None,
    y_label=None,
    series_labels=None,
    title=None,
):
    """Plot scaling graph with region shading.

        Plots scaling data from pandas dataframe(s). The 0-80% and 80-100% scaling
        regions are shaded for visual identification.  Multiple scaling lines may be
        plotted by passing a dict of dataframes.

        Parameters
        ----------
        scalings_dict: (list of) MetricSet, TraceSet, or pandas.DataFrame
            (List of) Pandas DataFrames containing scaling data.

        x_key: scalar
            Key of Dataframe column to use as x-axis.

        y_key: scalar
            key of Dataframe column to use as y-axis.

        x_label: str or None
            Label to be used for x-axis and data series. Defaults to `x_key`.

        y_label: str or None
            Label to be used for y-axis and data series. Defaults to `y_key`.

        series_labels: list of str or None
            Lables to be used for series, defaults to `y_label` for a single series,
            enumeration for multiple series.

        title: str or None
            Optional title for plot.

        Returns
        -------
        figure: matplotlib.figure.Figure
            Figure containing complete scaling plot.
        """
    with mpl.rc_context(pypop_mpl_params):
        return _plot_scalings_multiple(
            scalings, x_key, y_key, x_label, y_label, series_labels, title
        )


def _plot_scalings_multiple(
    scalings, x_key, y_key, x_label, y_label, series_labels, title,
):

    # Wrap single instance in list if needed
    if isinstance(scalings, (pandas.DataFrame, TraceSet, MetricSet)):
        scalings = [scalings]

    # Make sure we are dealing with all pd.Dataframe
    scalings = [_get_dataframe(dataset) for dataset in scalings]

    # Sort out labels as needed
    if x_label is None:
        x_label = x_key

    if y_label is None:
        y_label = y_key

    if series_labels is None:
        if len(scalings) == 1:
            series_labels = [y_label]
        else:
            series_labels = range(len(scalings))

    # Calculate x and y ranges
    cores_min = numpy.nan
    cores_max = numpy.nan
    y_max = numpy.nan

    for dataset in scalings:
        cores_min = numpy.nanmin([cores_min, dataset[x_key].min()])
        cores_max = numpy.nanmax([cores_max, dataset[x_key].max()])
        y_max = numpy.nanmax((y_max, dataset[y_key].max()))

    # And pad appropriately
    y_max *= 1.2
    x_margin = 0.02 * cores_max

    # Calculate ideal scalings
    ideal_scaling_cores = numpy.linspace(cores_min - x_margin, cores_max + x_margin)
    ideal_scaling = ideal_scaling_cores / cores_min
    ideal_scaling_80pc = 0.2 + 0.8 * ideal_scaling

    # Finally, plot
    fig = plt.figure(figsize=figparams["single.figsize"])
    ax = fig.add_axes(figparams["single.axlayout"][0])

    ax.fill_between(
        ideal_scaling_cores,
        0,
        ideal_scaling_80pc,
        label="80% Scaling",
        alpha=0.1,
        color="g",
        linestyle="-",
    )
    ax.fill_between(
        ideal_scaling_cores,
        ideal_scaling_80pc,
        ideal_scaling,
        label="Ideal Scaling",
        alpha=0.2,
        color="g",
        linestyle="-",
    )

    for dataset, label in zip(scalings, series_labels):
        ax.plot(
            dataset[x_key],
            dataset[y_key],
            label=label,
            marker="x",
            linestyle="-",
            alpha=0.8,
        )

    ax.set_xlim(ideal_scaling_cores.min(), ideal_scaling_cores.max())
    ax.set_ylim(0, y_max)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_locator(mtick.FixedLocator(scalings[0][x_key], 6))
    ax.legend()

    if title:
        ax.set_title(title)

    return fig
