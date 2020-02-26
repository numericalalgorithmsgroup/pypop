#!/usr/bin/env python3

import pytest

import os
from collections import namedtuple

import pandas
import numpy

import pypop.examples
from pypop.extrae import is_extrae_tracefile
from pypop.traceset import TraceSet
from pypop.metrics import MPI_Metrics
from pypop.trace import PRVTrace

MetricData = namedtuple("MetricData", ["testfiles", "metric_data"])


@pytest.fixture(scope="class")
def testdata():
    extrae_mpi_tracedir = os.path.join(
        pypop.examples.examples_directory(), "mpi", "epoch_example_traces"
    )

    extrae_mpi_traces = [
        os.path.join(extrae_mpi_tracedir, f)
        for f in os.listdir(extrae_mpi_tracedir)
        if is_extrae_tracefile(f)
    ]

    metrics = pandas.read_csv(os.path.join(extrae_mpi_tracedir, "precomputed.csv"))

    return MetricData(testfiles=extrae_mpi_traces, metric_data=metrics)


class TestMPI:
    def test_mpi_trace_analysis(self, testdata):

        statistics = TraceSet(testdata.testfiles, force_recalculation=True)

        metrics = MPI_Metrics(statistics.by_commsize())

        for metric in metrics.metrics:
            print(metric.key)
            assert numpy.all(
                numpy.isclose(
                    metrics.metric_data[metric.key].values.astype(numpy.float64),
                    testdata.metric_data[metric.key].values.astype(numpy.float64),
                    rtol=0,
                    atol=0.01,
                )
            )
