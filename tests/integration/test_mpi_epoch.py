#!/usr/bin/env python3

import os

import pypop.examples
from pypop.extrae import is_extrae_tracefile

class TestMPI:
    def test_mpi_traces_exist(self):

        extrae_mpi_tracedir = os.path.join(
            pypop.examples.examples_directory(), "mpi", "epoch_example_traces"
        )

        self._extrae_mpi_traces = [
            f for f in os.listdir(extrae_mpi_tracedir) if is_extrae_tracefile(f)
        ]
