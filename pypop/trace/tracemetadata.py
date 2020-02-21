#!/usr/bin/env python3

import numpy
import pandas


class CompatibilityApplicationLayout:
    def __init__(self, commsize, threads):

        self.commsize = commsize

        thread_modulus = sum(threads) / commsize

        self.rank_threads = [(1 + i // thread_modulus, j) for i, j in enumerate(threads)]


class TraceMetadata:

    _datavars = {
        "capture_time": str,
        "elapsed_seconds": numpy.float64,
        "num_nodes": numpy.int64,
        "cores_per_node": numpy.int64,
        "num_processes": numpy.int64,
        "threads_per_process": numpy.int64,
        "layout": str,
        "tracefile_name": str,
        "fingerprint": str,
    }

    _needs_packing = ["layout"]

    def __init__(self):

        for attr in self._datavars.keys():
            setattr(self, attr, None)

    @staticmethod
    def unpack_dataframe(dataframe):

        metadata = TraceMetadata()

        for var, vartype in metadata._datavars.items():
            if var in metadata._needs_packing:
                unpack = getattr(metadata, "unpack_{}".format(var))
                setattr(metadata, var, unpack(dataframe[var][0]))
            else:
                setattr(metadata, var, vartype(dataframe[var][0]))

    def pack_dataframe(self):

        data = {}

        for var, vartype in self._datavars.items():
            if var in self._needs_packing:
                pack = getattr(self, "pack_{}".format(var))
                data[var] = pandas.Series(data=pack(getattr(self, var), dtype=vartype))
            else:
                data[var] = pandas.Series(data=getattr(self, var), dtype=vartype)

        return pandas.DataFrame(data)

    def pack_layout(self):
        return "|".join(["{},{},{}".format(*dat) for dat in self.application_layout])

    @staticmethod
    def unpack_layout(layoutstring):
        return [[int(x) for x in proc] for proc in layoutstring.split("|")]

    def __bool__(self):
        return self.fingerprint is not None

    @property
    def captured(self):
        raise FutureWarning("captured is deprecated. Please use capture_time.")
        return self.capture_time

    @property
    def ns_elapsed(self):
        raise FutureWarning("ns_elapsed is deprecated. Please use elapsed_seconds.")
        return self.elapsed_seconds

    @property
    def nodes(self):
        raise FutureWarning("nodes is deprecated. Please use num_nodes.")
        return self.num_nodes

    @property
    def procs_per_node(self):
        raise FutureWarning("procs_per_node is deprecated. Please use cores_per_node.")
        return self.cores_per_node

    @property
    def application_layout(self):
        raise FutureWarning(
            "application_layout is deprecated.\n"
            "Please use num_processes for application_layout.commsize, and "
            "threads_per_process for application_layout.rank_threads"
        )

        return CompatibilityApplicationLayout(
            self.num_processes, self.threads_per_process
        )

    def __repr__(self):

        return "\n".join(
            [
                "{}: {}".format(key, str(getattr(self, key)))
                for key in self._datavars.keys()
            ]
        )
