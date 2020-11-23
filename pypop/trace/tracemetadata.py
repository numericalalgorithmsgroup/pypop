#!/usr/bin/env python3

from collections import namedtuple

from warnings import warn

import numpy
import pandas

ProcessLayout = namedtuple("ProcessLayout", ["node", "process", "num_threads"])


class CompatibilityApplicationLayout:
    def __init__(self, commsize, threads):

        self.commsize = commsize

        thread_modulus = sum(threads) / commsize

        self.rank_threads = [(j, 1 + i // thread_modulus) for i, j in enumerate(threads)]


class TraceMetadata:

    _tracesubclasskey = "PyPOPTraceSubclass"

    _datavars = {
        _tracesubclasskey: str,
        "capture_time": str,
        "elapsed_seconds": numpy.float64,
        "num_nodes": int,
        "cores_per_node": int,
        "num_processes": int,
        "threads_per_process": int,
        # "layout": str,
        "tracefile_name": str,
        "fingerprint": str,
        "tag": str,
    }

    _optional = {"tag"}

    _needs_packing = ["layout", "threads_per_process"]

    def __init__(self, parent):

        for attr in self._datavars.keys():
            setattr(self, attr, None)

        if parent is not None:
            setattr(self, TraceMetadata._tracesubclasskey, parent.__class__.__name__)
        else:
            setattr(self, TraceMetadata._tracesubclasskey, None)

    @staticmethod
    def unpack_dataframe(dataframe):

        metadata = TraceMetadata(None)

        for var, vartype in metadata._datavars.items():
            try:
                packed_data = dataframe[var]
            except KeyError as err:
                if var not in metadata._optional:
                    warn("Missing metadata entry: {}".format(err))
                setattr(metadata, var, None)
                continue
            if var in metadata._needs_packing:
                unpack = getattr(metadata, "unpack_{}".format(var))
                setattr(metadata, var, unpack(packed_data, vartype))
            else:
                setattr(metadata, var, vartype(packed_data[0]))

        return metadata

    def pack_dataframe(self):

        data = {}

        for var, vartype in self._datavars.items():
            if getattr(self, var) is None:
                continue
            if var in self._needs_packing:
                pack = getattr(self, "pack_{}".format(var))
                packed = pack(getattr(self, var), vartype)
                assert isinstance(packed, (pandas.Series, pandas.DataFrame))
                data[var] = packed
            else:
                data[var] = pandas.Series(data=getattr(self, var), dtype=vartype)

        return pandas.DataFrame(data)

    @staticmethod
    def pack_layout(layout, dtype):
        return pandas.Series(
            data="|".join(["{},{},{}".format(*dat) for dat in layout]), dtype=dtype
        )

    @staticmethod
    def unpack_layout(packed, vartype):
        layoutstring = packed[0]
        return [
            ProcessLayout(*(int(x) for x in proc)) for proc in layoutstring.split("|")
        ]

    @staticmethod
    def pack_threads_per_process(data, dtype):
        data = "|".join(str(x) for x in data)
        return pandas.Series(data=data, dtype=str)

    @staticmethod
    def unpack_threads_per_process(data, dtype):
        try:
            return [dtype(x) for x in data[0].split("|")]
        except AttributeError:
            return list(data)

    def __bool__(self):
        return self.fingerprint is not None

    @property
    def captured(self):
        warn("captured is deprecated. Please use capture_time.", FutureWarning)
        return self.capture_time

    @property
    def ns_elapsed(self):
        warn("ns_elapsed is deprecated. Please use elapsed_seconds.", FutureWarning)
        return self.elapsed_seconds

    @property
    def nodes(self):
        warn("nodes is deprecated. Please use num_nodes.", FutureWarning)
        return self.num_nodes

    @property
    def procs_per_node(self):
        warn("procs_per_node is deprecated. Please use cores_per_node.", FutureWarning)
        return self.cores_per_node

    @property
    def application_layout(self):
        warn(
            "application_layout is deprecated.\n"
            "Please use num_processes for application_layout.commsize, and "
            "threads_per_process for application_layout.rank_threads",
            FutureWarning,
        )

        return CompatibilityApplicationLayout(
            self.num_processes, self.threads_per_process
        )

    @property
    def trace_subclass_name(self):
        return getattr(self, TraceMetadata._tracesubclasskey)

    def __repr__(self):

        return "\n".join(
            [
                "{}: {}".format(key, str(getattr(self, key)))
                for key in self._datavars.keys()
            ]
        )
