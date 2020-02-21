#!/usr/bin/env python3

from hashlib import sha1

from .trace import Trace

from ..utils.io import zipopen


class PRVTrace(Trace):
    def _gather_metadata(self):

        with zipopen(self._tracefile, "r") as fh:
            headerline = fh.readline().decode().strip()

        elem = headerline.replace(":", ";", 1).split(":", 4)

        self._metadata.capture_time = elem[0][
            elem[0].find("(") + 1 : elem[0].find(")")
        ].replace(";", ":")

        self._metadata.elapsed_seconds = float(elem[1][:-3]) * 1e-9

        self._metadata.num_nodes, self._metadata.cores_per_node = self._split_nodestring(
            elem[2]
        )

        # elem 3 is the number of applications, currently only support 1
        if int(elem[3]) != 1:
            raise ValueError("Multi-application traces are not supported")

        # elem 4 contains the application layout processes/threads
        (
            self._metadata.num_processes,
            self._metadata.threads_per_process,
        ) = self._split_layoutstring(elem[4])

        self._metadata.tracefile_name = self._tracefile

        hasher = sha1()
        hasher.update(headerline)
        self._metadata.fingerprint = hasher.hexdigest()

    def _gather_statistics(self):
        self._statistics = "TODO"

    @staticmethod
    def _split_nodestring(prv_td):
        num_nodes, plist = prv_td.split("(", 2)
        num_nodes = int(num_nodes)
        cores_per_node = tuple(int(x) for x in plist.rstrip(",)").split(","))

        return (num_nodes, cores_per_node)

    @staticmethod
    def _split_layoutstring(prv_td):

        prv_td = prv_td.split(")")[0].strip()
        commsize, layoutstring = prv_td.split("(")

        commsize = int(commsize)
        threads = [int(x.split(":")[0]) for x in layoutstring.split()]

        return (commsize, threads)


PRVTrace.register_loader()
