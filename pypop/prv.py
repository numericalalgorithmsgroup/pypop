#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

from collections import namedtuple, OrderedDict
from csv import writer as csvwriter
import gzip
from io import StringIO
import pickle

try:
    from tqdm.auto import tqdm
except ImportError:
    from .utils import return_first_arg as tqdm

import pandas as pd
import numpy as np

# Event states - hopefully these will not change(!)
K_STATE_RUNNING = "1"
K_EVENT_OMP_PARALLEL = "60000001"

Trace = namedtuple("Trace", field_names=["info", "data"])

TraceInfo = namedtuple(
    "TraceInfo",
    field_names=[
        "captured",
        "ns_elapsed",
        "nodes",
        "procs_per_node",
        "num_appl",
        "application_layout",
    ],
)

ApplicationLayout = namedtuple(
    "ApplicationLayout", field_names=["commsize", "rank_threads"]
)


def zipopen(path, modespec):
    try:
        if gzip.open(path, mode=modespec).readline():
            return gzip.open(path, mode=modespec)
    except OSError:
        return open(path, mode=modespec)


class PRV:

    record_types = OrderedDict([(1, "state"), (2, "event"), (3, "comm")])

    colnames = {
        "state": ["task", "thread", "cpu", "time", "endtime", "state"],
        "event": ["task", "thread", "cpu", "time", "event", "value"],
        "comm": None,
    }

    coltypes = {
        "state": {
            "task": np.int32,
            "thread": np.int32,
            "cpu": np.int32,
            "begin": np.int64,
            "end": np.int64,
            "state": np.int32,
        },
        "event": {
            "task": np.int32,
            "thread": np.int32,
            "cpu": np.int32,
            "begin": np.int64,
            "event": np.int32,
            "value": np.int64,
        },
        "comm": None,
    }

    def __init__(self, prv_path):

        if prv_path and prv_path.endswith(".prv"):
            self._parse_prv(prv_path)

        else:
            try:
                self._load_pickle(prv_path)
            except ValueError:
                raise ValueError("Not a prv or valid pickle")

    def _parse_prv(self, prv_path):

        temp_data = [StringIO() for _ in range(len(PRV.record_types))]
        temp_writers = [csvwriter(x) for x in temp_data]

        line_processors = {
            k: getattr(self, "_process_{}line".format(v))
            for k, v in PRV.record_types.items()
        }

        with zipopen(prv_path, "r") as prv_fh:
            headerline = next(prv_fh)
            self.traceinfo = _parse_paraver_headerline(headerline)

            # Skip the communicator lines for now
            try:
                skiplines = int(headerline[headerline.rfind("),") + 2 :])
            except ValueError:
                skiplines = 0
            for i in range(skiplines):
                next(prv_fh)

            for line in prv_fh:
                line = [int(x) for x in line.split(":")]
                line_processors[line[0]](line, temp_writers[line[0] - 1])

            for iattr, attrname in PRV.record_types.items():
                temp_data[iattr - 1].seek(0)
                try:
                    setattr(
                        self,
                        attrname,
                        pd.read_csv(
                            temp_data[iattr - 1],
                            names=PRV.colnames[attrname],
                            dtype=PRV.coltypes[attrname],
                        ),
                    )
                    getattr(self, attrname).set_index(
                        ["task", "thread", "time"], inplace=True
                    )
                    getattr(self, attrname).sort_index(inplace=True)
                except pd.errors.EmptyDataError:
                    setattr(self, attrname, None)

    def _process_stateline(self, line, writer):
        writer.writerow([line[3], line[4], line[1], line[5], line[6], line[7]])

    def _process_eventline(self, line, writer):
        row_start = [line[3], line[4], line[1], line[5]]
        nrows = (len(line) - 6) // 2
        rows = [row_start + line[6 + 2 * x : 8 + 2 * x] for x in range(nrows)]
        writer.writerows(rows)

    def _process_commline(self, line, writer):
        pass

    def save(self, filename):
        savedata = (self.traceinfo, self.state, self.event, self.comm)

        with gzip.open(filename, "wb", compresslevel=6) as fh:
            pickle.dump(savedata, fh)

    def _load_pickle(self, filename):
        with gzip.open(filename, "rb") as fh:
            data = pickle.load(fh)

        try:
            self.traceinfo, self.state, self.event, self.comm = data
        except ValueError:
            raise ValueError("Invalid pickle -- missing data")

    def profile_openmp_regions(self, no_progress=False):

        # First generate appropriate event subsets grouped by rank
        rank_state_groups = self.state.query(
            "state == {}".format(K_STATE_RUNNING)
        ).groupby(
            level="task"
        )  # State transitions
        rank_event_groups = self.event.query(
            "event == {}".format(K_EVENT_OMP_PARALLEL)
        ).groupby(
            level="task"
        )  # OMP regions

        # Now start collecting OMP regions
        rank_stats = {}
        for (irank, rank_events), (_, rank_states) in tqdm(
            zip(rank_event_groups, rank_state_groups),
            total=self.traceinfo.application_layout.commsize,
            disable=no_progress,
            leave=None
        ):

            # OpenMP events mark region start/end on master thread
            thread_events = rank_events.droplevel("task").loc[1, :]
            if not np.alltrue(np.diff(thread_events.index) >= 0):
                raise ValueError("Event timings are non-monotonic")

            region_starts = thread_events.index[thread_events["value"] != 0]
            region_ends = thread_events.index[thread_events["value"] == 0]
            region_lengths = region_ends - region_starts
            region_computation_mean = np.zeros_like(region_starts)
            region_computation_max = np.zeros_like(region_starts)

            # Iterate over threads to get max, average
            thread_state_groups = rank_states.droplevel(0).groupby(level="thread")
            for thread, thread_states in thread_state_groups:

                thread_states = thread_states.droplevel(0)
                if not (
                    np.alltrue(np.diff(thread_states.index) >= 0)
                    and np.alltrue(np.diff(thread_states["endtime"]) >= 0)
                ):
                    raise ValueError("State timings are non-monotonic")

                for idx in range(region_starts.shape[0]):
                    # Extract useful states that exist within OMP region
                    start_idx = thread_states["endtime"].searchsorted(
                        region_starts[idx] + 1
                    )
                    end_idx = thread_states.index.searchsorted(region_ends[idx])

                    # Tiny dataframe with useful regions
                    useful_state = thread_states.iloc[start_idx:end_idx]

                    # Sum to get useful length on thread
                    useful_length = np.asarray(
                        useful_state["endtime"] - useful_state.index
                    ).sum()

                    region_computation_mean[idx] += useful_length / len(
                        thread_state_groups
                    )
                    region_computation_max[idx] = max(
                        region_computation_max[idx], useful_length
                    )

                if np.any(region_computation_max > region_lengths):
                    raise ValueError("Oversized region")

            region_load_balance = region_computation_mean / region_computation_max

            rank_stats[irank] = pd.DataFrame(
                {
                    "Region Start": region_starts,
                    "Region End": region_ends,
                    "Load Balance": region_load_balance,
                    "Average Computation Time": region_computation_mean,
                    "Maximum Computation Time": region_computation_max,
                }
            )

        return pd.concat(rank_stats, names=["rank", "region"])


def _format_timedate(prv_td):
    return prv_td[prv_td.find("(") + 1 : prv_td.find(")")].replace(";", ":")


def _format_timing(prv_td):
    return int(prv_td[:-3])


def _split_nodestring(prv_td):
    nnodes, plist = prv_td.split("(", 2)
    nnodes = int(nnodes)
    ncpu_list = tuple(int(x) for x in plist.rstrip(",)").split(","))

    return (nnodes, ncpu_list)


def _format_num_apps(prv_td):
    return int(prv_td)


def _format_app_list(prv_td):
    try:
        prv_td = prv_td[: prv_td.index("),") + 1]
    except IndexError:
        pass
    apps = [x for x in prv_td.split(")") if x]

    if len(apps) > 1:
        raise ValueError("Only 1 traced application supported")
    app = apps[0].strip(",")
    nranks, nprocs = app.split("(")
    nranks = int(nranks)
    nprocs = tuple(tuple(int(y) for y in x.split(":")) for x in nprocs.split(","))

    appdata = ApplicationLayout(nranks, nprocs)

    return appdata


def _parse_paraver_headerline(headerline):
    if not headerline.startswith("#") or headerline.count(":") < 5:
        raise ValueError("Invalid headerline format")

    elems = headerline.replace(":", ";", 1).split(":", 4)

    traceinfo = TraceInfo(
        _format_timedate(elems[0]),
        _format_timing(elems[1]),
        *_split_nodestring(elems[2]),
        _format_num_apps(elems[3]),
        _format_app_list(elems[4])
    )

    return traceinfo


def get_prv_header_info(prv_file):
    """Get basic run information from the prv file header

    Parameters
    ----------
    prv_file: str
        Path to prv trace file.

    Returns:
    --------
    traceinfo: NamedTuple
        Named tuple containing the header information.
    """

    with zipopen(prv_file, "rt") as fh:
        return _parse_paraver_headerline(fh.readline().strip())
