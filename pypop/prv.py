#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""\
PRV Trace Loader
----------------

Support for directly loading PRV trace data as Pandas dataframes
"""

from warnings import warn
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
K_EVENT_OMP_TASK_FUNCTION = "60000018"
K_EVENT_OMP_LOOP_FUNCTION = "60000023"
K_EVENT_OMP_TASK_FILE_AND_LINE = "60000118"
K_EVENT_OMP_LOOP_FILE_AND_LINE = "60000123"

Trace = namedtuple("Trace", field_names=["info", "data"])

TraceMetadata = namedtuple(
    "TraceMetadata",
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

        if prv_path and prv_path.endswith((".prv", ".prv.gz")):
            self._event_names = {}
            self._event_vals = {}
            self._parse_pcf(prv_path)

            self._parse_prv(prv_path)

        else:
            try:
                self._load_pickle(prv_path)
            except ValueError:
                raise ValueError("Not a prv or valid pickle")

        self._omp_region_data = None

    def _parse_pcf(self, prv_path):

        if prv_path.endswith(".gz"):
            prv_path = prv_path[:-3]

        pcf_path = "".join([prv_path[:-3], "pcf"])

        try:
            with open(pcf_path, "rt") as fh:
                block_mode = None
                for line in fh:
                    if not line.strip():
                        continue
                    if not line[0].isdigit():
                        block_mode = line.split()[0]
                        continue

                    if block_mode == "EVENT_TYPE":
                        linevals = line.strip().split(maxsplit=2)
                        eventkey = linevals[1]
                        self._event_names[eventkey] = linevals[2]
                        self._event_vals[eventkey] = {}
                        continue

                    if block_mode == "VALUES":
                        linevals = line.strip().split(maxsplit=1)
                        valuekey = linevals[0]
                        self._event_vals[eventkey][valuekey] = linevals[1]

        except FileNotFoundError:
            pass

    def _parse_prv(self, prv_path):

        temp_data = [StringIO() for _ in range(len(PRV.record_types))]
        temp_writers = [csvwriter(x) for x in temp_data]

        line_processors = {
            k: getattr(self, "_process_{}line".format(v))
            for k, v in PRV.record_types.items()
        }

        with zipopen(prv_path, "rt") as prv_fh:
            headerline = next(prv_fh)
            self.metadata = _parse_paraver_headerline(headerline)

            # Skip the communicator lines for now
            try:
                skiplines = int(headerline[headerline.rfind("),") + 2 :])
            except ValueError:
                skiplines = 0
            for i in range(skiplines):
                next(prv_fh)

            for line in prv_fh:
                # Skip comment lines
                if line.startswith("#"):
                    continue
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
        savedata = (self.metadata, self.state, self.event, self.comm)

        with gzip.open(filename, "wb", compresslevel=6) as fh:
            pickle.dump(savedata, fh)

    def _load_pickle(self, filename):
        with gzip.open(filename, "rb") as fh:
            data = pickle.load(fh)

        try:
            self.metadata, self.state, self.event, self.comm = data
        except ValueError:
            raise ValueError("Invalid pickle -- missing data")

    def profile_openmp_regions(self, no_progress=False):
        """Profile OpenMP Region Info
        """

        if self._omp_region_data is not None:
            return self._omp_region_data

        idx_master_threads = pd.IndexSlice[:, 1]

        # First generate appropriate event subsets grouped by rank
        rank_state_groups = self.state.query(
            "state == {}".format(K_STATE_RUNNING)
        ).groupby(
            level="task"
        )  # State transitions
        rank_event_groups = (
            self.event.loc[idx_master_threads, :]
            .query("event == {}".format(K_EVENT_OMP_PARALLEL))
            .droplevel("thread")
            .groupby(level="task")
        )  # OMP regions
        rank_func_groups = (
            self.event.query(
                "event == {} or event == {}".format(
                    K_EVENT_OMP_TASK_FUNCTION, K_EVENT_OMP_LOOP_FUNCTION
                )
            )
            .query("value != 0")
            .groupby(level="task")
        )  # OMP Functions

        rank_func_loc_groups = (
            self.event.query(
                "event == {} or event == {}".format(
                    K_EVENT_OMP_TASK_FILE_AND_LINE, K_EVENT_OMP_LOOP_FILE_AND_LINE
                )
            )
            .query("value != 0")
            .groupby(level="task")
        )  # OMP Functions

        # Now start collecting OMP regions
        rank_stats = {}
        for (
            (irank, rank_events),
            (_, rank_states),
            (_, rank_funcs),
            (_, rank_func_locs),
        ) in tqdm(
            zip(
                rank_event_groups,
                rank_state_groups,
                rank_func_groups,
                rank_func_loc_groups,
            ),
            total=self.metadata.application_layout.commsize,
            disable=no_progress,
            leave=None,
        ):

            # OpenMP events mark region start/end on master thread
            thread_events = rank_events.droplevel("task")
            if not np.alltrue(np.diff(thread_events.index) >= 0):
                raise ValueError("Event timings are non-monotonic")

            region_starts = thread_events.index[thread_events["value"] != 0]
            region_ends = thread_events.index[thread_events["value"] == 0]

            # Now sanity check regions and try to repair issues caused by missing events:

            # First region start should be earlier than first region end
            if region_ends[0] <= region_starts[0]:
                warn(
                    "Incomplete OpenMP region found. This likely means the trace was "
                    "cut through a region"
                )
                while len(region_ends) > 0 and region_ends[0] <= region_starts[0]:
                    region_ends = region_ends[1:]

            # Last region end should be after last region start
            if region_starts[-1] >= region_ends[-1]:
                warn(
                    "Incomplete OpenMP region found. This likely means the trace was "
                    "cut through a region"
                )
                while len(region_starts) > 0 and region_starts[-1] >= region_ends[-1]:
                    region_starts = region_starts[:-1]

            if np.any(region_starts > region_ends):
                raise ValueError("Unable to make sense of OpenMP region events.")

            region_lengths = region_ends - region_starts
            region_computation_mean = np.zeros_like(region_starts)
            region_computation_max = np.zeros_like(region_starts)
            region_computation_sum = np.zeros_like(region_starts)

            regionintervals = pd.IntervalIndex.from_arrays(region_starts, region_ends)

            funcbins = pd.cut(
                rank_funcs.droplevel(("task", "thread")).index, regionintervals
            ).codes
            region_funcs = rank_funcs.droplevel(("task", "thread")).groupby(funcbins)
            region_fingerprints_func = region_funcs.apply(
                # {"value": lambda x: ":".join("{}".format(int(y) for y in x.unique()))}
                lambda x: ":".join(["{:d}".format(int(y)) for y in x["value"].unique()])
            )

            funclocbins = pd.cut(
                rank_func_locs.droplevel(("task", "thread")).index, regionintervals
            ).codes
            region_func_locs = rank_func_locs.droplevel(("task", "thread")).groupby(
                funclocbins
            )
            region_fingerprints_loc = region_func_locs.apply(
                # {"value": lambda x: ":".join("{}".format(int(y) for y in x.unique()))}
                lambda x: ":".join(["{:d}".format(int(y)) for y in x["value"].unique()])
            )

            # Iterate over threads to get max, average
            thread_state_groups = rank_states.droplevel(0).groupby(level="thread")
            for thread, thread_states in tqdm(
                thread_state_groups,
                total=len(thread_state_groups),
                disable=no_progress,
                leave=None,
            ):

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

                    region_computation_sum[idx] += useful_length

                if np.any(region_computation_max > region_lengths):
                    raise ValueError("Oversized region")

            region_load_balance = 1 - (
                (region_computation_max - region_computation_mean) / region_lengths
            )

            region_parallel_efficiency = 1 - (
                (region_lengths - region_computation_mean) / region_lengths
            )

            rank_stats[irank] = pd.DataFrame(
                {
                    "Region Start": region_starts,
                    "Region End": region_ends,
                    "Region Length": region_lengths,
                    "Load Balance": region_load_balance,
                    "Parallel Efficiency": region_parallel_efficiency,
                    "Average Computation Time": region_computation_mean,
                    "Maximum Computation Time": region_computation_max,
                    "Computation Delay Time": region_computation_max
                    - region_computation_mean,
                    "Region Total Computation": region_computation_sum,
                    "Region Delay Time": region_lengths - region_computation_mean,
                    "Region Function Fingerprint": region_fingerprints_func,
                    "Region Location Fingerprint": region_fingerprints_loc,
                }
            )

        self._omp_region_data = pd.concat(rank_stats, names=["rank", "region"])

        return self._omp_region_data

    def region_location_from_fingerprint(self, fingerprint):
        if self._event_vals:
            fpvals = fingerprint.split(":")
            fpstrings = [self.omp_location_by_value(fpk, "MISSINGVAL") for fpk in fpvals]
            return ":".join(fpstrings)

        return fingerprint

    def omp_location_by_value(self, value, default="MISSINGVAL"):
        value_dict = {
            **self._event_vals.get(K_EVENT_OMP_TASK_FILE_AND_LINE, {}),
            **self._event_vals.get(K_EVENT_OMP_LOOP_FILE_AND_LINE, {}),
        }

        return value_dict.get(value, default)

    def region_function_from_fingerprint(self, fingerprint):
        if self._event_vals:
            fpvals = fingerprint.split(":")
            fpstrings = [self.omp_function_by_value(fpk, "MISSINGVAL") for fpk in fpvals]
            return ":".join(fpstrings)

        return fingerprint

    def omp_function_by_value(self, value, default="MISSINGVAL"):
        value_dict = {
            **self._event_vals.get(K_EVENT_OMP_TASK_FUNCTION, {}),
            **self._event_vals.get(K_EVENT_OMP_LOOP_FUNCTION, {}),
        }

        return value_dict.get(value, default)

    def openmp_region_summary(self, by_location=False):
        """Summarize OpenMP regions, grouped by either the function name or location
        (filename and line number) within the source.

        Parameters
        ----------
        by_location: bool
            If true, aggregate functions based on their location within the source,
            otherwise aggregate by function name alone.
        """

        if by_location:
            fingerprint_key = "Region Location Fingerprint"
            fingerprint_to_text_function = self.region_location_from_fingerprint
        else:
            fingerprint_key = "Region Function Fingerprint"
            fingerprint_to_text_function = self.region_function_from_fingerprint

        runtime = self.metadata.ns_elapsed

        self.profile_openmp_regions()

        summary = self._omp_region_data.groupby(fingerprint_key).agg(
            **{
                "Instances": ("Maximum Computation Time", "count"),
                "Relative Load Balance Efficiency": (
                    "Load Balance",
                    lambda x: np.average(
                        x, weights=self._omp_region_data.loc[x.index, "Region Length"],
                    ),
                ),
                "Relative Parallel Efficiency": (
                    "Parallel Efficiency",
                    lambda x: np.average(
                        x, weights=self._omp_region_data.loc[x.index, "Region Length"]
                    ),
                ),
                "Load Balance Efficiency": (
                    "Computation Delay Time",
                    lambda x: 1 - np.sum(x / runtime),
                ),
                "Parallel Efficiency": (
                    "Region Delay Time",
                    lambda x: 1 - np.sum(x / runtime),
                ),
                "Accumulated Region Time": ("Region Length", np.sum),
                "Accumulated Computation Time": ("Region Total Computation", np.sum),
                "Average Computation Time": ("Average Computation Time", np.average),
                "Maximum Computation Time": ("Maximum Computation Time", np.max),
                "Region Functions": (
                    fingerprint_key,
                    lambda x: fingerprint_to_text_function(x[0]),
                ),
            }
        )

        return summary.sort_values("Load Balance Efficiency")


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

    metadata = TraceMetadata(
        _format_timedate(elems[0]),
        _format_timing(elems[1]),
        *_split_nodestring(elems[2]),
        _format_num_apps(elems[3]),
        _format_app_list(elems[4])
    )

    return metadata


def get_prv_header_info(prv_file):
    """Get basic run information from the prv file header

    Parameters
    ----------
    prv_file: str
        Path to prv trace file.

    Returns:
    --------
    metadata: NamedTuple
        Named tuple containing the header information.
    """

    with zipopen(prv_file, "rt") as fh:
        return _parse_paraver_headerline(fh.readline().strip())
