#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

"""\
PRV Trace Loader
----------------

Support for directly loading PRV trace data as Pandas dataframes
"""

import pandas
import numpy

from hashlib import sha1
from warnings import warn
from collections import OrderedDict
from csv import writer as csvwriter
from io import StringIO

from pypop.utils.io import zipopen
from pypop.utils.pandas import HDFStoreContext
from pypop.utils.exceptions import WrongLoaderError

from pypop.trace.tracemetadata import TraceMetadata

try:
    from tqdm.auto import tqdm
except ImportError:
    from .utils import return_first_arg as tqdm

import pandas as pd
import numpy as np

# Event states - hopefully these will not change(!)
K_STATE_RUNNING = "1"
K_EVENT_OMP_PARALLEL = 60000001
K_EVENT_OMP_LOOP_FUNCTION = 60000018
K_EVENT_OMP_TASK_FUNCTION = 60000023
K_EVENT_OMP_LOOP_FILE_AND_LINE = 60000118
K_EVENT_OMP_TASK_FILE_AND_LINE = 60000123


class PRV(object):

    _metadatakey = "/PyPOPTraceMetadata"
    _statekey = "/PyPOPPRVStates"
    _eventkey = "/PyPOPPRVEvents"
    _commkey = "/PyPOPPRVComms"
    _eventnamekey = "/PyPOPPRVEventNames"
    _eventvaluekey = "/PyPOPPRVEventVals_"
    _ompregionkey = "/PyPOPPRVOMPRegionDetail"

    _formatversionkey = "/PyPOPPRVBinaryTraceFormatVersion"
    _formatversion = 1

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
            "value": np.int,
        },
        "comm": None,
    }

    def __init__(self, prv_path, lazy_load=False, ignore_cache=False):

        self._prv_path = prv_path

        self.metadata = TraceMetadata(self)

        self._omp_region_data = None
        self.event_names = {}
        self.event_vals = {}

        # These go deliberately unset for JIT loading with __getattr__
        # self.state =
        # self.event =
        # self.comm =

        try:
            if ignore_cache:
                raise FileNotFoundError("Ignoring Cache")
            self._load_binarycache()
        except (ValueError, FileNotFoundError):
            try:
                self._parse_pcf()
                if not lazy_load:
                    self._parse_prv()
            except (ValueError, WrongLoaderError):
                raise ValueError("Not a valid prv or binary cache file")

    def __getattr__(self, attr):
        """Need to intercept attempts to access lazily-loaded data objects and ensure
        they are loaded just-in-time
        """

        if attr in PRV.record_types.values():
            self._parse_prv()

        return super().__getattribute__(attr)

    @property
    def omp_region_data(self):
        if self._omp_region_data is None:
            self.profile_openmp_regions(no_progress=True)

        return self._omp_region_data

    @staticmethod
    def _generate_binaryfile_name(prvfile):
        return prvfile + ".bincache"

    def reload(self):
        if self._no_prv:
            warn("Data loaded directly from cachefile, reload from PRV not possible")
            return

        self._parse_pcf()
        self._parse_prv()

    def _parse_pcf(self):

        self.metadata = PRV._populate_metadata(self._prv_path, self.metadata)

        if self._prv_path.endswith(".gz"):
            pcf_path = "".join([self._prv_path[:-6], "pcf"])
        else:
            pcf_path = "".join([self._prv_path[:-3], "pcf"])

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
                        eventkey = int(linevals[1])
                        self.event_names[eventkey] = linevals[2]
                        self.event_vals[eventkey] = {}
                        continue

                    if block_mode == "VALUES":
                        linevals = line.strip().split(maxsplit=1)
                        valuekey = int(linevals[0])
                        self.event_vals[eventkey][valuekey] = linevals[1]

        except FileNotFoundError:
            raise FileNotFoundError("No PCF file accompanying PRV - cannot continue")

    def add_interpreted_names(self):

        if "Event Name" not in self.event:
            self.event["Event Name"] = self.event["event"].map(self.event_names)
        if "Event Value" not in self.event:
            self.event["Event Value"] = self.event.apply(
                lambda x: self.event_vals.get(x["event"], {}).get(
                    x["value"], str(x["value"])
                ),
                axis="columns",
            )

    def _parse_prv(self):

        self.state = None
        self.event = None
        self.comm = None

        temp_data = [StringIO() for _ in range(len(PRV.record_types))]
        temp_writers = [csvwriter(x) for x in temp_data]

        line_processors = {
            k: getattr(self, "_process_{}line".format(v))
            for k, v in PRV.record_types.items()
        }

        with zipopen(self._prv_path, "rt") as prv_fh:
            headerline = next(prv_fh)
            self.metadata = PRV._populate_metadata(headerline, self.metadata)

            # Skip the communicator lines for now
            try:
                skiplines = int(headerline[headerline.rfind("),") + 2 :])
            except ValueError:
                skiplines = 0
            for i in range(skiplines):
                next(prv_fh)

            linecounter = 0
            for line in prv_fh:
                # Skip comment lines
                if line.startswith("#"):
                    continue
                line = [int(x) for x in line.split(":")]
                line_processors[line[0]](line, temp_writers[line[0] - 1])
                linecounter += 1
                if linecounter % 1000000 == 0:
                    self._flush_prv_temp_data(temp_data)
                    temp_data = [StringIO() for _ in range(len(PRV.record_types))]
                    temp_writers = [csvwriter(x) for x in temp_data]
            self._flush_prv_temp_data(temp_data)

        for iattr, attrname in PRV.record_types.items():
            if getattr(self, attrname) is not None and not getattr(self, attrname).empty:
                getattr(self, attrname).set_index(
                    ["task", "thread", "time"], inplace=True
                )
                getattr(self, attrname).sort_index(inplace=True)

        self._write_binarycache()

    def _flush_prv_temp_data(self, temp_data):
        for iattr, attrname in PRV.record_types.items():
            temp_data[iattr - 1].seek(0)
            try:
                newdata = pd.read_csv(
                    temp_data[iattr - 1],
                    names=PRV.colnames[attrname],
                    dtype=PRV.coltypes[attrname],
                    engine="c",
                )
            except pd.errors.EmptyDataError:
                continue
            setattr(
                self, attrname, pd.concat([getattr(self, attrname), newdata]),
            )

    def _process_stateline(self, line, writer):
        writer.writerow([line[3], line[4], line[1], line[5], line[6], line[7]])

    def _process_eventline(self, line, writer):
        row_start = [line[3], line[4], line[1], line[5]]
        nrows = (len(line) - 6) // 2
        rows = [row_start + line[6 + 2 * x : 8 + 2 * x] for x in range(nrows)]
        writer.writerows(rows)

    def _process_commline(self, line, writer):
        pass

    def _write_binarycache(self):
        binaryfile = self._generate_binaryfile_name(self._prv_path)

        packed_metadata = self.metadata.pack_dataframe()

        packed_metadata[PRV._formatversionkey] = pandas.Series(
            data=PRV._formatversion, dtype=numpy.int32
        )

        with HDFStoreContext(binaryfile, mode="w") as hdfstore:
            hdfstore.put(self._metadatakey, packed_metadata, format="t")
            if self.state is not None and not self.state.empty:
                hdfstore.put(self._statekey, self.state, format="t", complib="blosc")
            if self.event is not None and not self.event.empty:
                hdfstore.put(self._eventkey, self.event, format="t", complib="blosc")
            if self.comm is not None and not self.comm.empty:
                hdfstore.put(self._commkey, self.comm, format="t", complib="blosc")
            if self._omp_region_data is not None and not self._omp_region_data.empty:
                hdfstore.put(
                    self._ompregionkey,
                    self._omp_region_data,
                    format="t",
                    complib="blosc",
                )

            hdfstore.put(self._eventnamekey, pd.Series(self.event_names), format="t")

            for evtkey, evtvals in self.event_vals.items():
                hdfstore.put(
                    "".join([self._eventvaluekey, str(evtkey)]),
                    pd.Series(evtvals),
                    format="t",
                )

    def _load_binarycache(self):

        try:
            self._read_binarycache(self._prv_path)
            self._no_prv = True
        except:
            # This will raise either ValueError or FileNotFoundError to be trapped and
            # handled by calling function
            self._read_binarycache(self._generate_binaryfile_name(self._prv_path))

    def _read_binarycache(self, filename):

        # HDFStoreContext will raise perfectly sensible errors, no need to trap
        with HDFStoreContext(filename, mode="r") as hdfstore:
            try:
                file_metadata = hdfstore[PRV._metadatakey]
                format_version = file_metadata[PRV._formatversionkey][0]
            except KeyError:
                raise ValueError("{} is not a binary event store".format(filename))

            if format_version > PRV._formatversion:
                warn(
                    "Trace data was written with a newer PyPOP version. The "
                    "format is intended to be backward compatible but you may wish "
                    "to upgrade your installed PyPOP version to support all "
                    "features."
                )

            try:
                self.metadata = TraceMetadata.unpack_dataframe(file_metadata)
                self.event_names = hdfstore[PRV._eventnamekey].to_dict()
                self.event_vals = {}
                for evtkey in (
                    x for x in hdfstore.keys() if x.startswith(PRV._eventvaluekey)
                ):
                    intkey = int(evtkey.replace(PRV._eventvaluekey, ""))
                    self.event_vals[intkey] = hdfstore[evtkey].to_dict()
            except KeyError:
                raise ValueError("{} corrupted binary event cache")

            try:
                self.state = hdfstore[PRV._statekey]
            except KeyError:
                self.state = None
            try:
                self.event = hdfstore[PRV._eventkey]
            except KeyError:
                self.event = None
            try:
                self.comm = hdfstore[PRV._commkey]
            except KeyError:
                self.comm = None
            try:
                self._omp_region_data = hdfstore[self._ompregionkey]
            except KeyError:
                self._omp_region_data = None

    def profile_openmp_regions(self, no_progress=False, ignore_cache=False):
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
            total=self.metadata.num_processes,
            disable=no_progress,
            leave=None,
        ):

            # OpenMP events mark region start/end on master thread
            thread_events = rank_events.droplevel("task")
            if not np.alltrue(np.diff(thread_events.index) >= 0):
                raise ValueError("Event timings are non-monotonic")

            region_starts = thread_events.index[thread_events["value"] != 0]
            region_ends = thread_events.index[thread_events["value"] == 0]

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

            # Now sanity check regions and try to repair issues caused by missing events:
            # Cut traces seem to have an extra end event injected by the cutter trying to
            # be "helpful" If this seems to be the case try trimming the end and post a
            # warning
            if len(region_ends) == len(region_starts) + 1:
                region_ends = region_ends[:-1]
                warn("Attempting to trim spurious events - cutter inserted?")

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
            # Remove negative categories codes -> events not in any region interval
            cleaned_rfuncs = rank_funcs.droplevel(("task", "thread"))[funcbins >= 0]
            cleaned_fbins = funcbins[funcbins >= 0]
            region_funcs = cleaned_rfuncs.groupby(cleaned_fbins)
            region_fingerprints_func = region_funcs.apply(
                lambda x: ":".join(
                    [
                        self.omp_function_by_value(int(y))
                        for y in x["value"].unique()
                    ]
                )
            )

            funclocbins = pd.cut(
                rank_func_locs.droplevel(("task", "thread")).index, regionintervals
            ).codes
            cleaned_rflocs = rank_func_locs.droplevel(("task", "thread"))[
                funclocbins >= 0
            ]
            cleaned_flbins = funclocbins[funclocbins >= 0]
            region_func_locs = cleaned_rflocs.groupby(cleaned_flbins)
            region_fingerprints_loc = region_func_locs.apply(
                lambda x: ":".join(
                    [
                        self.omp_location_by_value(int(y))
                        for y in x["value"].unique()
                    ]
                )
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
                    "Rank": np.full(region_starts.shape, irank),
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
        if self.event_vals:
            fpvals = fingerprint.split(":")
            fpstrings = [self.omp_location_by_value(fpk, "MISSINGVAL") for fpk in fpvals]
            return ":".join(fpstrings)

        return fingerprint

    def omp_location_by_value(self, value, default="MISSINGVAL"):
        value_dict = {
            **self.event_vals.get(K_EVENT_OMP_TASK_FILE_AND_LINE, {}),
            **self.event_vals.get(K_EVENT_OMP_LOOP_FILE_AND_LINE, {}),
        }

        return value_dict.get(value, default)

    def region_function_from_fingerprint(self, fingerprint):
        if self.event_vals:
            fpvals = fingerprint.split(":")
            fpstrings = [self.omp_function_by_value(fpk, "MISSINGVAL") for fpk in fpvals]
            return ":".join(fpstrings)

        return fingerprint

    def omp_function_by_value(self, value, default="MISSINGVAL"):
        value_dict = {
            **self.event_vals.get(K_EVENT_OMP_TASK_FUNCTION, {}),
            **self.event_vals.get(K_EVENT_OMP_LOOP_FUNCTION, {}),
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
        else:
            fingerprint_key = "Region Function Fingerprint"

        runtime = self.metadata.elapsed_seconds * 1e9
        nproc = len(set(self.omp_region_data["Rank"]))

        self.profile_openmp_regions()

        summary = self.omp_region_data.groupby(fingerprint_key).agg(
            **{
                "Instances": ("Maximum Computation Time", "count"),
                "Total Parallel Inefficiency Contribution": (
                    "Region Delay Time",
                    lambda x: np.sum(x) / (nproc * runtime),
                ),
                "Total Load Imbalance Contribution": (
                    "Computation Delay Time",
                    lambda x: np.sum(x) / (nproc * runtime),
                ),
                "Average Parallel Efficiency": (
                    "Parallel Efficiency",
                    lambda x: np.average(
                        x, weights=self.omp_region_data.loc[x.index, "Region Length"]
                    ),
                ),
                "Average Load Balance": (
                    "Load Balance",
                    lambda x: np.average(
                        x, weights=self.omp_region_data.loc[x.index, "Region Length"],
                    ),
                ),
                "Accumulated Region Time": ("Region Length", np.sum),
                "Accumulated Computation Time": ("Region Total Computation", np.sum),
                "Average Computation Time": ("Average Computation Time", np.average),
                "Maximum Computation Time": ("Maximum Computation Time", np.max),
                "Region Functions": (fingerprint_key, lambda x: x.iloc[0],),
            }
        )

        return summary.sort_values("Total Parallel Inefficiency Contribution")

    @staticmethod
    def _populate_metadata(prv_file, metadata):

        if prv_file.startswith("#Paraver"):
            headerline = prv_file
        else:
            try:
                with zipopen(prv_file, "rt") as fh:
                    headerline = fh.readline().strip()
            except IsADirectoryError:
                raise WrongLoaderError("Not a valid prv file")

            if not headerline.startswith("#Paraver"):
                raise WrongLoaderError("Not a valid prv file")

        elem = headerline.replace(":", ";", 1).split(":", 4)

        metadata.capture_time = elem[0][
            elem[0].find("(") + 1 : elem[0].find(")")
        ].replace(";", ":")

        metadata.elapsed_seconds = float(elem[1][:-3]) * 1e-9

        metadata.num_nodes, metadata.cores_per_node = PRV._split_nodestring(elem[2])

        # elem 3 is the number of applications, currently only support 1
        if int(elem[3]) != 1:
            raise ValueError("Multi-application traces are not supported")

        # elem 4 contains the application layout processes/threads
        (
            metadata.num_processes,
            metadata.threads_per_process,
        ) = PRV._split_layoutstring(elem[4])

        metadata.tracefile_name = prv_file

        hasher = sha1()
        hasher.update(headerline.encode())
        metadata.fingerprint = hasher.hexdigest()

        return metadata

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
        threads = [int(x.split(":")[0]) for x in layoutstring.split(",")]

        return (commsize, threads)
