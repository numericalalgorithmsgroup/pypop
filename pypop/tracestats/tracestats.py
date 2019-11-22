#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Copyright (c) 2019, The Numerical Algorithms Group, Ltd. All rights reserved.

import os
import pickle

from hashlib import md5

from warnings import warn


class TraceStatsMetadata:
    """Useful metadata about the trace and trace file
    """

    def __init__(
        self, tracefile, nodes, processes, threads_per_process, trace_length, **kwargs
    ):
        metadata = {
            "tracefile": str(tracefile),
            "nodes": int(nodes),
            "processes": int(processes),
            "threads_per_process": int(threads_per_process),
            "trace_length": float(trace_length),
        }

        self._metadata = kwargs
        self._metadata.update(metadata)

    def __getitem__(self, key):
        try:
            return self._metadata[key]
        except KeyError:
            raise KeyError('No metadata matching key "{}"'.format(key))

    def __hash__(self):
        hashstr = "".join(
            ["{}{}".format(k, v) for k, v in sorted(self._metadata.items())]
        )
        return int(md5(hashstr.encode("utf-8")).hexdigest(), base=16)

    @property
    def avail(self):
        """List of available metadata items
        """
        return list(self._metadata.keys())


class TraceStats:
    """Summary statistics for a tracefile.

    This is a superclass provides an interface for the calculation of summary statistics
    for tracefiles.  Note that this is intended to be subclassed to support the various
    tracefile formats produced by different tools.

    Attributes
    ----------
    metadata
    stats
    """

    _subclasses = set()

    def __new__(
        cls,
        tracefile=None,
        chop_to_roi=False,
        ignore_cache=False,
        cache_stats=True,
        **kwargs
    ):
        # If we are making a new TraceStats, instead create an appropriate subclass
        # based on tracefile
        if cls is TraceStats:
            for subclass in TraceStats._subclasses:
                if subclass.can_handle(tracefile):
                    return super().__new__(subclass)

            raise ValueError("Unsupported filetype: {}".format(tracefile))

        # By default just call super().__new__()
        return super().__new__(cls)

    def __init__(self, tracefile, ignore_cache=False, cache_stats=True, **kwargs):
        # First build metadata for this specific summary
        if hasattr(self, "_build_metadata") and hasattr(self, "_analyze_tracefile"):
            self._metadata = self._build_metadata(tracefile, **kwargs)
        else:
            raise NotImplementedError(
                "Error: class {} has no loader implemented"
                "".format(self.__class__.__name__)
            )

        # Now calculate cache hash and see if we have something existing:
        cache_hash = hash(self._metadata)
        cache_path = os.path.join(
            os.path.dirname(tracefile), "{}_stats.pkl".format(cache_hash)
        )

        # If user allows, try to make use of cache first
        if not ignore_cache:
            try:
                with open(cache_path, "rb") as pfh:
                    cached_metadata, cached_stats = pickle.load(pfh)

                    # Double check cache hashes match, and if they do, use cached stats
                    if hash(cached_metadata) == cache_hash:
                        self._stats = cached_stats
                        return

                    # If hash check failed something odd happened, likely a version
                    # mismatch
                    warn(
                        "Stats cache exists but appears to be invalid, this is probably "
                        "because it was created with an older version of PyPOP. "
                        "Statistics will now be recalculated."
                    )

            except ModuleNotFoundError:
                warn(
                    "Stats cache exists but appears to be invalid, this is probably "
                    "because it was created with an older version of PyPOP. "
                    "Statistics will now be recalculated."
                )

            except FileNotFoundError:
                pass

        # If we got to here cache loading failed or was skipped, so analyze tracefile:

        self._stats = self._analyze_tracefile(tracefile, **kwargs)

        # Finally, cache the calculated stats if desired
        if cache_stats:
            with open(cache_path, "wb") as pfh:
                pickle.dump((self._metadata, self.stats), pfh)

    def __hash__(self):
        return hash((self.metadata, self.chopped))

    def _repr_html_(self):
        return self.stats._repr_html_()

    @staticmethod
    def can_handle(tracefile):
        """Return true if class can handle provided tracefile.
        """
        return False

    @staticmethod
    def register_handler(handler):
        """Register a new tracefile handler

        Handlers should inherit from :class:`TraceStats` and implement the
        `_evaluate_trace()` method.

        Parameters
        ----------
        handler: class inheriting from :class:`TraceStats`
            New tracefile handler to register
        """
        if issubclass(handler, TraceStats):
            TraceStats._subclasses.add(handler)
        else:
            raise TypeError(
                "Handlers must inherit from {}".format(TraceStats.__name__)
            )

    @property
    def metadata(self):
        """Tracefile metadata
        """
        return self._metadata

    @property
    def stats(self):
        return self._stats

    @stats.setter
    def stats_setter(self, new_stats):
        warn(
            "Modifying the stats dict by hand is not supported and may lead to errors,"
            "incorrect results and other undefined behaviour."
        )
        self._stats = new_stats
