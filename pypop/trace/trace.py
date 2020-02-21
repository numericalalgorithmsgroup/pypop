#!/usr/bin/env python3

import warnings

from ..utils.pandas import HDFStoreContext
from ..utils.exceptions import WrongLoaderError

from .tracemetadata import TraceMetadata


class Trace:
    """Base class providing a skeleton for implementation of different trace classes.

    Includes support for summary files to allow fast resumption of analysis using cached
    data.
    """

    _loader_registry = {}

    def __new__(cls, filename, **kwargs):
        """Use a factory paradigm to return relevant subclass instance.

        Results should be agnostic to whether the trace or the summary file is passed,
        and return the appropriate subclass in either case.  Do this by caching the
        loader name in the metadata of the summary file. We can then peek at the
        summaryfile as we load it.
        """
        # If were are in a subclass __new__ then create a subclass instance
        if cls is not Trace:
            return super().__new__(cls)

        try:
            metadata, _ = Trace._parse_summary_file(filename)
        except WrongLoaderError:
            metadata = TraceMetadata()

        # If we have a metadata object, then simply instantiate the correct loader type
        # based on the metadata entry
        if metadata:
            return Trace._loader_registry[metadata[Trace._subclasskey]](
                filename, **kwargs
            )

        # Otherwise iterate to find a suitable loader
        for loader in Trace._loader_registry.values():
            try:
                return loader(filename, **kwargs)
            except WrongLoaderError:
                pass

        raise ValueError("Invalid or unsupported trace {}".format(filename))

    def __init__(self, filename, **kwargs):

        # Save keyword args for later
        self._kwargs = kwargs

        # First check to see if we have been passed a summary file. If so, load from that
        # and skip parsing the whole trace again
        try:
            self._metadata, self._statistics = self._parse_summary_file(filename)
        except WrongLoaderError:
            self._metadata = TraceMetadata()
            self._statistics = None

        # Now set datafile and tracefile based on what we actually are loading from
        if self._metadata:
            # Loading from a summary file so
            self._summaryfile = filename
            self._tracefile = self._metadata.tracefile_name
        else:
            # Loading from a tracefile so
            self._tracefile = filename
            self._summaryfile = None  # Set when we create it
            # And actually load the trace
            self._load_trace()

    def _load_trace(self):
        self._gather_metadata()

        if self._kwargs.get("eager_load", True):
            self._gather_statistics()

    def _gather_metadata(self):
        raise NotImplementedError

    def _gather_statistics(self):
        raise NotImplementedError

    def _ensure_statistics_gathered(self):
        if self._statistics is not None:
            return
        if self._analysis_failed is False:
            try:
                self._gather_statistics()
            except Exception as err:
                self._analysis_failed = str(err)
                raise
        raise RuntimeError(
            "Analysis previously failed ({})".format(self._analysis_failed)
        )

    @property
    def statistics(self):
        if self._stats is None:
            self._gather_statistics()
        return self._statistics

    def write_summary_file(self):
        """Save the trace statistics to a summary file for fast reload next time.
        """
        self._ensure_statistics_gathered()

        summary_filename = ".".join([self._datafile, "pypopsummary"])

        packed_metadata = self._pack_trace_metadata(self.metadata)

        with HDFStoreContext(summary_filename, "w") as hdfstore:
            hdfstore[Trace._metadatakey] = packed_metadata
            hdfstore[Trace._statskey] = self.statistics

    @staticmethod
    def _parse_summary_file(filename):
        """Load from HDF5 for reading
        """
        try:
            with HDFStoreContext(filename, mode="r") as hdfstore:
                try:
                    file_metadata = hdfstore[Trace._metadatakey]
                except KeyError:
                    raise WrongLoaderError(
                        '"{}" is not a PyPOP summary file'.format(filename)
                    )

                if file_metadata["FormatVersion"][0] > Trace._formatversion:
                    warnings.warn(
                        "Trace summary was written with a newer PyPOP version. The "
                        "format is intended to be backward compatible but you may wish "
                        "to upgrade your installed PyPOP version to support all "
                        "features."
                    )

                tracemetadata = Trace._unpack_trace_metadata(file_metadata)
                statistics = hdfstore["TraceStatistics"]
        except ValueError:
            raise WrongLoaderError('"{}" is not a PyPOP summary file'.format(filename))

        return (tracemetadata, statistics)

    @staticmethod
    def _pack_trace_metadata(unpacked):
        raise NotImplementedError

    @staticmethod
    def _unpack_trace_metadata(packed):
        raise NotImplementedError

    @classmethod
    def register_loader(cls):
        if issubclass(cls, Trace) and cls is not Trace:
            Trace._loader_registry[cls.__name__] = cls
            return
        raise TypeError("Can only register subclasses of Trace (passed {})".format(cls))
